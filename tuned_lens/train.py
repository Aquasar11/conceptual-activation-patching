import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import TunedLensConfig
from lens import TunedLens
from loss import tuned_lens_loss_layer
from model_utils import load_model, get_model_outputs
from data import build_dataloaders
from evaluate import evaluate


def train(config: TunedLensConfig):
    os.makedirs(config.output_dir, exist_ok=True)
    dtype = getattr(torch, config.dtype)
    device = config.device

    writer = SummaryWriter(log_dir=config.tensorboard_dir)

    print(f"Loading model: {config.model_name}")
    model, unembed_weight, hidden_dim, final_norm = load_model(config)

    lens = TunedLens(hidden_dim=hidden_dim, layer_indices=config.layers, final_norm=final_norm).to(device)
    optimizer = torch.optim.AdamW(lens.parameters(), lr=config.learning_rate)

    # Compile the hot-path functions — fuses matmul chain, log_softmax, and kl_div
    # into fewer Triton kernel launches. One-time compile cost ~2-5 min on first step.
    compiled_forward = torch.compile(lens.forward_layer, mode="reduce-overhead", dynamic=True)
    compiled_loss    = torch.compile(tuned_lens_loss_layer, mode="reduce-overhead", dynamic=True)

    print("Building dataloaders (tokenizing dataset)...")
    train_loader, val_loader = build_dataloaders(config)

    global_step = 0
    best_val_kld = float("inf")
    L = len(config.layers)

    # GPU accumulators — avoids CPU-GPU sync on every step
    loss_accum = torch.zeros(1, device=device)
    kld_accum  = torch.zeros(L, device=device)
    reg_accum  = torch.zeros(L, device=device)

    for epoch in range(config.num_epochs):
        lens.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}", dynamic_ncols=True)
        for input_ids in pbar:
            input_ids = input_ids.to(device, non_blocking=True)

            log_P_model, H = get_model_outputs(model, input_ids, config.layers, dtype)

            # Per-step GPU tensors — allocated once per step, no sync inside the loop
            optimizer.zero_grad()
            loss_step = torch.zeros(1, device=device)
            kld_step  = torch.zeros(L, device=device)
            reg_step  = torch.zeros(L, device=device)

            for i in range(L):
                # Autocast: H[i] is bfloat16, W/b are float32 — autocast handles the
                # mixed-dtype matmuls using bfloat16 tensor cores (312 vs 77.6 TFLOPS)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits_l = compiled_forward(H[i], unembed_weight, i)
                loss_l, kld_l, reg_l = compiled_loss(
                    logits_l, log_P_model, lens.W[i], lens.b[i], config.lambda_reg, hidden_dim
                )
                loss_l.backward()
                loss_step   += loss_l.detach()   # GPU accumulation — no sync
                kld_step[i]  = kld_l.detach()   # GPU assignment  — no sync
                reg_step[i]  = reg_l.detach()   # GPU assignment  — no sync

            optimizer.step()

            # ONE GPU→CPU sync per step (scalar needed for progress bar)
            total_loss_val = loss_step.item()
            loss_accum += loss_step
            kld_accum  += kld_step   # GPU accumulation — no sync
            reg_accum  += reg_step   # GPU accumulation — no sync
            global_step += 1

            pbar.set_postfix(loss=f"{total_loss_val:.4f}", step=global_step)

            if global_step % config.log_every == 0:
                # 3 syncs every log_every steps for logging (vs 33 syncs every step before)
                avg_loss = loss_accum.item() / config.log_every
                kld_log  = kld_accum.cpu()   / config.log_every
                reg_log  = reg_accum.cpu()   / config.log_every
                print(f"Epoch {epoch+1} | Step {global_step} | loss {avg_loss:.4f}")

                writer.add_scalar("train/total_loss", avg_loss, global_step)
                for i, l in enumerate(config.layers):
                    writer.add_scalar(f"train/layer_{l:02d}_kld", kld_log[i].item(), global_step)
                    writer.add_scalar(f"train/layer_{l:02d}_reg", reg_log[i].item(), global_step)

                loss_accum.zero_()
                kld_accum.zero_()
                reg_accum.zero_()

        if config.eval_every_epoch:
            print(f"\nEvaluating after epoch {epoch+1}...")
            results = evaluate(model, lens, val_loader, config, unembed_weight, hidden_dim)

            mean_val_kld = results["kld"].mean().item()

            print("Per-layer validation results:")
            for i, l in enumerate(config.layers):
                print(
                    f"  Layer {l:2d}: "
                    f"KLD={results['kld'][i].item():.4f}  "
                    f"CE={results['ce'][i].item():.4f}  "
                    f"Top1={results['top1'][i].item():.3f}  "
                    f"Top5={results['top5'][i].item():.3f}"
                )
            print(f"  Mean KLD: {mean_val_kld:.4f}")

            writer.add_scalar("val/mean_kld", mean_val_kld, global_step)
            for i, l in enumerate(config.layers):
                writer.add_scalar(f"val/layer_{l:02d}_kld",  results["kld"][i].item(),  global_step)
                writer.add_scalar(f"val/layer_{l:02d}_ce",   results["ce"][i].item(),   global_step)
                writer.add_scalar(f"val/layer_{l:02d}_top1", results["top1"][i].item(), global_step)
                writer.add_scalar(f"val/layer_{l:02d}_top5", results["top5"][i].item(), global_step)

            if mean_val_kld < best_val_kld:
                best_val_kld = mean_val_kld
                lens.save_layers(config.output_dir)
                print(f"  New best val KLD {best_val_kld:.4f} — saved layer checkpoints to {config.output_dir}")
            print()

            lens.train()

    writer.close()
    print(f"Training complete. Best val KLD: {best_val_kld:.4f}. Checkpoints in: {config.output_dir}")
