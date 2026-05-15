import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import TunedLensConfig
from lens import TunedLens
from loss import tuned_lens_loss
from model_utils import load_model, get_model_outputs
from data import build_dataloaders
from evaluate import evaluate


def train(config: TunedLensConfig):
    os.makedirs(config.output_dir, exist_ok=True)
    dtype = getattr(torch, config.dtype)
    device = config.device

    writer = SummaryWriter(log_dir=config.tensorboard_dir)

    print(f"Loading model: {config.model_name}")
    model, unembed_weight, hidden_dim = load_model(config)

    lens = TunedLens(hidden_dim=hidden_dim, layer_indices=config.layers).to(device)
    optimizer = torch.optim.AdamW(lens.parameters(), lr=config.learning_rate)

    train_loader, val_loader = build_dataloaders(config)

    global_step = 0
    best_val_kld = float("inf")

    # Running accumulators — reset every log_every steps
    running_total_loss = 0.0
    running_kld = torch.zeros(len(config.layers))
    running_reg = torch.zeros(len(config.layers))

    for epoch in range(config.num_epochs):
        lens.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}", dynamic_ncols=True)
        for input_ids in pbar:
            input_ids = input_ids.to(device, non_blocking=True)

            log_P_model, H = get_model_outputs(model, input_ids, config.layers, dtype)

            logits_all = lens(H, unembed_weight)  # (L, B, S-1, V)

            total_loss, kld_per_layer, reg_per_layer = tuned_lens_loss(
                logits_all, log_P_model, lens.W, lens.b, config.lambda_reg, hidden_dim
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_total_loss += total_loss.item()
            running_kld += kld_per_layer.detach().cpu()
            running_reg += reg_per_layer.detach().cpu()
            global_step += 1

            pbar.set_postfix(loss=f"{total_loss.item():.4f}", step=global_step)

            if global_step % config.log_every == 0:
                avg_loss = running_total_loss / config.log_every
                print(f"Epoch {epoch+1} | Step {global_step} | loss {avg_loss:.4f}")

                writer.add_scalar("train/total_loss", avg_loss, global_step)
                for i, l in enumerate(config.layers):
                    writer.add_scalar(f"train/layer_{l:02d}_kld", running_kld[i].item() / config.log_every, global_step)
                    writer.add_scalar(f"train/layer_{l:02d}_reg", running_reg[i].item() / config.log_every, global_step)

                running_total_loss = 0.0
                running_kld.zero_()
                running_reg.zero_()

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
