import json
import os
from collections import defaultdict
from itertools import chain

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import trange


def collect_images(loader, num_images, device):
    collected, got = [], 0
    for batch in loader:
        collected.append(batch)
        got += batch.size(0)
        if got >= num_images:
            break
    return torch.cat(collected, dim=0)[:num_images].to(device)


def train_one_epoch(model, opt_g, opt_d, loader_a, loader_b, criterion_g, criterion_d, device):
    model.train()
    losses_g, losses_d = [], []

    iter_a, iter_b = iter(loader_a), iter(loader_b)
    batches_per_epoch = min(len(loader_a), len(loader_b))

    for _ in trange(batches_per_epoch, leave=False, desc="train"):
        batch_a = next(iter_a).to(device, non_blocking=True)
        batch_b = next(iter_b).to(device, non_blocking=True)

        # G step
        opt_g.zero_grad(set_to_none=True)
        fake_a, fake_b, rec_a, rec_b = model(batch_a, batch_b)
        a_fake_pred_g = model.D_a(fake_a)
        b_fake_pred_g = model.D_b(fake_b)

        # identity forward только если нужен
        if getattr(criterion_g, "lambda_idt", 0.0) > 0:
            idt_a = model.G_ba(batch_a)
            idt_b = model.G_ab(batch_b)
        else:
            idt_a = idt_b = None

        loss_g = criterion_g(batch_a, batch_b, a_fake_pred_g, b_fake_pred_g, rec_a, rec_b, idt_a, idt_b)
        loss_g.backward()
        opt_g.step()

        # D step (с detached fake-ами)
        opt_d.zero_grad(set_to_none=True)
        a_real_pred, b_real_pred, a_fake_pred, b_fake_pred = model.discriminate(
            batch_a, batch_b, fake_a.detach(), fake_b.detach(),
        )
        loss_d = criterion_d(a_real_pred, a_fake_pred, b_real_pred, b_fake_pred)
        loss_d.backward()
        opt_d.step()

        losses_g.append(loss_g.item())
        losses_d.append(loss_d.item())

    return float(np.mean(losses_g)), float(np.mean(losses_d))


@torch.no_grad()
def validate(model, loader_a, loader_b, criterion_d, criterion_g, device):
    model.eval()
    val_data = defaultdict(list)

    iter_a, iter_b = iter(loader_a), iter(loader_b)
    batches_per_epoch = min(len(loader_a), len(loader_b))

    for _ in trange(batches_per_epoch, leave=False, desc="val"):
        batch_a = next(iter_a).to(device, non_blocking=True)
        batch_b = next(iter_b).to(device, non_blocking=True)

        fake_a, fake_b, rec_a, rec_b = model(batch_a, batch_b)
        a_real_pred, b_real_pred, a_fake_pred, b_fake_pred = model.discriminate(
            batch_a, batch_b, fake_a, fake_b,
        )

        if getattr(criterion_g, "lambda_idt", 0.0) > 0:
            idt_a = model.G_ba(batch_a)
            idt_b = model.G_ab(batch_b)
        else:
            idt_a = idt_b = None

        loss_d = criterion_d(a_real_pred, a_fake_pred, b_real_pred, b_fake_pred)
        loss_g = criterion_g(batch_a, batch_b, a_fake_pred, b_fake_pred, rec_a, rec_b, idt_a, idt_b)

        val_data["loss D"].append(loss_d.item())
        val_data["loss G"].append(loss_g.item())

        val_data["real pred A"].extend(a_real_pred.mean(dim=[1, 2, 3]).cpu().tolist())
        val_data["real pred B"].extend(b_real_pred.mean(dim=[1, 2, 3]).cpu().tolist())
        val_data["fake pred A"].extend(a_fake_pred.mean(dim=[1, 2, 3]).cpu().tolist())
        val_data["fake pred B"].extend(b_fake_pred.mean(dim=[1, 2, 3]).cpu().tolist())

    val_data["loss D"] = float(np.mean(val_data["loss D"]))
    val_data["loss G"] = float(np.mean(val_data["loss G"]))
    return val_data


@torch.no_grad()
def save_sample_grid(model, num_images, loader_a, loader_b, de_norm_a, de_norm_b, device, save_path):
    model.eval()
    imgs_a = collect_images(loader_a, num_images, device)
    imgs_b = collect_images(loader_b, num_images, device)

    n = min(imgs_a.size(0), imgs_b.size(0), num_images)
    if n == 0:
        return

    fake_b = model.G_ab(imgs_a)
    fake_a = model.G_ba(imgs_b)
    rec_a = model.G_ba(fake_b)
    rec_b = model.G_ab(fake_a)

    # 2n строк × 3 колонки: сверху ряд A (orig/A->B/rec A), снизу ряд B
    rows = 2 * n
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
    if rows == 1:
        axes = np.array([axes])

    for i in range(n):
        r = i
        axes[r, 0].imshow(de_norm_a(imgs_a[i])); axes[r, 0].set_title("A: original")
        axes[r, 1].imshow(de_norm_b(fake_b[i])); axes[r, 1].set_title("A -> B")
        axes[r, 2].imshow(de_norm_a(rec_a[i])); axes[r, 2].set_title("A recon")
    for i in range(n):
        r = n + i
        axes[r, 0].imshow(de_norm_b(imgs_b[i])); axes[r, 0].set_title("B: original")
        axes[r, 1].imshow(de_norm_a(fake_a[i])); axes[r, 1].set_title("B -> A")
        axes[r, 2].imshow(de_norm_b(rec_b[i])); axes[r, 2].set_title("B recon")

    for ax in axes.flatten():
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_loss_plot(plots, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    xs = np.arange(1, len(plots["train D"]) + 1)
    xs_val = np.arange(1, len(plots["val D"]) + 1)

    axes[0, 0].set_title("Discriminator loss")
    axes[0, 0].plot(xs, plots["train D"], label="train")
    axes[0, 0].plot(xs_val, plots["val D"], label="val")
    axes[0, 0].grid(True); axes[0, 0].legend()

    axes[0, 1].set_title("Generator loss")
    axes[0, 1].plot(xs, plots["train G"], label="train")
    axes[0, 1].plot(xs_val, plots["val G"], label="val")
    axes[0, 1].grid(True); axes[0, 1].legend()

    if plots.get("hist real A") and plots.get("hist gen A"):
        axes[1, 0].hist(plots["hist real A"][-1], bins=50, density=True, label="real", alpha=0.7)
        axes[1, 0].hist(plots["hist gen A"][-1], bins=50, density=True, label="generated", alpha=0.7)
        axes[1, 0].set_title("D_A predictions (last val)")
        axes[1, 0].legend()

    if plots.get("hist real B") and plots.get("hist gen B"):
        axes[1, 1].hist(plots["hist real B"][-1], bins=50, density=True, label="real", alpha=0.7)
        axes[1, 1].hist(plots["hist gen B"][-1], bins=50, density=True, label="generated", alpha=0.7)
        axes[1, 1].set_title("D_B predictions (last val)")
        axes[1, 1].legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def get_lr(optimizer):
    for pg in optimizer.param_groups:
        return pg["lr"]


def save_checkpoint(path, epoch, model, opt_g, opt_d, sched_g, sched_d, plots):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_g_state_dict": opt_g.state_dict(),
            "optimizer_d_state_dict": opt_d.state_dict(),
            "scheduler_g_state_dict": sched_g.state_dict() if sched_g is not None else None,
            "scheduler_d_state_dict": sched_d.state_dict() if sched_d is not None else None,
            "plots": plots,
        },
        path,
    )


def load_checkpoint(path, model, opt_g, opt_d, sched_g=None, sched_d=None, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    opt_g.load_state_dict(ckpt["optimizer_g_state_dict"])
    opt_d.load_state_dict(ckpt["optimizer_d_state_dict"])
    if sched_g is not None and ckpt.get("scheduler_g_state_dict") is not None:
        sched_g.load_state_dict(ckpt["scheduler_g_state_dict"])
    if sched_d is not None and ckpt.get("scheduler_d_state_dict") is not None:
        sched_d.load_state_dict(ckpt["scheduler_d_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("plots", None)


def make_optimizers(model, lr=2e-4, betas=(0.5, 0.999)):
    opt_g = torch.optim.Adam(
        chain(model.G_ab.parameters(), model.G_ba.parameters()), lr=lr, betas=betas,
    )
    opt_d = torch.optim.Adam(
        chain(model.D_a.parameters(), model.D_b.parameters()), lr=lr, betas=betas,
    )
    return opt_g, opt_d


def make_linear_decay_scheduler(optimizer, total_epochs, decay_start_epoch):
    def lr_lambda(epoch):
        return 1.0 - max(0, epoch + 1 - decay_start_epoch) / float(total_epochs - decay_start_epoch + 1)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def learning_loop(
    *,
    model,
    optimizer_g,
    optimizer_d,
    train_loader_a,
    train_loader_b,
    val_loader_a,
    val_loader_b,
    criterion_g,
    criterion_d,
    de_norm_a,
    de_norm_b,
    device,
    run_dir,
    scheduler_g=None,
    scheduler_d=None,
    epochs=200,
    val_every=1,
    images_every=1,
    images_per_validation=10,
    save_every=5,
    plots=None,
    starting_epoch=0,
    log_fn=print,
):
    """
    Кладёт под run_dir:
      - images/epoch_XXXX.png  — история генераций по эпохам
      - chkp/last.pt, chkp/epoch_XXXX.pt  — чекпоинты
      - losses.png             — текущие кривые потерь
      - plots.json             — скалярные метрики по эпохам
    """
    os.makedirs(os.path.join(run_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "chkp"), exist_ok=True)

    if plots is None:
        plots = {
            "train G": [], "train D": [],
            "val G": [], "val D": [],
            "lr G": [], "lr D": [],
            "hist real A": [], "hist gen A": [],
            "hist real B": [], "hist gen B": [],
        }

    for epoch in range(starting_epoch + 1, starting_epoch + epochs + 1):
        plots["lr G"].append(get_lr(optimizer_g))
        plots["lr D"].append(get_lr(optimizer_d))

        loss_g, loss_d = train_one_epoch(
            model, optimizer_g, optimizer_d,
            train_loader_a, train_loader_b,
            criterion_g, criterion_d, device,
        )
        plots["train G"].append(loss_g)
        plots["train D"].append(loss_d)
        log_fn(f"[epoch {epoch}] train G={loss_g:.4f} D={loss_d:.4f} lrG={plots['lr G'][-1]:.2e}")

        if val_every and epoch % val_every == 0:
            val_data = validate(model, val_loader_a, val_loader_b, criterion_d, criterion_g, device)
            plots["val G"].append(val_data["loss G"])
            plots["val D"].append(val_data["loss D"])
            plots["hist real A"].append(val_data["real pred A"])
            plots["hist gen A"].append(val_data["fake pred A"])
            plots["hist real B"].append(val_data["real pred B"])
            plots["hist gen B"].append(val_data["fake pred B"])
            log_fn(f"[epoch {epoch}]   val G={val_data['loss G']:.4f} D={val_data['loss D']:.4f}")

        if images_every and epoch % images_every == 0:
            img_path = os.path.join(run_dir, "images", f"epoch_{epoch:04d}.png")
            save_sample_grid(
                model, images_per_validation,
                val_loader_a, val_loader_b,
                de_norm_a, de_norm_b, device, img_path,
            )

        save_loss_plot(plots, os.path.join(run_dir, "losses.png"))

        # scalar-only snapshot для быстрого глазом-просмотра
        plots_scalar = {
            k: v for k, v in plots.items()
            if k in {"train G", "train D", "val G", "val D", "lr G", "lr D"}
        }
        with open(os.path.join(run_dir, "plots.json"), "w") as f:
            json.dump(plots_scalar, f)

        # schedulers step в конце эпохи
        if scheduler_g is not None:
            scheduler_g.step()
        if scheduler_d is not None:
            scheduler_d.step()

        # чекпоинты
        save_checkpoint(
            os.path.join(run_dir, "chkp", "last.pt"),
            epoch, model, optimizer_g, optimizer_d, scheduler_g, scheduler_d, plots,
        )
        if save_every and epoch % save_every == 0:
            save_checkpoint(
                os.path.join(run_dir, "chkp", f"epoch_{epoch:04d}.pt"),
                epoch, model, optimizer_g, optimizer_d, scheduler_g, scheduler_d, plots,
            )

    return model, optimizer_g, optimizer_d, plots
