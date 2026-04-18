"""
Запуск обучения CycleGAN на одной конкретной GPU.

Примеры:
  # один процесс на cuda:0
  python run.py --gpu 0 --run-name exp_baseline --epochs 200

Каждый прогон пишет всё под runs/<run_name>/:
  images/epoch_XXXX.png,
  chkp/last.pt, chkp/epoch_XXXX.pt
  losses.png, plots.json, log.txt, config.json
"""
import argparse
import json
import os
import sys
import time

import torch

from data import build_dataloaders, build_datasets, get_transforms
from download import download_and_preview
from losses import FullDiscriminatorLoss, FullGeneratorLoss
from models import CycleGAN
from training import (
    learning_loop,
    load_checkpoint,
    make_linear_decay_scheduler,
    make_optimizers,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, required=True, help="(cuda:<gpu>)")
    p.add_argument("--run-name", type=str, required=True, help="runs/<name>")
    p.add_argument("--runs-root", type=str, default="runs")
    p.add_argument("--dataset", type=str, default="ukiyoe2photo")
    p.add_argument("--dataset-root", type=str, default="data")

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--decay-start", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)  # больше - плохо
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda-cyc", type=float, default=10.0)
    p.add_argument("--lambda-idt", type=float, default=0.0, help="baseline: 0.5 * lambda_cyc)")
    p.add_argument("--pool-size", type=int, default=50, help="image pool size")
    p.add_argument("--is-mse", action="store_true", default=True, help="LSGAN (MSE) вместо BCE")
    p.add_argument("--no-mse", dest="is_mse", action="store_false")

    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--images-every", type=int, default=5)
    p.add_argument("--images-per-validation", type=int, default=10)
    p.add_argument("--save-every", type=int, default=5)

    p.add_argument("--resume", type=str, default=None, help="chpt_path")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def setup_device(gpu_idx):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    n = torch.cuda.device_count()
    if gpu_idx >= n:
        raise ValueError(f"Currently {n} GPUs awailable")
    device = torch.device(f"cuda:{gpu_idx}")
    torch.cuda.set_device(device)
    return device


def make_logger(log_path):
    f = open(log_path, "a", buffering=1)

    def log(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        f.write(line + "\n")

    return log


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = setup_device(args.gpu)

    run_dir = os.path.join(args.runs_root, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    log = make_logger(os.path.join(run_dir, "log.txt"))

    log(f"run_dir: {run_dir}")
    log(f"device: {device}  |  pid: {os.getpid()}")
    log(f"args: {vars(args)}")
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # данные
    target_folder = os.path.join(args.dataset_root, args.dataset)
    if not os.path.exists(target_folder):
        log(f"downloading dataset {args.dataset} to {target_folder}")
        target_folder = download_and_preview(
            args.dataset, dataset_folder=args.dataset_root, preview=False,
        )
    else:
        log(f"dataset already present at {target_folder}")

    # модельные константы как в статье CycleGAN
    UPSCALE_SIZE = 286
    CROP_SIZE = 256
    NGF = 64
    NDF = 64
    N_RES_BLOCKS = 9

    # нормализация в [-1, 1] под Tanh
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    train_tf_a, val_tf_a, de_norm_a = get_transforms(
        norm_mean, norm_std, upscale_size=UPSCALE_SIZE, crop_size=CROP_SIZE,
    )
    train_tf_b, val_tf_b, de_norm_b = get_transforms(
        norm_mean, norm_std, upscale_size=UPSCALE_SIZE, crop_size=CROP_SIZE,
    )

    ds = build_datasets(target_folder, train_tf_a, val_tf_a, train_tf_b, val_tf_b)
    loaders = build_dataloaders(ds, batch_size=args.batch_size, num_workers=args.num_workers)
    log(f"sizes: train_a={len(ds.train_a)} train_b={len(ds.train_b)} "
        f"test_a={len(ds.test_a)} test_b={len(ds.test_b)}")

    model = CycleGAN(ngf=NGF, ndf=NDF, n_res_blocks=N_RES_BLOCKS).to(device)

    opt_g, opt_d = make_optimizers(model, lr=args.lr)
    sched_g = make_linear_decay_scheduler(opt_g, args.epochs, args.decay_start)
    sched_d = make_linear_decay_scheduler(opt_d, args.epochs, args.decay_start)

    criterion_d = FullDiscriminatorLoss(is_mse=args.is_mse)
    criterion_g = FullGeneratorLoss(
        lambda_cyc=args.lambda_cyc,
        lambda_idt=args.lambda_idt,
        is_mse=args.is_mse,
    )

    # resume
    starting_epoch = 0
    plots = None
    if args.resume is not None:
        log(f"resuming from {args.resume}")
        starting_epoch, plots = load_checkpoint(
            args.resume, model, opt_g, opt_d, sched_g, sched_d, map_location=device,
        )
        log(f"resumed at epoch {starting_epoch}")

    log("starting training loop")
    learning_loop(
        model=model,
        optimizer_g=opt_g,
        optimizer_d=opt_d,
        train_loader_a=loaders.train_a,
        train_loader_b=loaders.train_b,
        val_loader_a=loaders.test_a,
        val_loader_b=loaders.test_b,
        criterion_g=criterion_g,
        criterion_d=criterion_d,
        de_norm_a=de_norm_a,
        de_norm_b=de_norm_b,
        device=device,
        run_dir=run_dir,
        scheduler_g=sched_g,
        scheduler_d=sched_d,
        epochs=args.epochs,
        val_every=args.val_every,
        images_every=args.images_every,
        images_per_validation=args.images_per_validation,
        save_every=args.save_every,
        plots=plots,
        starting_epoch=starting_epoch,
        log_fn=log,
        pool_size=args.pool_size,
    )

    log("done")


if __name__ == "__main__":
    sys.exit(main())
