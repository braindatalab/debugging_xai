# watermarks/run_all.py
import os
import itertools
import argparse
import traceback

import numpy as np
import torch

import watermarks.train_watermarks as tc  # uses make_loaders, train_one_seed, suffix_for, SEEDS


def bools_from_ints(ints):
    return [bool(int(x)) for x in ints]


def job_exists(model_dir, base, split_index, scale, position, invert, seed_val):
    tag = f"{tc.suffix_for(scale, position, invert)}_split{split_index}".lstrip("_")
    ckpt = os.path.join(model_dir, f"cnn_{base}_{tag}_seed{seed_val}.pt")
    return os.path.exists(ckpt), ckpt


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run all splits/seeds over config grid.")
    ap.add_argument("--artifacts-dir", type=str, default="./artifacts")
    ap.add_argument("--model-dir", type=str, default="./models")

    ap.add_argument("--splits", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--bases", type=str, nargs="+",
                    choices=["suppressor", "confounder", "no_watermark"],
                    default=["suppressor", "confounder", "no_watermark"])
    ap.add_argument("--positions", type=str, nargs="+",
                    choices=["fixed", "variable"], default=["fixed", "variable"])
    ap.add_argument("--scales", type=str, nargs="+",
                    choices=["zero_one", "neg_one_one"], default=["zero_one", "neg_one_one"])
    ap.add_argument("--inverts", type=int, nargs="+", default=[0, 1], help="0=non-inverted, 1=inverted")

    ap.add_argument("--seed-indexes", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="Indexes into train_cnn.SEEDS for model init")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--momentum", type=float, default=0.9)

    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip training if checkpoint already exists")
    ap.add_argument("--dry-run", action="store_true", help="Print plan only")
    args, _ = ap.parse_known_args(argv)

    invert_bools = bools_from_ints(args.inverts)
    grid = list(itertools.product(args.splits, args.bases, args.positions, args.scales, invert_bools))

    print(f"Planned jobs (per seed): {len(grid)}")
    total_runs = 0
    skipped = 0
    failed_groups = 0

    # Optional: reduce CPU threads on laptops
    try:
        torch.set_num_threads(max(1, torch.get_num_threads() // 2))
    except Exception:
        pass

    for split_index, base, position, scale, invert in grid:
        tag = f"{tc.suffix_for(scale, position, invert)}_split{split_index}".lstrip("_")
        print(f"\n=== [{base}] split={split_index} pos={position} scale={scale} invert={int(invert)} :: {tag} ===")

        # Pre-load loaders once per (split, base, config)
        try:
            if args.dry_run:
                # Touch dataset paths to confirm existence
                _ = tc.dataset_path(args.artifacts_dir, split_index, base, "train", scale, position, invert)
                _ = tc.dataset_path(args.artifacts_dir, split_index, base, "val",   scale, position, invert)
                print("  [OK] artifacts present (train/val)")
                tr_loader = va_loader = None
            else:
                tr_loader, va_loader = tc.make_loaders(
                    artifacts_dir=args.artifacts_dir,
                    split_index=split_index,
                    base=base,
                    scale=scale,
                    position=position,
                    invert=invert,
                    batch_size=args.batch_size
                )
        except FileNotFoundError as e:
            print(f"  [SKIP GROUP] Missing artifacts for this configuration: {e}")
            failed_groups += 1
            continue
        except Exception as e:
            print(f"  [SKIP GROUP] Failed to prepare loaders: {e}")
            traceback.print_exc()
            failed_groups += 1
            continue

        for si in args.seed_indexes:
            seed_val = tc.SEEDS[si]
            exists, ckpt = job_exists(args.model_dir, base, split_index, scale, position, invert, seed_val)
            if args.skip_existing and exists:
                print(f"  [SKIP] seed={seed_val}  checkpoint exists: {os.path.basename(ckpt)}")
                skipped += 1
                continue

            if args.dry_run:
                print(f"  [PLAN] seed={seed_val} -> {os.path.basename(ckpt)}")
                total_runs += 1
                continue

            try:
                tc.train_one_seed(
                    seed=seed_val,
                    base=base,
                    tr_loader=tr_loader,
                    va_loader=va_loader,
                    lr=args.lr,
                    epochs=args.epochs,
                    wd=args.weight_decay,
                    momentum=args.momentum,
                    model_dir=args.model_dir,
                    tag=tag
                )
                total_runs += 1
            except Exception as e:
                print(f"  [FAIL] seed={seed_val}: {e}")
                traceback.print_exc()

    print(f"\nDone. Groups failed: {failed_groups} | runs executed: {total_runs} | skipped (existing): {skipped}")


if __name__ == "__main__":
    main()
