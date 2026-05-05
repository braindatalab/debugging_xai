# watermarks/train_all_runner.py
import os, itertools, argparse, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import watermarks.train_cnn as tc  # reuses helpers & training

def job_exists(model_dir, base, split_index, scale, position, invert, seed_val):
    tag = f"{tc.suffix_for(scale, position, invert)}_split{split_index}".lstrip("_")
    ckpt = os.path.join(model_dir, f"cnn_{base}_{tag}_seed{seed_val}.pt")
    return os.path.exists(ckpt), ckpt

def run_group_on_gpu(args_tuple):
    """
    One process pinned to a single GPU runs all bases x seeds for one (split, pos, scale, invert).
    """
    (gpu, artifacts_dir, model_dir, split_index, bases,
     position, scale, invert, seed_indexes,
     batch_size, epochs, lr, wd, momentum, skip_existing) = args_tuple

    report = {"gpu": gpu, "split": split_index, "pos": position, "scale": scale,
              "invert": int(invert), "runs": 0, "skipped": 0, "fail": []}

    try:
        # pre-load loaders per base on this process/GPU
        for base in bases:
            try:
                tr_loader, va_loader = tc.make_loaders(
                    artifacts_dir=artifacts_dir,
                    split_index=split_index,
                    base=base,
                    scale=scale,
                    position=position,
                    invert=invert,
                    batch_size=batch_size,
                    num_workers=4,
                    pin_memory=True
                )
            except FileNotFoundError as e:
                print(f"[GPU {gpu}] [split {split_index}] [{base}] missing artifacts: {e}")
                continue

            for si in seed_indexes:
                seed_val = tc.SEEDS[si]
                exists, ckpt = job_exists(model_dir, base, split_index, scale, position, invert, seed_val)
                if skip_existing and exists:
                    print(f"[GPU {gpu}] skip existing {os.path.basename(ckpt)}")
                    report["skipped"] += 1
                    continue

                try:
                    device = (f"cuda:{gpu}")
                    tag = f"{tc.suffix_for(scale, position, invert)}_split{split_index}".lstrip("_")
                    tc.train_one_seed(
                        seed=seed_val,
                        base=base,
                        tr_loader=tr_loader,
                        va_loader=va_loader,
                        lr=lr,
                        epochs=epochs,
                        wd=wd,
                        momentum=momentum,
                        model_dir=model_dir,
                        tag=tag,
                        device=(tc.torch.device(device) if tc.torch.cuda.is_available() else tc.torch.device("cpu"))
                    )
                    report["runs"] += 1
                except Exception as e:
                    tb = traceback.format_exc()
                    report["fail"].append((base, seed_val, str(e)))
                    print(f"[GPU {gpu}] FAILED base={base} seed={seed_val}: {e}\n{tb}")

    except Exception as e:
        tb = traceback.format_exc()
        report["fail"].append(("__group__", -1, f"{e}\n{tb}"))

    return report

def run_seed_unit_on_gpu(args_tuple):
    """
    Alternative unit: one process runs exactly one (split, pos, scale, invert, base, seed) on a GPU.
    """
    (gpu, artifacts_dir, model_dir, split_index, base,
     position, scale, invert, seed_index,
     batch_size, epochs, lr, wd, momentum, skip_existing) = args_tuple

    seed_val = tc.SEEDS[seed_index]
    exists, ckpt = job_exists(model_dir, base, split_index, scale, position, invert, seed_val)
    if skip_existing and exists:
        return {"gpu": gpu, "runs": 0, "skipped": 1, "fail": []}

    try:
        tr_loader, va_loader = tc.make_loaders(
            artifacts_dir=artifacts_dir,
            split_index=split_index,
            base=base,
            scale=scale,
            position=position,
            invert=invert,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True
        )
    except FileNotFoundError as e:
        return {"gpu": gpu, "runs": 0, "skipped": 0, "fail": [("artifacts", -1, str(e))]}

    try:
        device = (tc.torch.device(f"cuda:{gpu}") if tc.torch.cuda.is_available() else tc.torch.device("cpu"))
        tag = f"{tc.suffix_for(scale, position, invert)}_split{split_index}".lstrip("_")
        tc.train_one_seed(
            seed=seed_val,
            base=base,
            tr_loader=tr_loader,
            va_loader=va_loader,
            lr=lr,
            epochs=epochs,
            wd=wd,
            momentum=momentum,
            model_dir=model_dir,
            tag=tag,
            device=device
        )
        return {"gpu": gpu, "runs": 1, "skipped": 0, "fail": []}
    except Exception as e:
        tb = traceback.format_exc()
        return {"gpu": gpu, "runs": 0, "skipped": 0, "fail": [(base, seed_val, f"{e}\n{tb}") ]}

def main(argv=None):
    ap = argparse.ArgumentParser(description="GPU runner for all splits/seeds/configs.")
    ap.add_argument("--artifacts-dir", type=str, default="./artifacts")
    ap.add_argument("--model-dir", type=str, default="./models")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0], help="GPU indices to use, e.g. --gpus 0 1 2 3")

    ap.add_argument("--splits", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--bases", type=str, nargs="+",
                    choices=["suppressor","confounder","no_watermark"],
                    default=["suppressor","confounder","no_watermark"])
    ap.add_argument("--positions", type=str, nargs="+", choices=["fixed","variable"], default=["fixed","variable"])
    ap.add_argument("--scales", type=str, nargs="+", choices=["zero_one","neg_one_one"], default=["zero_one","neg_one_one"])
    ap.add_argument("--inverts", type=int, nargs="+", default=[0,1])

    ap.add_argument("--seed-indexes", type=int, nargs="+", default=[0,1,2,3,4])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--momentum", type=float, default=0.9)

    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--unit", choices=["group","seed"], default="group",
                    help="group: one proc per (split,pos,scale,invert) running all bases×seeds; "
                         "seed: one proc per (split,pos,scale,invert,base,seed)")
    ap.add_argument("--dry-run", action="store_true")
    args, _ = ap.parse_known_args(argv)

    invert_bools = [bool(i) for i in args.inverts]

    if args.unit == "group":
        # Build (split, position, scale, invert) groups
        groups = list(itertools.product(args.splits, args.positions, args.scales, invert_bools))
        work = []
        for ix, (split, pos, sc, inv) in enumerate(groups):
            gpu = args.gpus[ix % len(args.gpus)]
            work.append((
                gpu, args.artifacts_dir, args.model_dir, split, args.bases,
                pos, sc, inv, args.seed_indexes,
                args.batch_size, args.epochs, args.lr, args.weight_decay, args.momentum, args.skip_existing
            ))

        print(f"Planned groups: {len(work)}  (procs = {min(len(work), len(args.gpus))})")
        if args.dry_run:
            for w in work:
                print(f"[GPU {w[0]}] split={w[3]} pos={w[5]} scale={w[6]} invert={int(w[7])} seeds={w[8]} bases={args.bases}")
            return

        results = []
        with ProcessPoolExecutor(max_workers=len(args.gpus)) as ex:
            futures = [ex.submit(run_group_on_gpu, w) for w in work]
            for fut in as_completed(futures):
                results.append(fut.result())

    else:
        # unit == "seed" : finer parallelism
        seed_jobs = list(itertools.product(args.splits, args.bases, args.positions, args.scales, invert_bools, args.seed_indexes))
        work = []
        for ix, (split, base, pos, sc, inv, si) in enumerate(seed_jobs):
            gpu = args.gpus[ix % len(args.gpus)]
            work.append((
                gpu, args.artifacts_dir, args.model_dir, split, base,
                pos, sc, inv, si,
                args.batch_size, args.epochs, args.lr, args.weight_decay, args.momentum, args.skip_existing
            ))

        print(f"Planned seed-jobs: {len(work)}  (procs = {min(len(work), len(args.gpus))})")
        if args.dry_run:
            for w in work[:min(20, len(work))]:
                print(f"[GPU {w[0]}] split={w[3]} base={w[4]} pos={w[5]} scale={w[6]} invert={int(w[7])} seed={w[8]}")
            if len(work) > 20: print("... (truncated)")
            return

        results = []
        with ProcessPoolExecutor(max_workers=len(args.gpus)) as ex:
            futures = [ex.submit(run_seed_unit_on_gpu, w) for w in work]
            for fut in as_completed(futures):
                results.append(fut.result())

    # Summarize
    total_runs = sum(r.get("runs", 0) for r in results)
    total_skips = sum(r.get("skipped", 0) for r in results)
    total_fail = sum(len(r.get("fail", [])) for r in results)
    print(f"\nDone. runs={total_runs}  skipped={total_skips}  failures={total_fail}")
    if total_fail:
        for r in results:
            for f in r.get("fail", []):
                print("FAIL:", r.get("gpu"), f)

if __name__ == "__main__":
    main()
