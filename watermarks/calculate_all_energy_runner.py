# watermarks/calculate_all_energy_runner.py  (replace)
import os, argparse, itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

def suffix_for(scale, position, invert):
    s = ""
    if scale == "neg_one_one": s += "_rescaled"
    if position == "variable": s += "_variablepos"
    if invert: s += "_inverted"
    return s

def outputs_exist(energies_dir, explanations_dir, split, seed_value, scale, position, invert):
    suf = suffix_for(scale, position, invert)
    seed_tag = f"seed{seed_value}"
    bases = ["water_conf","water_sup","water_no","no_water_conf","no_water_sup","no_water_no"]
    expect = []
    for b in bases:
        expect += [
            os.path.join(energies_dir,     f"energy_{b}_pred{suf}_split{split}_{seed_tag}.pickle"),
            os.path.join(explanations_dir, f"explanations_{b}_pred{suf}_split{split}_{seed_tag}.pickle"),
        ]
    return all(os.path.exists(p) for p in expect)

def run_job(t):
    (gpu, split, seed_idx, artifacts_dir, models_dir, energies_dir, explanations_dir,
     position, scale, invert, limit, skip_existing, seed_value,
     mask_mode, watermark, alpha_thresh) = t

    if skip_existing and outputs_exist(energies_dir, explanations_dir, split, seed_value, scale, position, invert):
        return {"gpu": gpu, "split": split, "seed": seed_idx, "status": "skipped"}

    try:
        from watermarks.calculate_energy import main as explain_main
        argv = [
            "--split-index", str(split),
            "--seed-index", str(seed_idx),
            "--position", position,
            "--scale", scale,
            "--invert", str(int(invert)),
            "--gpu", str(gpu),
            "--artifacts-dir", artifacts_dir,
            "--models-dir", models_dir,
            "--energies-dir", energies_dir,
            "--explanations-dir", explanations_dir,
            "--mask-mode", mask_mode,
        ]
        if mask_mode == "derive-fixed":
            argv += ["--watermark", watermark, "--alpha-thresh", str(alpha_thresh)]
        if limit is not None:
            argv += ["--limit", str(limit)]
        explain_main(argv)
        return {"gpu": gpu, "split": split, "seed": seed_idx, "status": "ok"}
    except Exception as e:
        return {"gpu": gpu, "split": split, "seed": seed_idx, "status": "fail", "err": str(e)}

def main():
    ap = argparse.ArgumentParser(description="Multi-GPU runner for explanation energies")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--splits", type=int, nargs="+", default=[0,1,2,3,4])
    ap.add_argument("--seed-indexes", type=int, nargs="+", default=[0,1,2,3,4])

    ap.add_argument("--positions", choices=["fixed","variable"], nargs="+", default=["variable"])
    ap.add_argument("--scales", choices=["zero_one","neg_one_one"], nargs="+", default=["zero_one"])
    ap.add_argument("--inverts", type=int, nargs="+", default=[0])

    ap.add_argument("--artifacts-dir", type=str, default="./artifacts")
    ap.add_argument("--models-dir", type=str, default="./models")
    ap.add_argument("--energies-dir", type=str, default="./energies")
    ap.add_argument("--explanations-dir", type=str, default="./explanations")

    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--limit", type=int, default=None)

    # new: derive fixed masks from the banner
    ap.add_argument("--derive-fixed-mask", action="store_true")
    ap.add_argument("--watermark", type=str, default="./watermark banner.jpg")
    ap.add_argument("--alpha-thresh", type=float, default=5/255)

    args = ap.parse_args()
    invert_bools = [bool(i) for i in args.inverts]

    from watermarks.train_watermarks_server import SEEDS as TRAIN_SEEDS

    work = []
    for (split, seed_idx, position, scale, inv) in itertools.product(
        args.splits, args.seed_indexes, args.positions, args.scales, invert_bools
    ):
        gpu = args.gpus[len(work) % len(args.gpus)]
        mask_mode = ("derive-fixed" if (position=="fixed" and args.derive_fixed_mask) else "auto")
        work.append((
            gpu, split, seed_idx,
            args.artifacts_dir, args.models_dir, args.energies_dir, args.explanations_dir,
            position, scale, inv, args.limit, args.skip_existing, TRAIN_SEEDS[seed_idx],
            mask_mode, args.watermark, args.alpha_thresh
        ))

    print(f"Planned jobs: {len(work)} | GPUs: {args.gpus}")
    os.makedirs(args.energies_dir, exist_ok=True); os.makedirs(args.explanations_dir, exist_ok=True)

    results = []
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=len(args.gpus)) as ex:
        futs = [ex.submit(run_job, w) for w in work]
        for f in as_completed(futs):
            r = f.result(); results.append(r)
            msg = f"[GPU {r['gpu']}] split={r['split']} seed={r['seed']} -> {r['status']}"
            if r["status"] == "fail": msg += f" | {r.get('err','')}"
            print(msg)

    ok = sum(r["status"]=="ok" for r in results)
    skipped = sum(r["status"]=="skipped" for r in results)
    fail = sum(r["status"]=="fail" for r in results)
    print(f"\nDONE: ok={ok} skipped={skipped} fail={fail}")

if __name__ == "__main__":
    main()
