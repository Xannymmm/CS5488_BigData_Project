"""
Part C main experiment: speedup of B's Random Forest training with n_jobs = 1, 2, 4, 8.
(Optional second stage: K-Means on pickup coordinates, thread-parallel.)

Follows B's hand-over document:
  * reads train_features.parquet directly (NO feature engineering is re-run)
  * calls B's  train_and_evaluate(df, n_jobs, n_estimators=20)  ->
        (model, features, train_time, y_val, preds)
  * keeps n_estimators=20 (B's setting; do not change for the official run)

Design for a fair, low-noise test
  * repeats are INTERLEAVED (rep 1: p=1,2,4,8; rep 2: p=8,4,2,1; ...) so thermal drift
    and background load spread over all settings instead of hitting one of them
  * time = train_time returned by B's function (model fitting); wall time also saved
  * RMSLE and a prediction checksum are recorded for every run: the parallel model must
    give the same predictions as n_jobs=1 (same seed), otherwise the speedup is meaningless

Run from the project root (folder containing modeling.py):
    pip install pandas numpy pyarrow scikit-learn threadpoolctl matplotlib
    python bench_rf.py --smoke            # ~1-2 min, 3 trees: checks that everything runs
    caffeinate -i python bench_rf.py      # full run, about 10-15 min (3 reps)

Output: parallel_bench/compute_timing.csv, figures/speedup_curve.png, figures/time_bars.png
"""
import argparse
import importlib.util
import os
import statistics as st
import sys
import time

import numpy as np
import pandas as pd

TARGET_RMSLE = 0.4188  # B's reported validation RMSLE


def load_modeling(path):
    if not os.path.exists(path):
        sys.exit(f"Cannot find {path}. Run from the folder that contains modeling.py, or pass --modeling.")
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec = importlib.util.spec_from_file_location("modeling_b", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "train_and_evaluate"):
        sys.exit("modeling.py has no train_and_evaluate(df, n_jobs, n_estimators).")
    return mod.train_and_evaluate


def rmsle(y_val, preds):
    """Works whether B returns log1p-space or second-space values."""
    y, p = np.asarray(y_val, float), np.asarray(preds, float)
    if y.max() > 50:                       # seconds -> log space
        y, p = np.log1p(y), np.log1p(np.clip(p, 0, None))
    return float(np.sqrt(np.mean((y - p) ** 2)))


def run_rf(train_and_evaluate, df, p, n_trees):
    t0 = time.perf_counter()
    _, _, train_time, y_val, preds = train_and_evaluate(df, n_jobs=p, n_estimators=n_trees)
    wall = time.perf_counter() - t0
    preds = np.asarray(preds, float)
    return float(train_time), wall, rmsle(y_val, preds), preds


def run_kmeans(X, p, k):
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits
    t0 = time.perf_counter()
    with threadpool_limits(limits=p):
        KMeans(n_clusters=k, n_init=1, max_iter=50, tol=0.0, random_state=42).fit(X)
    return time.perf_counter() - t0


def summarise(rows):
    out = pd.DataFrame(rows)
    out["speedup_mean"] = out.groupby("stage")["mean_s"].transform(lambda s: s.iloc[0] / s)
    out["speedup_best"] = out.groupby("stage")["min_s"].transform(lambda s: s.iloc[0] / s)
    out["efficiency_mean"] = out["speedup_mean"] / out["workers"]
    return out


def make_row(stage, p, ts, n_rows, extra=None):
    r = {"stage": stage, "workers": p, "n_rows": n_rows, "reps": len(ts),
         "mean_s": st.mean(ts), "min_s": min(ts),
         "std_s": st.stdev(ts) if len(ts) > 1 else 0.0,
         "all_s": ";".join(f"{t:.3f}" for t in ts)}
    r.update(extra or {})
    return r


def plot(out, fig_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping figures")
        return
    os.makedirs(fig_dir, exist_ok=True)
    workers = sorted(out["workers"].unique())
    colours = {"rf": "#B60007", "kmeans": "#156082"}
    labels = {"rf": "Random Forest (20 trees)", "kmeans": "K-Means (10 clusters)"}
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(workers, workers, "--", color="grey", label="Ideal (linear)")
    for stage in out["stage"].unique():
        d = out[out.stage == stage]
        ax.errorbar(d["workers"], d["speedup_mean"], marker="o", lw=2.5,
                    color=colours.get(stage), label=labels.get(stage, stage))
    ax.set_xlabel("Workers (n_jobs)")
    ax.set_ylabel("Speedup  S = T1 / Tp")
    ax.set_xticks(workers)
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "speedup_curve.png"), dpi=200)
    plt.close(fig)

    d = out[out.stage == "rf"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar([str(w) for w in d["workers"]], d["mean_s"], yerr=d["std_s"], color="#B60007", capsize=4)
    for i, v in enumerate(d["mean_s"]):
        ax.text(i, v, f"{v:.1f}s", ha="center", va="bottom")
    ax.set_xlabel("Workers (n_jobs)")
    ax.set_ylabel("Random Forest training time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "time_bars.png"), dpi=200)
    plt.close(fig)
    print(f"Figures saved in {fig_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="default: first existing of train_features.parquet, train_features_3.parquet, train_features_by_B.parquet")
    ap.add_argument("--modeling", default="modeling.py")
    ap.add_argument("--out", default="parallel_bench/compute_timing.csv")
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--trees", type=int, default=20, help="keep 20 for the official run (B's setting)")
    ap.add_argument("--no-kmeans", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="3 trees, 1 rep, small data: only checks the pipeline")
    args = ap.parse_args()
    args.workers = sorted(set(args.workers))
    if args.workers[0] != 1:
        sys.exit("--workers must include 1 (serial baseline).")
    if args.smoke:
        args.trees, args.reps = 3, 1

    train_and_evaluate = load_modeling(args.modeling)
    if args.data is None:
        cands = ["train_features.parquet", "train_features_3.parquet", "train_features_by_B.parquet"]
        args.data = next((c for c in cands if os.path.exists(c)), cands[0])
    if not os.path.exists(args.data):
        sys.exit(f"Cannot find {args.data}. Put B's feature file in the project root or pass --data <path>.")
    df = pd.read_parquet(args.data)
    print(f"Loaded {args.data}: shape {df.shape} (expected (1453583, 16))")
    if args.smoke:
        df = df.sample(100_000, random_state=42).reset_index(drop=True)
        print("SMOKE mode: 100,000 rows. Do not use these numbers.")
    print(f"CPU logical cores: {os.cpu_count()} | workers: {args.workers} | reps: {args.reps} | trees: {args.trees}")

    # ---------------- Random Forest, interleaved repeats ----------------
    times = {p: [] for p in args.workers}
    walls = {p: [] for p in args.workers}
    rm = {}
    ref_preds, max_diff = None, {p: 0.0 for p in args.workers}
    for rep in range(args.reps):
        order = args.workers if rep % 2 == 0 else args.workers[::-1]
        for p in order:
            tt, wall, r, preds = run_rf(train_and_evaluate, df, p, args.trees)
            times[p].append(tt)
            walls[p].append(wall)
            rm[p] = r
            if ref_preds is None and p == 1:
                ref_preds = preds
            elif ref_preds is not None and len(preds) == len(ref_preds):
                max_diff[p] = max(max_diff[p], float(np.max(np.abs(preds - ref_preds))))
            print(f"[RF] rep {rep+1}/{args.reps}  n_jobs={p}  train_time={tt:7.2f}s  RMSLE={r:.4f}", flush=True)

    rows = [make_row("rf", p, times[p], len(df),
                     {"wall_mean_s": st.mean(walls[p]), "rmsle": rm[p], "max_pred_diff_vs_p1": max_diff[p]})
            for p in args.workers]

    # ---------------- K-Means (secondary stage) ----------------
    if not args.no_kmeans:
        cols = ["pickup_longitude", "pickup_latitude"]
        if all(c in df.columns for c in cols):
            X = df[cols].to_numpy()
            kt = {p: [] for p in args.workers}
            for rep in range(args.reps):
                order = args.workers if rep % 2 == 0 else args.workers[::-1]
                for p in order:
                    kt[p].append(run_kmeans(X, p, 10))
                    print(f"[KM] rep {rep+1}/{args.reps}  threads={p}  time={kt[p][-1]:6.2f}s", flush=True)
            rows += [make_row("kmeans", p, kt[p], len(df)) for p in args.workers]
        else:
            print("pickup coordinate columns not found: skipping K-Means")

    out = summarise(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}\n")
    print(out[["stage", "workers", "mean_s", "min_s", "std_s", "speedup_mean", "speedup_best",
               "efficiency_mean"]].round(3).to_string(index=False))

    rf = out[out.stage == "rf"]
    print("\n--- correctness checks ---")
    print(f"Max |prediction difference| vs n_jobs=1: {rf['max_pred_diff_vs_p1'].max():.2e}  (should be ~0)")
    print(f"RMSLE per n_jobs: {dict(zip(rf.workers, rf.rmsle.round(4)))}  (B reported {TARGET_RMSLE})")
    if not args.smoke and abs(rf.rmsle.iloc[0] - TARGET_RMSLE) > 0.01:
        print("WARNING: RMSLE differs from B's 0.4188 - check data file / n_estimators / modeling.py version.")
    print("\n--- load balance (equal-cost trees) ---")
    for p in args.workers:
        per = -(-args.trees // p)
        print(f"  {p} workers: busiest builds {per} trees -> speedup bound {args.trees / per:.2f}x")
    plot(out, args.fig_dir)


if __name__ == "__main__":  # required for multiprocessing-safe use on macOS
    main()
