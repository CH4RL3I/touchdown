"""Monte-Carlo evaluation against ground truth, tables and figures."""

from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .closed_loop import TOUCHDOWN_SUCCESS_RADIUS, Episode, pose_errors, run_episode

ALT_EDGES = np.array([0.15, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 11.5])
COMMIT_ALT = 0.6  # detection rates are quoted above this altitude, where the pad is in view


def _worker(args: tuple[int, float]) -> Episode:
    import cv2

    cv2.setNumThreads(1)
    seed, level = args
    return run_episode(seed, level).lean()


def run_many(seeds: list[int], level: float, workers: int = 2) -> list[Episode]:
    jobs = [(s, level) for s in seeds]
    if workers <= 1:
        return [_worker(j) for j in jobs]
    with Pool(min(workers, 2)) as pool:  # the machine is shared: never more than 2 processes
        return pool.map(_worker, jobs, chunksize=1)


def _pct(x: np.ndarray, q: float) -> float:
    x = x[~np.isnan(x)]
    return float(np.percentile(x, q)) if len(x) else float("nan")


def error_vs_altitude(eps: list[Episode]) -> dict:
    """Median and 90th-percentile pose error per altitude bin, raw PnP vs filter."""
    cols = {k: [] for k in ["alt", "raw_pos", "raw_att", "filt_pos", "filt_att", "marker"]}
    for ep in eps:
        e = pose_errors(ep)
        cols["alt"].append(ep.p_true[:, 2])
        for k in ["raw_pos", "raw_att", "filt_pos", "filt_att"]:
            cols[k].append(e[k])
        cols["marker"].append(ep.marker)
    c = {k: np.concatenate(v) for k, v in cols.items()}
    rows = []
    for lo, hi in zip(ALT_EDGES[:-1], ALT_EDGES[1:], strict=True):
        m = (c["alt"] >= lo) & (c["alt"] < hi)
        row = {"alt_lo": float(lo), "alt_hi": float(hi), "frames": int(m.sum())}
        for k in ["raw_pos", "raw_att", "filt_pos", "filt_att"]:
            row[f"{k}_med"] = _pct(c[k][m], 50)
            row[f"{k}_p90"] = _pct(c[k][m], 90)
        row["detect_rate"] = float((c["marker"][m] > 0).mean()) if m.any() else float("nan")
        row["large_rate"] = float((c["marker"][m] == 1).mean()) if m.any() else float("nan")
        row["small_rate"] = float((c["marker"][m] == 2).mean()) if m.any() else float("nan")
        rows.append(row)
    return {"bins": rows}


def summarise(eps: list[Episode]) -> dict:
    n = len(eps)
    landed = [e for e in eps if e.outcome == "landed"]
    td = np.array([e.touchdown_error for e in landed])
    aborts: dict[str, int] = {}
    for e in eps:
        if e.outcome != "landed":
            key = e.abort_reason or e.outcome
            aborts[key] = aborts.get(key, 0) + 1
    in_view = np.concatenate([e.p_true[:, 2] >= COMMIT_ALT for e in eps])
    detected = np.concatenate([e.marker > 0 for e in eps])
    tm = np.concatenate([e.timing_ms for e in eps])
    total = tm.sum(1)
    out = {
        "runs": n,
        "landed": len(landed),
        "success": int(sum(e.success for e in eps)),
        "success_rate": sum(e.success for e in eps) / n,
        "success_radius_m": TOUCHDOWN_SUCCESS_RADIUS,
        "aborted_or_timeout": aborts,
        "touchdown_error_m": {
            "mean": float(td.mean()) if len(td) else None,
            "median": float(np.median(td)) if len(td) else None,
            "p90": _pct(td, 90),
            "p95": _pct(td, 95),
            "max": float(td.max()) if len(td) else None,
            "rms": float(np.sqrt((td**2).mean())) if len(td) else None,
            "within_pad_half_size": float((td < 0.9).mean()) if len(td) else None,
        },
        "detection_rate_above_commit_altitude": float(detected[in_view].mean()),
        "mean_descent_time_s": float(np.mean([e.duration for e in landed])) if landed else None,
        "runtime_ms": {
            "perception_mean": float(tm[:, 0].mean()),
            "filter_mean": float(tm[:, 1].mean()),
            "guidance_mean": float(tm[:, 2].mean()),
            "total_mean": float(total.mean()),
            "total_p95": float(np.percentile(total, 95)),
            "total_p99": float(np.percentile(total, 99)),
            "total_max": float(total.max()),
            "fps_at_mean": float(1000.0 / total.mean()),
            "fps_at_p95": float(1000.0 / np.percentile(total, 95)),
            "frames": int(len(total)),
        },
    }
    return out


def sweep(levels: list[float], runs_per_level: int, seed0: int, workers: int) -> list[dict]:
    rows = []
    for L in levels:
        eps = run_many([seed0 + i for i in range(runs_per_level)], L, workers)
        s = summarise(eps)
        rows.append(
            {
                "level": L,
                "runs": runs_per_level,
                "detection_rate": s["detection_rate_above_commit_altitude"],
                "success_rate": s["success_rate"],
                "landed": s["landed"],
                "touchdown_median": s["touchdown_error_m"]["median"],
                "aborts": s["aborted_or_timeout"],
            }
        )
        print(
            f"  level {L:.2f}: detect {rows[-1]['detection_rate']:.3f}"
            f" success {rows[-1]['success_rate']:.2f}",
            flush=True,
        )
    return rows


def _plain_axes(a) -> None:
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    for axis in (a.xaxis, a.yaxis):
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())


def make_figures(eps: list[Episode], alt: dict, sweep_rows: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 130,
            "savefig.bbox": "tight",
        }
    )
    RAW, FILT, GREY = "#d95f02", "#1b6ca8", "#666666"
    bins = alt["bins"]
    mid = np.array([np.sqrt(b["alt_lo"] * b["alt_hi"]) for b in bins])

    # 1. pose error vs altitude
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for a, key, unit, title in [
        (ax[0], "pos", "m", "Position error"),
        (ax[1], "att", "deg", "Attitude error"),
    ]:
        for name, col, lab in [("raw", RAW, "raw PnP"), ("filt", FILT, "filtered")]:
            med = np.array([b[f"{name}_{key}_med"] for b in bins])
            p90 = np.array([b[f"{name}_{key}_p90"] for b in bins])
            a.plot(mid, med, "-o", color=col, label=f"{lab} (median)", ms=4)
            a.plot(mid, p90, "--", color=col, alpha=0.7, label=f"{lab} (90th pct)")
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("true altitude (m)")
        a.set_ylabel(f"{title.lower()} ({unit})")
        a.set_title(title)
        a.set_xticks([0.3, 0.7, 1.5, 3, 6, 10])
        a.set_yticks([0.003, 0.01, 0.03, 0.1, 0.3, 1.0] if key == "pos" else [0.2, 0.5, 1, 2, 5])
        _plain_axes(a)
        a.invert_xaxis()
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Pose error vs altitude ({len(eps)} closed-loop runs)", y=1.02)
    fig.savefig(out / "pose_error_vs_altitude.png")
    plt.close(fig)

    # 2. detection vs altitude and vs nuisance level
    n_panels = 2 if sweep_rows else 1
    fig, ax = plt.subplots(1, n_panels, figsize=(5 * n_panels, 3.8), squeeze=False)
    a = ax[0][0]
    a.plot(mid, [b["detect_rate"] for b in bins], "-o", color=FILT, label="any marker", ms=4)
    a.plot(
        mid, [b["large_rate"] for b in bins], "-s", color="#7570b3", label="large marker used", ms=4
    )
    a.plot(mid, [b["small_rate"] for b in bins], "-^", color=RAW, label="small marker used", ms=4)
    a.set_xscale("log")
    a.set_xlabel("true altitude (m)")
    a.set_ylabel("fraction of frames")
    a.set_ylim(-0.02, 1.02)
    a.set_title("Detection rate vs altitude (nuisance 0.5)")
    a.set_xticks([0.3, 0.7, 1.5, 3, 6, 10])
    _plain_axes(a)
    a.invert_xaxis()
    a.legend(frameon=False, fontsize=8)
    if sweep_rows:
        a = ax[0][1]
        lv = [r["level"] for r in sweep_rows]
        a.plot(
            lv,
            [r["detection_rate"] for r in sweep_rows],
            "-o",
            color=FILT,
            label="detection rate (above 0.6 m)",
        )
        a.plot(
            lv,
            [r["success_rate"] for r in sweep_rows],
            "-s",
            color=RAW,
            label="landing success rate",
        )
        a.set_ylim(-0.02, 1.02)
        a.set_xlabel("nuisance level (0 clean, 0.5 nominal, 1 harsh)")
        a.set_ylabel("rate")
        a.set_title(f"Robustness to image nuisances ({sweep_rows[0]['runs']} runs per level)")
        a.legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(out / "detection_and_robustness.png")
    plt.close(fig)

    # 3. touchdown distribution
    landed = [e for e in eps if e.outcome == "landed"]
    xy = np.array([e.touchdown_xy for e in landed])
    r = np.linalg.norm(xy, axis=1)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
    a = ax[0]
    a.add_patch(
        plt.Rectangle((-0.9, -0.9), 1.8, 1.8, fill=False, ec=GREY, lw=1, label="pad (1.8 m)")
    )
    a.add_patch(
        plt.Rectangle(
            (-0.6, -0.6), 1.2, 1.2, fill=False, ec=GREY, lw=1, ls=":", label="large marker"
        )
    )
    a.add_patch(
        plt.Circle(
            (0, 0), TOUCHDOWN_SUCCESS_RADIUS, fill=False, ec=RAW, lw=1.2, label="success radius"
        )
    )
    a.scatter(xy[:, 0], xy[:, 1], s=14, color=FILT, alpha=0.8, label="touchdown")
    a.set_aspect("equal")
    lim = max(1.0, float(np.abs(xy).max()) * 1.1)
    a.set_xlim(-lim, lim)
    a.set_ylim(-lim, lim)
    a.set_xlabel("east (m)")
    a.set_ylabel("north (m)")
    a.set_title("Touchdown positions")
    a.legend(frameon=False, fontsize=7, loc="lower right")
    a = ax[1]
    a.hist(r, bins=np.linspace(0, max(0.3, r.max() * 1.05), 20), color=FILT, alpha=0.85)
    a.axvline(np.median(r), color=RAW, ls="--", label=f"median {np.median(r) * 100:.1f} cm")
    a.axvline(
        np.percentile(r, 90),
        color=GREY,
        ls=":",
        label=f"90th pct {np.percentile(r, 90) * 100:.1f} cm",
    )
    a.set_xlabel("radial touchdown error (m)")
    a.set_ylabel("runs")
    a.set_title(f"Touchdown error ({len(landed)} landings of {len(eps)} runs)")
    a.legend(frameon=False, fontsize=8)
    fig.savefig(out / "touchdown_error.png")
    plt.close(fig)

    # 4. runtime
    tm = np.concatenate([e.timing_ms for e in eps])
    total = tm.sum(1)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    a = ax[0]
    a.hist(total, bins=np.linspace(0, min(total.max(), 60), 60), color=FILT, alpha=0.85)
    a.axvline(1000 / 30, color=RAW, ls="--", label="30 fps budget (33 ms)")
    a.axvline(
        np.percentile(total, 95), color=GREY, ls=":", label=f"p95 {np.percentile(total, 95):.1f} ms"
    )
    a.set_xlabel("per-frame processing time (ms)")
    a.set_ylabel("frames")
    a.set_title("Perception + filter + guidance")
    a.legend(frameon=False, fontsize=8)
    a = ax[1]
    names, vals = ["detect + PnP", "filter", "guidance"], tm.mean(0)
    a.barh(names, vals, color=[FILT, "#7570b3", GREY])
    for i, v in enumerate(vals):
        a.text(v, i, f" {v:.2f} ms", va="center", fontsize=8)
    a.set_xlabel("mean time per frame (ms)")
    a.set_title("Mean cost by stage")
    a.grid(axis="y", visible=False)
    fig.savefig(out / "runtime.png")
    plt.close(fig)


def evaluate(
    runs: int,
    out: Path,
    seed0: int = 1000,
    level: float = 0.5,
    sweep_runs: int = 8,
    sweep_levels: list[float] | None = None,
    workers: int = 2,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    print(f"Monte Carlo: {runs} runs at nuisance level {level} ({workers} workers)", flush=True)
    eps = run_many([seed0 + i for i in range(runs)], level, workers)
    summary = summarise(eps)
    alt = error_vs_altitude(eps)
    rows: list[dict] = []
    if sweep_runs > 0:
        levels = sweep_levels or [0.0, 0.25, 0.5, 0.75, 1.0]
        print(f"Nuisance sweep: {levels} x {sweep_runs} runs", flush=True)
        rows = sweep(levels, sweep_runs, seed0 + 5000, workers)
    result = {
        "nominal": summary,
        "error_vs_altitude": alt,
        "nuisance_sweep": rows,
        "seeds": [seed0, seed0 + runs - 1],
        "level": level,
    }
    (out / "results.json").write_text(json.dumps(result, indent=2))
    make_figures(eps, alt, rows, out)
    print(json.dumps(summary, indent=2))
    return result
