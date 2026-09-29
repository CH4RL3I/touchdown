"""Command line: ``landsim run | eval | video``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _cmd_run(a: argparse.Namespace) -> int:
    from .closed_loop import pose_errors, run_episode

    ep = run_episode(a.seed, a.level, max_time=a.max_time)
    e = pose_errors(ep)
    det = float((ep.marker > 0).mean())
    total = ep.timing_ms.sum(1)
    out = {
        "seed": a.seed,
        "level": a.level,
        "outcome": ep.outcome,
        "abort_reason": ep.abort_reason or None,
        "touchdown_error_m": None if np.isnan(ep.touchdown_error) else round(ep.touchdown_error, 4),
        "success": ep.success,
        "duration_s": round(ep.duration, 2),
        "detection_rate": round(det, 3),
        "median_pos_error_m": {
            "raw": round(float(np.nanmedian(e["raw_pos"])), 4),
            "filtered": round(float(np.nanmedian(e["filt_pos"])), 4),
        },
        "per_frame_ms": {
            "mean": round(float(total.mean()), 2),
            "p95": round(float(np.percentile(total, 95)), 2),
        },
    }
    print(json.dumps(out, indent=2))
    return 0


def _cmd_eval(a: argparse.Namespace) -> int:
    from .evaluate import evaluate

    evaluate(a.runs, Path(a.out), a.seed, a.level, a.sweep_runs, workers=a.workers)
    return 0


def _cmd_video(a: argparse.Namespace) -> int:
    from .video import write_video

    gif = Path(a.gif) if a.gif else None
    mp4 = Path(a.mp4) if a.mp4 else None
    ep = write_video(a.seed, a.level, gif, mp4, stride=a.stride)
    for p in (gif, mp4):
        if p is not None:
            print(f"wrote {p} ({p.stat().st_size / 1e6:.2f} MB)")
    print(f"episode: {ep.outcome}, touchdown error {ep.touchdown_error:.3f} m, {len(ep.t)} frames")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="landsim", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one closed-loop descent and print its summary")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--level", type=float, default=0.5, help="nuisance level, 0 clean .. 1 harsh")
    r.add_argument("--max-time", type=float, default=60.0)
    r.set_defaults(fn=_cmd_run)

    e = sub.add_parser("eval", help="Monte-Carlo evaluation: tables and figures")
    e.add_argument("--runs", type=int, default=100)
    e.add_argument("--seed", type=int, default=1000, help="first seed; runs use seed..seed+runs-1")
    e.add_argument("--level", type=float, default=0.5)
    e.add_argument("--sweep-runs", type=int, default=8, help="runs per nuisance level (0 disables)")
    e.add_argument("--workers", type=int, default=2, help="capped at 2")
    e.add_argument("--out", default="docs")
    e.set_defaults(fn=_cmd_eval)

    v = sub.add_parser("video", help="write an annotated GIF/MP4 of one descent")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--level", type=float, default=0.5)
    v.add_argument("--gif", default="docs/descent.gif")
    v.add_argument("--mp4", default=None)
    v.add_argument("--stride", type=int, default=3, help="keep every n-th frame")
    v.set_defaults(fn=_cmd_video)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
