#!/usr/bin/env python
"""Standard label-diagnostic plots for Lux, in the shared style.

Predicted-vs-reference for every label coloured by the reduced chi^2 of the
spectral fit, residual-systematics grids (offsets vs labels / SNR / numax),
and a Kiel map coloured by residual, via the shared ``stardiag`` module
(sibling checkout, same pattern as the AnniesLasso import).

Usage:
    python src/plot_label_diagnostics.py                   # auto-detect
    python src/plot_label_diagnostics.py --results <parquet> --split test
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # Lux/src
REPO = HERE.parent

for _c in (REPO.parent / "stardiag", Path.home() / "code" / "stardiag",
           Path.home() / "scr_mk27" / "stardiag"):
    if (_c / "stardiag.py").exists():
        sys.path.insert(0, str(_c))
        break
else:
    sys.exit("stardiag module not found (expected a 'stardiag' checkout "
             "next to this repo)")
import stardiag  # noqa: E402

RESULTS_CANDIDATES = [
    REPO / "notebooks" / "lux-results-rgb-wilett-all-missions.parquet",
    Path("/scratch/mk27/mj8805/Lux/notebooks/"
         "lux-results-rgb-wilett-all-missions.parquet"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", help="lux results parquet "
                    "(default: auto-detect)")
    ap.add_argument("--split", default="test", choices=["test", "train", "all"],
                    help="which split to plot (default: test)")
    ap.add_argument("--plot-dir", default=str(REPO / "plots" / "diagnostics"),
                    help="where to write PNGs")
    args = ap.parse_args()

    results = (Path(args.results) if args.results else
               next((p for p in RESULTS_CANDIDATES if p.exists()), None))
    if results is None or not results.exists():
        sys.exit(f"no results parquet found; looked for "
                 f"{[str(p) for p in RESULTS_CANDIDATES]} — run the last cell "
                 f"of notebooks/train-rgb-wilett-all-missions.ipynb first")
    print(f"predictions: {results}")

    spec = stardiag.load_lux(results)
    if args.split != "all" and "split" in spec["df"].columns:
        spec["df"] = spec["df"][spec["df"]["split"] == args.split]
        print(f"{len(spec['df'])} stars in split='{args.split}'")

    for p in stardiag.make_all(spec, args.plot_dir, prefix=f"{args.split}_"):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
