import argparse
import csv
import json

from . import fid as fid_mod, mesh as mesh_mod, stats


def read_column(path, col):
    with open(path, newline="") as f:
        return {r["id"]: float(r[col]) for r in csv.DictReader(f)}


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m monovision.eval")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fid", help="FID between two image folders")
    f.add_argument("a")
    f.add_argument("b")
    f.add_argument("--limit", type=int, default=None, help="use the first N images of each folder")
    m = sub.add_parser("mesh", help="mesh metrics, pred vs gt (.ply/.obj)")
    m.add_argument("pred")
    m.add_argument("gt")
    m.add_argument("--thresh", type=float, default=0.05)
    c = sub.add_parser("compare", help="paired test on one column of two CSVs with an 'id' column")
    c.add_argument("a_csv")
    c.add_argument("b_csv")
    c.add_argument("--col", default="fscore")
    a = p.parse_args(argv)
    if a.cmd == "fid":
        out = {"fid": fid_mod.fid(a.a, a.b, a.limit)}
    elif a.cmd == "mesh":
        out = mesh_mod.mesh_metrics(a.pred, a.gt, a.thresh)
    else:
        x, y = read_column(a.a_csv, a.col), read_column(a.b_csv, a.col)
        ids = sorted(set(x) & set(y))
        out = stats.paired_test([x[i] for i in ids], [y[i] for i in ids])
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
