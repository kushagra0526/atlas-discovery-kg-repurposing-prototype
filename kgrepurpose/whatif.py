"""Added after the held-out run: (a) what if the rare diseases' pathway links were all recorded, (b) a blend of mechanism paths and drug popularity."""
import numpy as np

from . import rank, run, world


def z(a):
    return (a - a.mean(0, keepdims=True)) / (a.std(0, keepdims=True) + 1e-9)


def main(seeds=range(1000, 1010)):
    res = {"observed links": [], "all pathway links recorded": [], "blend (observed links): mechanism + popularity": []}
    for s in seeds:
        w = world.World(s, **run.CONFIG["held"])
        rng = np.random.default_rng(s + 77)
        pairs = np.argwhere(w.truth & w.rare[None, :] & ~w.known)
        pairs = [tuple(p) for p in pairs[rng.random(len(pairs)) < 0.3]]
        sc = rank.scores(w, w.known)
        res["observed links"].append(rank.auc_at(sc[rank.METHODS[2]], w.truth, pairs))
        res["blend (observed links): mechanism + popularity"].append(rank.auc_at(z(sc[rank.METHODS[2]]) + z(sc[rank.METHODS[0]]), w.truth, pairs))
        w.dp = w.true_dp.copy()
        res["all pathway links recorded"].append(rank.auc_at(rank.scores(w, w.known)[rank.METHODS[2]], w.truth, pairs))
    for k, v in res.items():
        print(f"mechanism paths on rare diseases, {k}: AUC {100 * np.nanmean(v):.1f}%")


if __name__ == "__main__":
    main()
