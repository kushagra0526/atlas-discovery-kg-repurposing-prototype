import json

import numpy as np

from . import rank, world

CONFIG = {"dev": dict(), "held": dict(genes=600, diseases=400, drugs=900, rare_share=0.5, obs=0.75, obs_rare_paths=0.25, known_common=0.6, known_rare=0.05)}


def score(seeds, cfg):
    acc = {m: {"random_split": [], "rare_diseases": [], "chance": []} for m in rank.METHODS}
    for s in seeds:
        w = world.World(s, **cfg)
        rng = np.random.default_rng(s + 77)
        common = ~w.rare
        pairs = np.argwhere(w.known & common[None, :])
        hold = pairs[rng.random(len(pairs)) < 0.2]                        # random split: hide 20% of the recorded indications of common diseases
        train = w.known.copy()
        train[hold[:, 0], hold[:, 1]] = False
        rare_pairs = np.argwhere(w.truth & w.rare[None, :] & ~w.known)   # rare diseases: every real mechanism indication that is not recorded
        rare_pairs = rare_pairs[rng.random(len(rare_pairs)) < 0.3]
        sc = rank.scores(w, train)
        for m in rank.METHODS:
            acc[m]["random_split"].append(rank.auc_at(sc[m], w.truth, [tuple(p) for p in hold]))
            acc[m]["rare_diseases"].append(rank.auc_at(sc[m], w.truth, [tuple(p) for p in rare_pairs]))
        acc[rank.METHODS[0]]["chance"].append(10 / w.known.shape[0])
    out = {m: {"auc_random_split_pct": round(100 * float(np.nanmean(a["random_split"])), 1), "auc_rare_diseases_pct": round(100 * float(np.nanmean(a["rare_diseases"])), 1)} for m, a in acc.items()}
    return out


def main():
    res = {"dev": score(range(0, 20), CONFIG["dev"]), "held_out": score(range(1000, 1020), CONFIG["held"])}
    json.dump(res, open("out/result.json", "w"), indent=1)
    for k, v in res.items():
        print(k)
        for m, d in v.items():
            print("  ", m, d)


if __name__ == "__main__":
    main()
