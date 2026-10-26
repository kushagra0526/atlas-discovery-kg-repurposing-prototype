"""Three ways to rank drugs for a disease from the observed graph and the known indications that are visible at training time."""
import numpy as np

METHODS = ["drug popularity", "borrow from similar diseases", "mechanism paths (drug-gene-pathway-disease)"]


def scores(world, train_known):
    pop = np.repeat(train_known.sum(1, keepdims=True), world.known.shape[1], axis=1).astype(float)
    dp = world.dp.astype(float)
    inter = dp @ dp.T
    sim = inter / (dp.sum(1, keepdims=True) + dp.sum(1)[None, :] - inter + 1e-9)
    np.fill_diagonal(sim, 0.0)
    borrow = train_known.astype(float) @ sim
    paths = world.dg.astype(float) @ world.gp.astype(float) @ dp.T
    mech = paths / (1.0 + np.log1p(world.dg.sum(1, keepdims=True)))
    return {METHODS[0]: pop, METHODS[1]: borrow, METHODS[2]: mech}


def auc_at(score, truth, test_pairs):
    """Mean over test pairs of: share of non-indication drugs the true drug outranks (0.5 = chance)."""
    out = []
    for r, d in test_pairs:
        neg = score[~truth[:, d], d]
        out.append(float(((neg < score[r, d]).sum() + 0.5 * (neg == score[r, d]).sum()) / max(len(neg), 1)))
    return float(np.mean(out)) if out else float("nan")
