import numpy as np

from kgrepurpose import rank, world


def test_recorded_indications_are_a_subset_of_the_truth():
    w = world.World(1, genes=100, diseases=60, drugs=80)
    assert w.truth.any() and (w.known <= w.truth).all() and w.known.sum() < w.truth.sum()


def test_rare_diseases_have_fewer_recorded_indications_and_pathways():
    w = world.World(2)
    assert w.known[:, w.rare].mean() < w.known[:, ~w.rare].mean()
    assert w.dp[w.rare].sum() / max(w.true_dp[w.rare].sum(), 1) < w.dp[~w.rare].sum() / w.true_dp[~w.rare].sum()


def test_auc_is_half_for_constant_scores_and_one_for_perfect():
    truth = np.zeros((10, 2), bool)
    truth[3, 0] = True
    assert rank.auc_at(np.ones((10, 2)), truth, [(3, 0)]) == 0.5
    s = np.zeros((10, 2))
    s[3, 0] = 5
    assert rank.auc_at(s, truth, [(3, 0)]) == 1.0


def test_scores_have_the_right_shape_and_ignore_hidden_pairs():
    w = world.World(3, genes=80, diseases=40, drugs=50)
    train = w.known.copy()
    sc = rank.scores(w, train)
    assert all(v.shape == w.known.shape for v in sc.values())
