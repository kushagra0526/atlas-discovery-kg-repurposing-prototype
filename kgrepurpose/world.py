"""A synthetic biomedical knowledge graph with planted drug-disease mechanisms, and an incomplete observed copy of it.

Drugs hit genes, genes sit in pathways, diseases are driven by pathways. A drug treats a disease when one of its genes is in a driver pathway (and the effect is real half the time).
Well-studied (common) diseases have most of their links and known treatments recorded; rare diseases have few. All entities are generated."""
import numpy as np


class World:
    def __init__(self, seed, genes=400, paths=40, diseases=300, drugs=600, rare_share=0.33, obs=0.9, obs_rare_paths=0.4, known_common=0.7, known_rare=0.1):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.rare = rng.random(diseases) < rare_share
        gp = np.zeros((genes, paths), bool)
        for g in range(genes):
            gp[g, rng.choice(paths, size=int(rng.integers(1, 3)), replace=False)] = True
        dp = np.zeros((diseases, paths), bool)
        for d in range(diseases):
            dp[d, rng.choice(paths, size=int(rng.integers(1, 3)), replace=False)] = True
        pop = rng.lognormal(0, 0.9, drugs)                                  # well-studied drugs have more targets and more annotations
        dg = np.zeros((drugs, genes), bool)
        for r in range(drugs):
            k = int(np.clip(rng.poisson(1 + 2 * pop[r]), 1, 12))
            dg[r, rng.choice(genes, size=k, replace=False)] = True
        reach = (dg.astype(int) @ gp.astype(int) @ dp.astype(int).T) > 0    # drug hits a driver pathway of the disease
        self.truth = reach & (rng.random(reach.shape) < 0.5)                 # real effect
        known_p = np.where(self.rare, known_rare, known_common)[None, :] * np.clip(pop / pop.mean(), 0.2, 3.0)[:, None]
        self.known = self.truth & (rng.random(self.truth.shape) < np.clip(known_p, 0, 1))    # recorded indications (popular drugs are recorded more)
        self.dg = dg & (rng.random(dg.shape) < obs)
        self.gp = gp & (rng.random(gp.shape) < obs)
        self.dp = dp & (rng.random(dp.shape) < np.where(self.rare, obs_rare_paths, obs)[:, None])
        self.true_dp = dp
