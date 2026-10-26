# kgrepurpose: how much does a random-split score say about rare-disease drug rescue?

Atlas Discovery mines "the corpus of biomedical knowledge" with agents that "highlight additional intersecting disease-to-drug pathways within a biomedical knowledge graph" to find treatments for rare diseases (https://atlasdiscovery.bio). Two things can make such a ranker look better than it is: scoring it on a random split of known indications, and drug popularity (well-studied drugs have more recorded indications). This prototype is a synthetic knowledge graph with planted mechanisms that measures both. It is not their system and uses no biomedical data.

## The setup
`kgrepurpose/world.py`: drugs hit genes, genes sit in pathways, diseases are driven by pathways; a drug treats a disease when one of its genes is in a driver pathway and the effect is real (half the time). Recorded indications and graph links are an incomplete copy: well-studied diseases keep most of theirs, rare diseases few; popular drugs are recorded more. Dev: 600 drugs, 300 diseases (33% rare). Held-out: 900 drugs, 400 diseases (50% rare), noisier graph (75% of links recorded, only 25% of rare diseases' pathway links), fewer known indications.
`kgrepurpose/rank.py`: three rankers: drug popularity (number of recorded indications); borrow from similar diseases (disease similarity by shared pathways); mechanism paths (drug-gene-pathway-disease path counts, normalised by drug degree). Score: AUC, the share of non-indication drugs a true drug outranks (50% = chance), averaged over test pairs. Two tests: a random split (20% of recorded indications of common diseases hidden) and rare diseases (real but unrecorded indications of rare diseases).

## Results (20 worlds each; held-out seeds 1000-1019, fixed in git before the first run)
| | dev: random split | dev: rare diseases | held-out: random split | held-out: rare diseases |
|---|---|---|---|---|
| drug popularity | 79.0% | 66.1% | 79.9% | 66.9% |
| borrow from similar diseases | 84.9% | 63.7% | 77.4% | 57.9% |
| mechanism paths | 82.8% | 65.2% | 70.1% | 56.9% |
- A random split flatters every ranker: 13-20 points above the rare-disease score.
- Drug popularity, a baseline with no biology, ties or beats both smarter rankers on the held-out rare diseases (66.9% vs 57-58%). A leaderboard on a random split would not show that.
- Added after the held-out run (`kgrepurpose/whatif.py`, 10 held-out worlds, rare diseases): mechanism paths score 56.7% on the observed graph, 76.4% if every rare disease's pathway links were recorded, and a blend of mechanism paths and popularity (equal z-scores) scores 71.3%. In this model the missing links of rare diseases, not the method, are what holds mechanism paths back.

## Read this before believing the table
The graph, the 50% real-effect rate, the recording rates and the popularity effect are all invented. Popularity helps here partly because popular drugs hit more genes and so truly treat more diseases in this world. Nothing here is a statement about Atlas Discovery's agents or data.

## Run it (3 commands)
```
pip install numpy pytest
python -m pytest -q tests
python -m kgrepurpose.run
```
numpy only. 4 tests; the run takes about a minute.

Unofficial prototype by Kushagra Pratap Singh. Not affiliated with Atlas Discovery.
