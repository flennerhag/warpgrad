# Maze Navigation Experiment

Code to reproduce the Maze Navigation experiment in 
Meta-Learning with Warped Gradient Descent
(https://openreview.net/forum?id=rkeiQlBFPB).

Originally proposed by 
[Miconi et al.. ICML 2018](https://arxiv.org/abs/1804.02464), 
[Miconi et al.. ICLR 2019](https://openreview.net/pdf?id=r1lrAiA5Ym). We have
only modified the `Network` class in and the optimizer update step in
`run.py` from the original script, 
other changes are refactoring to improve readibility.

`run.py` only runs Warpgrad, for baselines and plotting 
see [original implementation](https://github.com/uber-research/backpropamine/tree/master/maze).

To replicate the paper, run (you may want to parallelise it)

```bash
for SEED in 0 1 2 3 4 5 6 7 8 9
do
  python run.py --rngseed $SEED
done
```
