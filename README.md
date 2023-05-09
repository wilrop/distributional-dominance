# Distributional Dominance
This repo contains all the code necessary to reproduce the results for the paper "Distributional Multi-Objective Decision Making".

# Structure
The repo is structured as follows:
- `algs` contains two versions of DIMOQ, the algorithm proposed in the paper. The first version implements the algorithm exactly as written in the pseudocode. The second makes a simplification to estimate the returns for the next states more efficiently. In addition, it contains an implementation of multi-objective distributional value iteration.
- `envs` contains the environments used in the paper as well as an implementation of the Space Traders environment introduced by Vamplew et al. These all approximately follow the Gym API.
- `distrib` contains all code necessary to deal with the return distributions. This includes a multivariate categorical distribution, the dominance relations and pruning operators. 
- `utils` contains miscellaneous code that is used throughout the repo.

The files in the top level folder run the experiments, pruning, analysis, plotting and tests.

## Citation

If you use this code or the results in your research, please use the following BibTeX entry:

```
@misc{ropke2023distributional,  
  url = {TBD},
  
  author = {Röpke, Willem and Hayes, Conor F. and Mannion, Patrick and Howley, Enda and Nowé, Ann and Roijers, Diederik M.},
  
  title = {Distributional Multi-Objective Decision Making},
  
  year = {2023},
  
  note={Accepted at IJCAI 2023}
}
```