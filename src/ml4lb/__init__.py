"""ML4LB: machine learning for the local branching heuristic.

Library package used by the training/evaluation scripts in src/:

- localbranching.py          the local branching (LB) heuristic algorithm;
- localbranching_ml.py       training and evaluation of the ML models
                             (regression for the initial k, RL for adapting
                             k and t) that guide the LB search (Section 5);
- execute_heuristics.py      SCIP integrated with the (ML-based) LB primal
                             heuristic and its result aggregation (Section 6);
- primal_heur_localbranch.py the LB algorithm as a SCIP primal heuristic;
- models.py / models_rl.py   the GNN regression model and the RL policies;
- dataset.py / event.py      dataset wrappers and SCIP event handlers;
- utilities.py               shared constants and helper functions;
- ecole_extend/              customized ecole environments.
"""
