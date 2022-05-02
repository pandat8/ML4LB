<<<<<<< HEAD
# Code and data for learning to search in local branching


## Installation

1. Please install Python 3.7, PyscipOPt 3.1.1, and SCIP 7.01 on your computer

2. install package ecole according to the following instructions:
    - go into the folder 'ecole'
    - install the package according to 'installation.rst'

## Running the experiments

#### Only Computing and Plot the evaluation results in the paper (e.g. primal integral, primal gap)
```
compute_evaluation_results.py
```

#### Evaluating the trained model on 5 datasets by your own machine
```
# evaluate Algorithm lb-baseline, lb-sr, lb-srm, 
evaluation_regression_k_prime.py

# evaluate Algorithm lb-rl, lb-srmrl
evaluation_reinforce4lb.py

# compute and plot the evaluation results
compute_evaluation_results.py
```

#### Train your own regression model, RL model, and then evaluating them by your own machine
```
# train regression models
train_regression.py

# train RL models
train_reinforce4lb.py

# evaluate Algorithm lb-sr, lb-srm by your own regression model, evaluate lb-baseline 
evaluation_regression_k_prime.py --regression_model_path='path to your own model trained by mixed dataset' # after training, you can select the models from '.results/saved_models/regression/' folder 

# evaluate Algorithm lb-rl, lb-srmrl
evaluation_reinforce4lb.py --rl_model_path='path to your own model' # after training, you can select your models from '.results/saved_models/rl/reinforce/setcovering/' folder

# compute and plot the evaluation results
compute_evaluation_results.py
```
=======
boost the priximity search of local branching algorithm with ml techniques.
>>>>>>> Update README.md
