# ICAPS2024Exp

This repository contains the code to run the experiments described in the paper [JaxPlan and GurobiPlan: Optimization Baselines for Replanning in Discrete and Mixed Discrete-Continuous Probabilistic Domains](https://ojs.aaai.org/index.php/ICAPS/article/view/31480) published at ICAPS 2024.

## Usage

To run the JAX planner, Gurobi planner and other simple policies (random, no-op), use the command

```shell-session
main.py <domain> <instance> <method> <online> <tuning> <time>'
```

where:
- ``<domain>`` is the RDDL domain
- ``<method>`` is either ``jaxplan``, ``gurobiplan``, ``random`` or ``noop``
- ``<online>`` whether to run in replanning (True) or straight line planning (False) mode
- ``<tuning>`` whether to do hyperparameter tuning
- ``<time>`` is the maximum time allowed to think (in seconds) per decision

A batch script is provided to run all the baselines in one go (with tuning):

```shell-session
source ./run.sh <domain> <online> <time>
```

To run the prost docker:

```shell-session
$ docker build -t prost .
$ source ./run.sh <domain> <instance> <time>
```
