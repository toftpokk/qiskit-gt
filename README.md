# How We Scaled Heart Disease Screening with Quantum Approximate Optimization Algorithm (QAOA)

## Background

**Objective**

Given a total set of possible measurements, identify which five provide the most value to determine whether a patient has heart disease
Minimize overlapping information while maximizing relevancy

**Datasets**

UCI Heart Disease - Cleveland subset

Relevant features: “"sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"

## Comparison to Other Methods

**QAOA**

~6000 training cost evaluations at 3 layers to find optimum

**Greedy feed forward**

55 operations to find optimum

**Simulated annealing**

~100-400 to find optimum


No quantum advantage at n = 13

## Quantum Advantage
A larger feature set and more measurements permitted would exponentially increase the subset space

- 20 available features: 15504 subsets
- 30 available features: 142506 subsets

Classical methods would be infeasible here
