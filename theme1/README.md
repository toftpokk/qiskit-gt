Prompt B: Heart Disease Readings

A clinic wants a cheap screening test for heart disease and can only collect 5 measurements per patient. Some measurements are very informative but redundant with each other (cholesterol and age may tell you overlapping things), so the best 5 are not simply the 5 individually strongest.
Picking the best set is a combinatorial problem the number of possible subsets explodes as features grow. That is exactly the structure quantum optimization algorithms target.
Each feature is a yes/no decision: keep or drop One qubit per feature: |1> = keep, |0> = drop. A 13-feature problem is a 13-qubit problem.

QUBO: A formula over 0/1 variables that you minimize. Quantum optimizers accept this format 
QAOA: A quantum algorithm that tries to find low-value QUBO answers, with p layers controlling circuit depth 
Approximation Ratio: How close a method's answer is to the true optimum (1.0 = perfect) |

UCI Heart Disease: 303 patients, 13 features, licensed CC BY 4.0. Load with fetch_ucirepo(id=45) from the ucimlrepo package. The Cleveland subset has 297 patients with no missing values, so drop rows with missing entries The target ranges 0 to 4, so binarize 0 = no disease, 1-4 = disease.
Page: https://archive.ics.uci.edu/ml/datasets/heart%20disease

You will minimize the expression:

- ...
                   
- r_i (relevance): mutual information between feature i and the label.
- c_{ij} (redundancy): absolute correlation between features i and j.
- lambda: how hard you punish redundancy.
- P: a large penalty that forces choosing exactly kfeatures.
-                   : keep (1) or drop (0).
