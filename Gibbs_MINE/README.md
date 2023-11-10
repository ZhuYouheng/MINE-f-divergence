## varify_xxx
Varifying the correctness of the neural estamator using multi-variable gaussian distribution.
## Sample_Gibbs_from_gaussian.py
The realization of functions related to directly sampling from the Gibbs posterior which takes the form of a gaussian distribution in the case of Random Feature Model.
## kl/skl_estimator.ipynb
The original form of MINE.
## pw_estimator.ipynb(And its copies)
Using neural estimator $F$ to mimic $\log(P_w)$. The function that maximize the D-V representation is $\frac{P_{w,s}}{P_wP_s}=\frac{P_{w|s}}{P_w}$。When the problem is restricted on Random Feature Model, $P_{w|s}$ is gaussian and therefore is part of our prior knowledge, hence we only need to train a neural network that only deal with $w$.
## MINE.py
Configuration of the structures and inputs of the neural network.
## MALA generalization error (modified).ipynb
Estimating the population risk, empirical risk and the corresponding generalization error.
## Gibbs.py
Sampling Gibbs posterior using MALA.