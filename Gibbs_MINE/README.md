# Coding Files
### varify_xxx
Varifying the correctness of the neural estamator using multi-variable gaussian distribution.
### Sample_Gibbs_from_gaussian.py
The realization of functions related to directly sampling from the Gibbs posterior which takes the form of a gaussian distribution in the case of Random Feature Model.
### kl/skl_estimator.ipynb
The original form of MINE.
### pw_estimator.ipynb(And its copies)
Using neural estimator $F$ to mimic $\log(P_w)$. The function that maximize the D-V representation is $\frac{P_{w,s}}{P_wP_s}=\frac{P_{w|s}}{P_w}$。When the problem is restricted on Random Feature Model, $P_{w|s}$ is gaussian and therefore is part of our prior knowledge, hence we only need to train a neural network that only deal with $w$. <bar>
Note that **'pw_estimator.ipynb'** is used for estimating **mutual information** while **'pw_estimator.ipynb copy 1-3'** are used for estimating **lautum information**. The differences are small, and lies in the specific form of the loss function.
### MINE.py
Configuration of the structures and inputs of the neural network.
### MALA generalization error (modified).ipynb
Estimating the population risk, empirical risk and the corresponding generalization error.
### Gibbs.py
Sampling Gibbs posterior using MALA.
### Calculate_MI_from_model.ipynb(And its copies)
Using pretrained models to estimate mutual information, this includes a process of restarting the training procedure with larger samples size but relatively less steps. The data collected from the second training procedure is used for estimation.
### Estimate_MI.ipynb
Similar to 'Calculate_MI_from_model.ipynb', but instead of taking in a model and retrain it, it uses existing data generated from a training process and analyse the mutual information.

# Folders
### kl_pw_result
The result of lautum information throughout the training epoches, subject to using $P_w$ estimator as mentioned above.
### pw_res
The culminative models (each representing a $P_w$ ) after training. We can use **'Calculate_MI_from_model.ipynb(And its copies)'** to obtain the corresponding mutual information or lautum information(, in which case these models are seen as pre-trained models).
### pw_result_trained_twice
The culminative models (each representing a $P_w$ ) after training twice. This means that the models are generated from **'Calculate_MI_from_model.ipynb(And its copies)'**. Again, we can also use **'Calculate_MI_from_model.ipynb(And its copies)'** to obtain the corresponding mutual information or lautum information(, in which case these models are trained for a **third** time).