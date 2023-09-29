# MINE-f-divergence
MINE(Mutual Information Neural Estimation), implemented on f-divergence(SKL in specific).
## Brief explaination
The estimator use numerical method to obtain the value and gradient of function $ f* $ since in the case of SKL divergence and other types of f-divergence, f* might not have a explicit form.
## Varification
The approach is tested out on a five dimensional gaussian distribution $ X~N(0,\Sigma_0) $. With an estimator using the numerical method, we estimated the following KL-divergence and SKL-divergence:<br>
$$ 
D_{KL}(P_{X}||P_{X_{1~3}\times X_{4~5}})
$$
$$ 
D_{SKL}(P_{X}||P_{X_{1~3}\times X_{4~5}})
$$
where the former indicates the mutual information between the the 1~3 and 4~5 dimension of a gaussian distribution.
