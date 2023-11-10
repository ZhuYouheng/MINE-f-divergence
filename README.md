# MINE-f-divergence
MINE(Mutual Information Neural Estimation), implemented on f-divergence(**Symmetrized KL information** in specific).
## Brief explaination
The estimator use numerical method to obtain the value and gradient of function $` f^* `$ since in the case of SKL divergence and other types of f-divergence, $`f^*`$ might not have an explicit form.
## Varification
The approach is tested out on a five dimensional gaussian distribution $` X\sim \mathcal{N}(0,\Sigma_0)\in \mathbb{R}^5 `$. With an estimator using the numerical method, we estimated the following KL-divergence and SKL-divergence:<br>
$$D_{KL}(P_{X}||P_{X_{1,2,3}\ \times X_{4,5}})$$
$$D_{SKL}(P_{X}||P_{X_{1,2,3}\ \times X_{4,5}})$$
which respectively indicates the mutual information and the symmetrized KL-information between the the 1,2,3 and 4,5 dimension of a gaussian distribution.<br>
Then the result of estimated mutual information is compared with the value theoratically calculated from the formula below:<br>
$$D_{KL}(\mathcal{N}(m_1,\Sigma_1)||\mathcal{N}(m_0,\Sigma_0))=\frac{\log e}{2}(m_1-m_0)^T\Sigma_0^{-1}(m_1-m_0)+\frac{1}{2}(\log |\Sigma_0|-\log |\Sigma_1|+\mathbf{tr}(\Sigma_0^{-1}\Sigma_1 - I)\log e)$$
$$D_{SKL}(\mathcal{N}(m_1,\Sigma_1)||\mathcal{N}(m_0,\Sigma_0))=D_{KL}(\mathcal{N}(m_1,\Sigma_1)||\mathcal{N}(m_0,\Sigma_0))+D_{KL}(\mathcal{N}(m_0,\Sigma_0)||\mathcal{N}(m_1,\Sigma_1))$$

## Other Methods
Noticeably, the optimization function $f$ in Donsker-Varadhan representation of kl-divergence $` D(P||Q)=\sup_f[\mathbb{E}_P(f(X))-\log\mathbb{E}_Q(e^{f(X)})] `$ can be simply seen as followed:<br>
$$f = \log\frac{dP}{dQ} $$
similarily, the D-V representation of $D(Q||P)$ can be optimized by $`f'=\log\frac{dQ}{dp}=-f`$.  
It is worth mentioning that here we assume $P\ll Q$ and $Q\ll P$, since either of them causes $D_{SKL}$ to go to infinty.
A trivial result is that:
$$D_{SKL}(P||Q)=\sup_{f\in\mathcal{C}_Q}[\mathbb{E}_P(f(X))-\mathbb{E}_Q(f(X))-\log\mathbb{E}_P(e^{-f(X)})-\log\mathbb{E}_Q(e^{f(X)})]$$

# More Experiments
see Gibbs_MINE folder
