import numpy as np
def generate_sw_tuples_batch(batch_size,p=10,random_seed=None):
    """ Automatically sampling a batch of (S,W) turples.
 
    :param batch_size: number of (S,W) turples
    :param p: dimension of student model（可变）
    :param random_seed: fix samples
    :return: list of (S,W) tuples, S is a ndarray of N*(d+1), w is a ndarray of (p,)
    """


    """
    参数设置：dataset从一个维度为p0的teacher model中生成，而用来拟合的模型的维度则为p
    """
    original_random_state = np.random.get_state()
    np.random.seed(20230929)
    #gaussian distribution sampler: normal(mean=0.0, variance_sqrt=1.0, size=None) s = np.random.normal(0,1)
    #teacher model
    #dimension of x: d（固定不变）10->5
    d = 5
    #number of training samples（固定不变）100->20
    N = 20
    #dimension of hypothesis space
    p0 = 5 #dimension of teacher model（固定不变）
    p = p #dimension of student model（可变）#p=200
    p_max = 2000 #upper bound of dimension of student model（固定不变）
    #point-wise activate function f:tanh
    #variance of random noise added to y
    sigma = 0.1

    #random feature matrix
    F0 = np.random.normal(0,1,(d,p0)) #生成teacher model的random feature matrix F0（固定不变）
    F = np.random.normal(0,1,(d,p_max))[:,:p] #生成student model的random feature matrix F(d*p)。先按照p_max的大小生成，再根据当前维度p的不同截取前p列形成每个p对应的F。
        #注意，这个F对于一般的神经网络是可训练的，但是对于RFM为了简化直接设定为固定值。（固定不变）
    #teacher parameter w(p) with lambda = ? until each dim of Y~1e0
    lambda_ = 0.0001
    #w_0 = np.random.normal(0,sigma/np.sqrt(lambda_*N),p)
    w_0 = np.random.normal(0,1,p0) #生成teacher model的权重向量（固定不变）
    #w_0 = np.concatenate((w_0,np.zeros(5)),axis = 0)
    
    np.random.set_state(original_random_state)
    if random_seed:
        np.random.seed(random_seed)
    
    batch=[]
    for _ in range(batch_size):
        '''
        采样S：在采样每一个(S,W)元组的时候，最根本的就是在此处采样一个S=(X,Y)
             采样X(N*d)，X经过F0和w_0作用后再加上高斯扰动（这个扰动是必要的，不然构成确定映射不影响互信息）形成Y，
             Y与X一起构成了从dataset里采样的N个样本，对应了论文中的=>S （可变）
        '''
        #samples X(N*d) 
        X = np.random.normal(0,1,(N,d)) #生成N个d维随机向量，作为N条样本组成的训练集。
        #X after the random feature matrix
        X_rf0 = np.tanh(X.dot(F0)/np.sqrt(d))
        X_rf = np.tanh(X.dot(F)/np.sqrt(d))
        Y_pure = X_rf0.dot(w_0)
        Y = Y_pure + np.random.normal(0,sigma,N) #由X通过teacher model（也就是F0和w_0）生成的Y
        S = np.concatenate((X,Y.reshape(N,-1)),axis=1)  #S=(X,Y)
        # print(Y_pure)
        # print(Y)


        '''
        采样W：根据S训练得到一个W
        在S的条件下，以P_{W|S}（也就是gibbs分布）采样一个W，从而构成(S,W)元组。此后均由算法自动完成。
        '''
        #MCMC
        #empirical risk
        def L_S(w):
            diff = Y-X_rf.dot(w)
            Nloss = diff.dot(diff)
            return Nloss/N
        
        def grad_L_S(w):
            return -2/N*X_rf.T.dot(Y-X_rf.dot(w))
        #grad_L_S(w_0+0.1) #when N is small, diffusion.

        #minus log distribution: f
        beta = 10000  #param beta: also change h if change this
        sigma_q = 0.05 #param sigma_q: can be set according to N.

        def f(w): 
            return beta*L_S(w)+(1/2/sigma_q**2)*(w.dot(w))

        def grad_f(w):
            grad = beta*grad_L_S(w)+1/sigma_q**2*w
            return grad
    
        h = (10*1.41421356238/beta)**2  #stepsize h(sqrt h*beta=10*sqrt 2)
    
        w = np.random.normal(0,sigma_q,p)  #MCMC的初始分布，如何设置？传统SGLD直接设为了0。MCMC的收敛速度要求初始分布接近目标分布，于是这里选择gibbs里先验的高斯分布，有些问题有待考证
        for i in range(4000):  #迭代次数要使MCMC收敛到平稳分布才行。这里足够了吗？应该够了
            grad_f_w = grad_f(w)
            proposal_state = w-h*grad_f_w+np.sqrt(2*h)*np.random.normal(0,1,p)
            reject_thresh = min(1,np.exp(
                f(w)-f(proposal_state)+(1/4/h)*(np.linalg.norm(proposal_state-w+h*grad_f_w)**2-np.linalg.norm(w-proposal_state+h*grad_f(proposal_state))**2)
                ))

            U = np.random.rand(1)
            if U <= reject_thresh:
                w = proposal_state

        batch.append((S,w))
        
    return batch