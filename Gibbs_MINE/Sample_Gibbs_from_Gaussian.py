import numpy as np
def sample_w(p,N,beta,sigma_q,B,Y,sample_num = 1):
    lambda_ = 1/(2*sigma_q**2*beta)
    sigma_sq = N/(2*beta)
    W = np.linalg.inv(lambda_*N*np.identity(p)+B.T.dot(B)).dot(B.T).dot(Y)
    Sigma = sigma_sq*np.linalg.inv(lambda_*N*np.identity(p)+B.T.dot(B))
    w_s = np.random.multivariate_normal(W,Sigma,sample_num)
    return w_s
def sample_w_ori(p,N,lambda_,sigma_sq,B,Y,sample_num = 1):
    W = np.linalg.inv(lambda_*N*np.identity(p)+B.T.dot(B)).dot(B.T).dot(Y)
    Sigma = sigma_sq*np.linalg.inv(lambda_*N*np.identity(p)+B.T.dot(B))
    w_s = np.random.multivariate_normal(W,Sigma,sample_num)
    return w_s
def minimized_w(p,N,lambda_,B,Y):
    W = np.linalg.inv(lambda_*N*np.identity(p)+B.T.dot(B)).dot(B.T).dot(Y)
    return W

def log_P(W,B,Y,N,beta,sigma_q):
    d = W.shape[0]
    lambda_ = 1/(2*sigma_q**2*beta)
    sigma_sq = N/(2*beta)
    mean = np.linalg.inv(lambda_*N*np.identity(d)+B.T.dot(B)).dot(B.T).dot(Y)
    Cov = sigma_sq*np.linalg.inv(lambda_*N*np.identity(d)+B.T.dot(B))
    res = -d/2*np.log(2*np.pi)-1/2*np.log(np.linalg.det(Cov))-1/2*(W-mean).T.dot(np.linalg.inv(Cov)).dot(W-mean)
    return res
    