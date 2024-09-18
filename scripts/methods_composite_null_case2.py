import numpy as np 
import os 
import pickle
from glob import glob
from scipy.stats import permutation_test

def test_by_betting(seq1, seq2, d, epsilon, alpha): 
    """Run testing by betting algorithm with ONS betting strategy 
    on given sequences"""
    
    wealth_A = 1
    wealth_B = 1
    wealth_hist_A = [1]
    wealth_hist_B = [1] 
    const = 2 / (2 - np.log(3))
    lambd_a = 0 
    lambd_b = 0 
    zat2 = 0 
    zbt2 = 0
    for t in range(1,min(len(seq1), len(seq2))):        
        At = 1 - lambd_a*(seq1[t] - seq2[t]- epsilon)
        Bt = 1 - lambd_b*(seq2[t] - seq1[t]- epsilon)
        wealth_A = wealth_A * At
        wealth_B = wealth_B * Bt 
        wealth_hist_A.append(wealth_A)
        wealth_hist_B.append(wealth_B)
        if wealth_A > 2/alpha: 
            return wealth_hist_A, 'reject'
        elif wealth_B > 2/alpha: 
            return wealth_hist_B, 'reject'
            
        # Update lambda via ONS  
        a = seq1[t] - seq2[t] - epsilon
        b = seq2[t] - seq1[t] - epsilon
        z_a = a / (1 - lambd_a*a)
        z_b = b / (1 - lambd_b*b)
       
        zat2 += z_a**2
        zbt2 += z_b**2
 
        lambd_a = max(min(lambd_a - const*z_a/(1 + zat2), 0), -d)
        lambd_b = max(min(lambd_b - const*z_b/(1 + zbt2), 0), -d)
    U = np.random.uniform()
    if wealth_A > (2*U)/alpha: 
        return wealth_hist_A, 'reject'
    elif wealth_B > (2*U)/alpha: 
        return wealth_hist_B, 'reject'
        
    return wealth_hist_B, 'sustain'


def betting_experiment(seq1, seq2, alphas, iters, shift_time=None): 
    """Helper to run a batch of sequential tests via betting"""
    
    results = []
    rejections = []
    for i in range(iters): 
        np.random.seed(i)
        s1, s2 = shuffle_sequences(seq1, seq2, shift_time)

        q1 = np.array(s1)
        q2 = np.array(s2)

        #estimate d_t
        s1_pre = q1[:10]
        s2_pre = q2[:10]
        diff = np.abs(s1_pre[:, np.newaxis] - s2_pre)
        twice_dt = 2*np.max(diff) #twice the maximum value of previous 10 points
        d = 1/(2*twice_dt)

        #estimate epsilon
        q1_pre=q1[:20]
        epsilon_lst=[]
        for j in range(1000):
            np.random.shuffle(q1_pre)
            a=q1_pre[:10]
            b=q1_pre[10:20]
            epsilon_lst.append(2*np.abs(np.mean(a)-np.mean(b)))
        epsilon=np.mean(epsilon_lst) 
    

        s1_post=q1[10:]
        s2_post=q2[10:]
        taus = []
        rejects = []
        for alpha in alphas: 
            wealth, reject = test_by_betting(s1_post, s2_post, d, epsilon=epsilon, alpha=alpha)
            real_tau = len(wealth)+10
            taus.append(real_tau)
            rejects.append(True if reject == 'reject' else False)
        results.append(taus)
        rejections.append(rejects)
        
    return results, rejections

def shuffle_sequences(seq1, seq2, shift_time=None):
    """Shuffle sequences while ensure the same order for different methods."""
    if shift_time is not None:
        # Split sequences into parts before and after shift_time
        s1_pre, s1_post = seq1[:shift_time], seq1[shift_time:]
        s2_pre, s2_post = seq2[:shift_time], seq2[shift_time:]
        
        # Shuffle each part separately
        np.random.shuffle(s1_pre)
        np.random.shuffle(s1_post)
        np.random.shuffle(s2_pre)
        np.random.shuffle(s2_post)
        
        # Concatenate shuffled parts back together
        s1 = np.concatenate((s1_pre, s1_post))
        s2 = np.concatenate((s2_pre, s2_post))
    else:
        # Shuffle the entire sequences
        s1, s2 = np.copy(seq1), np.copy(seq2)
        np.random.shuffle(s1)
        np.random.shuffle(s2)
    return s1, s2

# Case-2
def test_stat(x, y, axis):
    """Test statistic for permutation test"""
    return np.abs(np.mean(x, axis=axis) - np.mean(y, axis=axis))

def perm_test(seq1, seq2, e, p):    
    """Perform permutation test with a threshold e"""
    
    res = permutation_test((seq1, seq2), test_stat, vectorized=True,
                           n_resamples=2000, alternative='two-sided', permutation_type='samples')
    
    if res.statistic > e and res.pvalue <= p: 
        return True 
    return False


def seq_perm_test(seq1, seq2, e, k, p=0.05, bonferroni=False): 
    """Perform sequential permutation test at given significance level (p). 
    Hypothesis is tested after each k observations. 
    If indicated, perform bonferroni like correction by dividing required significance
    by 2 each batch"""

    l = min(len(seq1), len(seq2))
    for i in range(int(l/k)): 
        pi = p / 2**(i+1) if bonferroni else p
        if perm_test(seq1[i*k:k*(i+1)], seq2[i*k:k*(i+1)], e, pi): 
            # print('Reject')
            return k*(i+1), 'reject'
    return l+10, 'sustain'


def seq_perm_test_experiment(seq1, seq2, alphas, iters, k, bonferroni=False, shift_time=None): 
    """Helper to run batch of sequential permutation tests. k indicates window size"""
    
    results = []
    rejections = []
    s1, s2 = seq1, seq2
    for i in range(iters): 
        taus, rejects = [], []
        np.random.seed(i)
        s1, s2 = shuffle_sequences(seq1, seq2, shift_time)
        
        q1 = np.array(s1)
        #estimate epsilon
        q1_pre=q1[:20]
        epsilon_lst=[]
        for j in range(1000):
            np.random.shuffle(q1_pre)
            a=q1_pre[:10]
            b=q1_pre[10:20]
            epsilon_lst.append(2*np.abs(np.mean(a)-np.mean(b)))
        
        epsilon=np.mean(epsilon_lst) 

        for alpha in alphas: 
            steps, reject = seq_perm_test(s1, s2, e=epsilon, p=alpha, k=k, bonferroni=bonferroni)
            taus.append(steps)
            rejects.append(True if reject == 'reject' else False)
        results.append(taus)
        rejections.append(rejects)
        
    return results, rejections
    
