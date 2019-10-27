import numpy as np
import logging
from RAGE import RAGE
from ALBA_YELIM import ALBA_YELIM
from LIN_GAP_ELIM import LIN_GAP_ELIM
from XY_ORACLE import XY_ORACLE
from XY_STATIC import XY_STATIC
import itertools
import pickle
import os
import sys
import functools
import multiprocess


"""
To run this example, first download the yahoo data and extract the data for May 1st, and run process_yahoo.py
"""

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

data_dir = os.path.join(os.getcwd(), 'yahoo_data_dir')

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

def sim_wrapper(item_list, seed_list, count):
    item_list[count].algorithm(seed_list[count], binary=True)
    return item_list[count]


X = np.load("yahoo_features.npy")
Y = np.load("yahoo_targets.npy")

d = X.shape[1]
keep_idx = np.where(X[:, 0] > 0)[0]

X = X[keep_idx]
Y = Y[keep_idx]

theta_star = np.linalg.inv(X.T@X + .01*np.eye(36))@X.T@Y
theta_star = theta_star.reshape(-1, 1)

X_final = X

rewards = X_final@theta_star
best_arm = np.argmax(rewards)
max_reward = np.max(rewards)
suboptimal_arms = np.where(rewards.reshape(-1) < max_reward-0.01)[0]


def yahoo_problem_instance(num_x):
    
    while True:
        arm_set = np.random.choice(suboptimal_arms, num_x-1, replace=False).tolist() + [best_arm]
        X_subset = X_final[arm_set]
        if np.linalg.matrix_rank(X_subset) == 36:
            break
    return X_subset

count = 20
delta = 0.05
sweep = [40]
factor = 10.
eps = 0
pool_num = 10
arguments = sys.argv[1:]

for n in sweep:

    np.random.seed(43)    
    X_set = []
    theta_star_set = []
    for i in range(count):
        X = yahoo_problem_instance(n)
        X_set.append(X)
        theta_star_set.append(theta_star)

    # RAGE
    if 'rage' in arguments:  
        np.random.seed(43)
        instance_list = [RAGE(X, theta_star, factor, delta) for X, theta_star in zip(X_set, theta_star_set)]
        seed_list = list(np.random.randint(0, 100000, count))
        parallel_sim = functools.partial(sim_wrapper, instance_list, seed_list)
        pool = multiprocess.Pool(pool_num)
        num_list = list(range(count))
        
        instance_list = []
        for instance in pool.imap_unordered(parallel_sim, num_list):
            try:
                instance_list.append(instance)
                print('Finished RAGE Instance')
                sample_complexity = np.array([instance.N for instance in instance_list])
                mean = np.mean(sample_complexity)
                se = np.std(sample_complexity)/np.sqrt(count)
                pickle.dump((mean, se), open(os.path.join(data_dir, "rage_" + str(n) + "_data.p"), "wb"))
                print('completed %d: mean %d and se %d' % (n, mean, se))
                pickle.dump(instance_list, open(os.path.join(data_dir, "rage_" + str(n) + ".p"), "wb"))
            except:
                print('error')
        
        pool.close()
        pool.join()
        
        
    # LINGAP
    if 'lingap' in arguments:
        np.random.seed(43)
        instance_list = [LIN_GAP_ELIM(X, theta_star, eps, delta) for X, theta_star in zip(X_set, theta_star_set)]
        seed_list = list(np.random.randint(0, 100000, count))
        parallel_sim = functools.partial(sim_wrapper, instance_list, seed_list)
        pool = multiprocess.Pool(pool_num)
        num_list = list(range(count))
        instance_list = pool.map(parallel_sim, num_list)
        
        instance_list = []
        for instance in pool.imap_unordered(parallel_sim, num_list):
            try:
                instance_list.append(instance)

                print('Finished LINGAP Instance')
                sample_complexity = np.array([instance.N for instance in instance_list])
                mean = np.mean(sample_complexity)
                se = np.std(sample_complexity)/np.sqrt(count)
                pickle.dump((mean, se), open(os.path.join(data_dir, "lingap_" + str(n) + "_data.p"), "wb"))
                print('completed %d: mean %d and se %d' % (n, mean, se))
                pickle.dump(instance_list, open(os.path.join(data_dir, "lingap_" + str(n) + ".p"), "wb"))
            except:
                print('error')
        
        pool.close()
        pool.join()  
        
        
    if 'alba' in arguments:
        np.random.seed(43)
        instance_list = [ALBA_YELIM(X, theta_star, delta) for X, theta_star in zip(X_set, theta_star_set)]
        seed_list = list(np.random.randint(0, 100000, count))
        parallel_sim = functools.partial(sim_wrapper, instance_list, seed_list)
        pool = multiprocess.Pool(pool_num)
        num_list = list(range(count))
        
        instance_list = []
        for instance in pool.imap_unordered(parallel_sim, num_list):
            try:
                instance_list.append(instance)

                print('Finished ALBA Instance')
                sample_complexity = np.array([instance.N for instance in instance_list])
                mean = np.mean(sample_complexity)
                se = np.std(sample_complexity)/np.sqrt(count)
                pickle.dump((mean, se), open(os.path.join(data_dir, "alba_" + str(n) + "_data.p"), "wb"))
                print('completed %d: mean %d and se %d' % (n, mean, se))
                pickle.dump(instance_list, open(os.path.join(data_dir, "alba_" + str(n) + ".p"), "wb"))
            except:
                print('error')
        
        pool.close()
        pool.join() 
        
        
    # STATIC
    if 'static' in arguments:
        np.random.seed(43)
        instance_list = [XY_STATIC(X, theta_star, delta) for X, theta_star in zip(X_set, theta_star_set)]
        seed_list = list(np.random.randint(0, 100000, count))
        parallel_sim = functools.partial(sim_wrapper, instance_list, seed_list)
        pool = multiprocess.Pool(pool_num)
        num_list = list(range(count))
        
        instance_list = []
        for instance in pool.imap_unordered(parallel_sim, num_list):
            try:
                instance_list.append(instance)
                print('Finished STATIC Instance')
                sample_complexity = np.array([instance.N for instance in instance_list])
                mean = np.mean(sample_complexity)
                se = np.std(sample_complexity)/np.sqrt(count)
                pickle.dump((mean, se), open(os.path.join(data_dir, "static_" + str(n) + "_data.p"), "wb"))
                print('completed %d: mean %d and se %d' % (n, mean, se))
                pickle.dump(instance_list, open(os.path.join(data_dir, "static_" + str(n) + ".p"), "wb"))
            except:
                print('error')
        
        pool.close()
        pool.join()
        
    # ORACLE
    if 'oracle' in arguments:
        np.random.seed(43)
        instance_list = [XY_ORACLE(X, theta_star, delta) for X, theta_star in zip(X_set, theta_star_set)]
        seed_list = list(np.random.randint(0, 100000, count))
        parallel_sim = functools.partial(sim_wrapper, instance_list, seed_list)
        pool = multiprocess.Pool(pool_num)
        num_list = list(range(count))
        
        instance_list = []
        for instance in pool.imap_unordered(parallel_sim, num_list):
            try:
                instance_list.append(instance)
                print('Finished ORACLE Instance')
                sample_complexity = np.array([instance.N for instance in instance_list])
                mean = np.mean(sample_complexity)
                se = np.std(sample_complexity)/np.sqrt(count)
                pickle.dump((mean, se), open(os.path.join(data_dir, "oracle_" + str(n) + "_data.p"), "wb"))
                print('completed %d: mean %d and se %d' % (n, mean, se))
                pickle.dump(instance_list, open(os.path.join(data_dir, "oracle_" + str(n) + ".p"), "wb"))
            except:
                print('error')
        
        pool.close()
        pool.join()
