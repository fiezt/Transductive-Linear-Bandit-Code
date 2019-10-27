import numpy as np
import logging
from RAGE import RAGE
from XY_ORACLE import XY_ORACLE
from XY_STATIC import XY_STATIC
from ALBA_YELIM import ALBA_YELIM
from LIN_GAP_ELIM import LIN_GAP_ELIM
import pickle
import os
import sys
import functools
import multiprocess

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

data_dir = os.path.join(os.getcwd(), 'direction_data_dir')

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

    
def sim_wrapper(item_list, seed_list, count):
    item_list[count].algorithm(seed_list[count])
    return item_list[count]


def many_arm_problem_instance(n):
    
    d = 2

    x = .1*np.random.rand(n-2)
    arm1 = [[np.cos(.78+x[i]), np.sin(.78+x[i])] + [0 for _ in range(d-2)] for i in range(n-2)]
    arm2 = [[1] + [0 for _ in range(d-1)]]
    arm3 = [[-.707, .707] + [0 for _ in range(d-2)]]
            
    X = np.vstack(arm1 + arm2 + arm3)
    
    theta_star = np.array([1, 0] + [0 for _ in range(d-2)]).reshape(-1, 1)
    
    return X, theta_star

count = 20
delta = 0.05
alpha = .1
eps = 0
sweep = [10000, 15000, 20000, 25000, 30000]
factor = 10
pool_num = 2
arguments = sys.argv[1:]

for n in sweep:

    np.random.seed(43)
    X_set = []
    theta_star_set = []
    for i in range(count):
        X, theta_star = many_arm_problem_instance(n)
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
        
        
    # ALBA
    if 'alba' in arguments:
        np.random.seed(43)
        instance_list = [ALBA_YELIM(X, theta_star, delta) for X, theta_star in zip(X_set, theta_star_set)]
        seed_list = list(np.random.randint(0, 100000, count))
        parallel_sim = functools.partial(sim_wrapper, instance_list, seed_list)
        pool = multiprocess.Pool(5)
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
        
        
