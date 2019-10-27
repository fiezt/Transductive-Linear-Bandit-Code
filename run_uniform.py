import numpy as np
import logging
from RAGE import RAGE
from ALBA_YELIM import ALBA_YELIM
from LIN_GAP_ELIM import LIN_GAP_ELIM
from XY_ORACLE import XY_ORACLE
from XY_STATIC import XY_STATIC
import pickle
import os
import sys
import functools
import multiprocess

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

data_dir = os.path.join(os.getcwd(), 'uniform_data_dir')

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

def sim_wrapper(item_list, seed_list, count):
    item_list[count].algorithm(seed_list[count])
    return item_list[count]

def uniform_instance(n, d):

    #alpha = 0.4
    alpha = 0
    
    X = np.random.randn(n, d)
    X /= np.linalg.norm(X, axis=1).reshape(-1, 1)

    closest = []
    dists = []
    for i in range(len(X)):
        norms = np.linalg.norm(X[i]-X, axis=1)
        norms[i] = np.inf
        closest.append(np.argmin(norms))
        dists.append(np.min(norms))

    arm1 = np.argmin(dists)
    arm2 = closest[arm1]
    theta_star = X[arm1] + alpha*(X[arm2] - X[arm1])
    theta_star = theta_star.reshape(-1, 1)
        
    return X, theta_star

count = 20

delta = 0.05
eps = 0
sweep = [250, 500, 1000, 1500, 2000]
factor = 10
pool_num = 5
d = 9

arguments = sys.argv[1:]

for n in sweep:
    
    np.random.seed(43)
    X_set = []
    theta_star_set = []
    for i in range(count):
        X, theta_star = uniform_instance(n, d)
        X_set.append(X)
        rewards = X@theta_star
        rewards = sorted(rewards, reverse=True)
        print(rewards[0]-rewards[1])
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

                print('Finished Rage Instance')
                sample_complexity = np.array([instance.N for instance in instance_list])
                mean = np.mean(sample_complexity)
                se = np.std(sample_complexity)/np.sqrt(count)
                pickle.dump((mean, se), open(os.path.join(data_dir, "rage_" + str(n) + "_data.p"), "wb"))
                print('completed %d: mean %d and se %d' % (d, mean, se))
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
                print('completed %d: mean %d and se %d' % (d, mean, se))
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
                print('completed %d: mean %d and se %d' % (d, mean, se))
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

                print('Finished Static Instance')
                sample_complexity = np.array([instance.N for instance in instance_list])
                mean = np.mean(sample_complexity)
                se = np.std(sample_complexity)/np.sqrt(count)
                pickle.dump((mean, se), open(os.path.join(data_dir, "static_" + str(n) + "_data.p"), "wb"))
                print('completed %d: mean %d and se %d' % (d, mean, se))
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

                print('Finished Oracle Instance')
                sample_complexity = np.array([instance.N for instance in instance_list])
                mean = np.mean(sample_complexity)
                se = np.std(sample_complexity)/np.sqrt(count)
                pickle.dump((mean, se), open(os.path.join(data_dir, "oracle_" + str(n) + "_data.p"), "wb"))
                print('completed %d: mean %d and se %d' % (d, mean, se))
                pickle.dump(instance_list, open(os.path.join(data_dir, "oracle_" + str(n) + ".p"), "wb"))
            except:
                print('error')
        
        pool.close()
        pool.join()
