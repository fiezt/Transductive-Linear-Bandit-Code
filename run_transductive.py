import numpy as np
import logging
from RAGE import RAGE
from XY_ORACLE import XY_ORACLE
from XY_STATIC import XY_STATIC
import pickle
import os
import sys
import functools
import multiprocess

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

data_dir = os.path.join(os.getcwd(), 'transductive_dir')

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

def sim_wrapper(item_list, seed_list, count):
    item_list[count].algorithm(seed_list[count])
    return item_list[count]


def transductive_problem_instance(d, rad):
    
    theta_star = np.zeros((d, 1))
    theta_star[0, 0] = 2.0
    X = np.eye(2*d)
    Z = np.eye(2*d)[0:d,:]
    Zp = np.cos(rad)*np.eye(2*d)[0:d,:]+np.sin(rad)*np.eye(2*d)[d:2*d,:] 
    Z = np.vstack((Z, Zp))
    
    theta_star = np.zeros((2*d, 1))
    theta_star[0, 0] = 1
    
    return X, Z, theta_star

count = 20
delta = 0.05
rad = .1
sweep = [20, 40, 60, 80]
factor = 10
arguments = sys.argv[1:]

for d in sweep:

    np.random.seed(43)
    X, Z, theta_star = transductive_problem_instance(d, rad)

    # RAGE
    if 'rage' in arguments:  
        np.random.seed(43)
        instance_list = [RAGE(X, theta_star, factor, delta, Z) for i in range(count)]
        seed_list = list(np.random.randint(0, 100000, count))
        parallel_sim = functools.partial(sim_wrapper, instance_list, seed_list)
        pool = multiprocess.Pool(5)
        num_list = list(range(count))
        
        instance_list = []
        for instance in pool.imap_unordered(parallel_sim, num_list):
            try:
                instance_list.append(instance)
                print('Finished RAGE Instance')
                sample_complexity = np.array([instance.N for instance in instance_list])
                mean = np.mean(sample_complexity)
                se = np.std(sample_complexity)/np.sqrt(count)
                pickle.dump((mean, se), open(os.path.join(data_dir, "rage_" + str(d) + "_data.p"), "wb"))
                print('completed %d: mean %d and se %d' % (d, mean, se))
                pickle.dump(instance_list, open(os.path.join(data_dir, "rage_" + str(d) + ".p"), "wb"))
            except:
                print('error')
        
        pool.close()
        pool.join()
        
        
    # STATIC
    if 'static' in arguments:
        np.random.seed(43)
        instance_list = [XY_STATIC(X, theta_star, delta, Z) for i in range(count)]
        seed_list = list(np.random.randint(0, 100000, count))
        parallel_sim = functools.partial(sim_wrapper, instance_list, seed_list)
        pool = multiprocess.Pool(5)
        num_list = list(range(count))
        
        instance_list = []
        for instance in pool.imap_unordered(parallel_sim, num_list):
            try:
                instance_list.append(instance)
                print('Finished STATIC Instance')
                sample_complexity = np.array([instance.N for instance in instance_list])
                mean = np.mean(sample_complexity)
                se = np.std(sample_complexity)/np.sqrt(count)
                pickle.dump((mean, se), open(os.path.join(data_dir, "static_" + str(d) + "_data.p"), "wb"))
                print('completed %d: mean %d and se %d' % (d, mean, se))
                pickle.dump(instance_list, open(os.path.join(data_dir, "static_" + str(d) + ".p"), "wb"))
            except:
                print('error')
        
        pool.close()
        pool.join()
        
    # ORACLE
    if 'oracle' in arguments:
        np.random.seed(43)
        instance_list = [XY_ORACLE(X, theta_star, delta, Z) for i in range(count)]
        seed_list = list(np.random.randint(0, 100000, count))
        parallel_sim = functools.partial(sim_wrapper, instance_list, seed_list)
        pool = multiprocess.Pool(5)
        num_list = list(range(count))
        
        instance_list = []
        for instance in pool.imap_unordered(parallel_sim, num_list):
            try:
                instance_list.append(instance)
                print('Finished ORACLE Instance')
                sample_complexity = np.array([instance.N for instance in instance_list])
                mean = np.mean(sample_complexity)
                se = np.std(sample_complexity)/np.sqrt(count)
                pickle.dump((mean, se), open(os.path.join(data_dir, "oracle_" + str(d) + "_data.p"), "wb"))
                print('completed %d: mean %d and se %d' % (d, mean, se))
                pickle.dump(instance_list, open(os.path.join(data_dir, "oracle_" + str(d) + ".p"), "wb"))
            except:
                print('error')
        
        pool.close()
        pool.join()