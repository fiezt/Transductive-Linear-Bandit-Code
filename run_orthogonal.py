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

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

data_dir = os.path.join(os.getcwd(), 'orthogonal_data_dir')

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

def sim_wrapper(item_list, seed_list, count):
    item_list[count].algorithm(seed_list[count])
    return item_list[count]


def orthogonal_design_problem_instance(D, num_sparse=None):

    alpha1 = 1
    alpha2 = 0.5
    variants = list(range(D))
    individual_index_dict = {}

    count = 0
    for key in variants:
        individual_index_dict[key] = count
        count += 1

    pairwise_index_dict = {}
    count = 0
    pairs = []
    for pair in itertools.combinations(range(D), 2):
        pairs.append(pair)
        key1 = pair[0]
        key2 = pair[1]
        pairwise_index_dict[(key1, key2)] = count
        count += 1

    individual_offset = 1
    pairwise_offset = 1 + len(individual_index_dict)
    num_features = 1 + len(individual_index_dict) + len(pairwise_index_dict)
    num_arms = 2**D

    combinations = list(itertools.product([-1, 1], repeat=D))

    X = -np.ones((num_arms, num_features))

    for idx in range(num_arms):
        bias_feature_index = [0]
        individual_feature_index = [individual_offset + individual_index_dict[i] for i, val in enumerate(combinations[idx]) if val == 1]
        pairwise_feature_index = [pairwise_offset + pairwise_index_dict[pair] for pair in pairs if combinations[idx][pair[0]] == combinations[idx][pair[1]]]
        feature_index = bias_feature_index + individual_feature_index + pairwise_feature_index
        X[idx, feature_index] = 1

    while True:
        theta_star = np.random.randint(-10, 10, (num_features, 1))/100
        theta_star[individual_offset:pairwise_offset] = alpha1*theta_star[individual_offset:pairwise_offset]
        theta_star[pairwise_offset] = alpha2*theta_star[pairwise_offset]

        if num_sparse:
            sparse_index = np.zeros(D)
            sparse_index[np.random.choice(len(sparse_index), num_sparse, replace=False)] = 1
            bias_feature_index = [0]
            individual_feature_index = [individual_offset + individual_index_dict[i] for i, val in enumerate(sparse_index) if val == 1]
            pairwise_feature_index = [pairwise_offset + pairwise_index_dict[pair] for pair in pairs if sparse_index[pair[0]] == 1 and sparse_index[pair[1]] == 1]
            feature_index = bias_feature_index + individual_feature_index + pairwise_feature_index
            theta_star[~np.array(feature_index)] = 0

        rewards = (X@theta_star).reshape(-1)
        top_rewards = sorted(rewards, reverse=True)[:2]
        
        if top_rewards[0] - top_rewards[1] < 10e-6:
            continue
        else:
            break
            
    return X, theta_star

count = 50
delta = 0.05
sweep = [3, 4, 5, 6]
factor = 10.
pool_num = 20
eps = 0
arguments = sys.argv[1:]

for d in sweep:

    np.random.seed(43)    
    X_set = []
    theta_star_set = []
    for i in range(count):
        X, theta_star = orthogonal_design_problem_instance(d)
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
                pickle.dump((mean, se), open(os.path.join(data_dir, "rage_" + str(d) + "_data.p"), "wb"))
                print('completed %d: mean %d and se %d' % (d, mean, se))
                pickle.dump(instance_list, open(os.path.join(data_dir, "rage_" + str(d) + ".p"), "wb"))
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
                pickle.dump((mean, se), open(os.path.join(data_dir, "lingap_" + str(d) + "_data.p"), "wb"))
                print('completed %d: mean %d and se %d' % (d, mean, se))
                pickle.dump(instance_list, open(os.path.join(data_dir, "lingap_" + str(d) + ".p"), "wb"))
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
                pickle.dump((mean, se), open(os.path.join(data_dir, "alba_" + str(d) + "_data.p"), "wb"))
                print('completed %d: mean %d and se %d' % (d, mean, se))
                pickle.dump(instance_list, open(os.path.join(data_dir, "alba_" + str(d) + ".p"), "wb"))
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
                pickle.dump((mean, se), open(os.path.join(data_dir, "oracle_" + str(d) + "_data.p"), "wb"))
                print('completed %d: mean %d and se %d' % (d, mean, se))
                pickle.dump(instance_list, open(os.path.join(data_dir, "oracle_" + str(d) + ".p"), "wb"))
            except:
                print('error')
        
        pool.close()
        pool.join()
