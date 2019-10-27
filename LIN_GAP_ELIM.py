import numpy as np
import cvxpy as cvx
import itertools
import logging
import time
import pickle
import mosek


class LIN_GAP_ELIM(object):
    def __init__(self, X, theta_star, eps, delta, hyper=1):
        
        self.X = X
        self.K = len(X)
        self.d = X.shape[1]
        self.theta_star = theta_star
        self.opt_arm = np.argmax(X@theta_star)
        self.eps = eps
        self.delta = delta
        self.hyper = 1
        self.L = np.linalg.norm(X, axis=1).max()
        self.S = np.linalg.norm(theta_star)
        
        
    def algorithm(self, seed, greedy=False, binary=False):
        
        self.seed = seed
        np.random.seed(self.seed)
        
        self.binary = binary

        self.arm_counts = np.zeros(self.K)
        self.N = 0
        self.p_dict = {}
                          
        self.A = self.hyper*np.eye(self.d)
        self.A_inv = np.linalg.inv(self.A)
        self.det_A = np.linalg.det(self.A)
        self.b = np.zeros((self.d, 1))
        
        for arm_idx in range(self.K):
            arm = self.X[arm_idx, :, None]
            r = self.pull(arm)
            self.b += arm*r    
            self.det_A *= (1 + arm.T@self.A_inv@arm).item()
            self.A += arm@arm.T
            self.A_inv -= (self.A_inv@arm@arm.T@self.A_inv)/(1+arm.T@self.A_inv@arm)
            self.arm_counts[arm_idx] += 1
            self.N += 1
            
        self.theta_hat = self.A_inv@self.b
        
        while True:
            
            i, j, B = self.select_direction()
            y = self.X[i, :, None] - self.X[j, :, None]
            
            if B <= self.eps:
                break
            
            if greedy:
                arm_idx = self.select_greedy_arm(y)
            else:
                arm_idx = self.select_ratio_arm(i, j)
                
            arm = self.X[arm_idx, :, None]
        
            r = self.pull(arm)
            self.b += arm*r     
            self.det_A *= (1 + arm.T@self.A_inv@arm).item()
            self.A += arm@arm.T
            self.A_inv -= (self.A_inv@arm@arm.T@self.A_inv)/(1+arm.T@self.A_inv@arm)
            self.theta_hat = self.A_inv@self.b
            
            self.arm_counts[arm_idx] += 1
            self.N += 1

            if self.N % 10000 == 0:
                logging.info('\n\n')
                logging.info('B= %s' % str(B))
                logging.info('arm counts %s' % str(self.arm_counts))
                logging.info('total sample count %s' % str(self.N))
                logging.info('\n\n')
        
        del self.b
        del self.A
        del self.A_inv
        self.success = (self.opt_arm == i)
        logging.critical('Succeeded? %s' % str(self.success))
        logging.critical('Sample complexity %s' % str(self.N))
                    
    
    def select_greedy_arm(self, y):
                    
        arm_idx = np.argmin([-((y.T@self.A_inv@x@x.T@self.A_inv@y)/(1 + x.T@self.A_inv@x)).item() for x in self.X[:, :, None]])
                
        return arm_idx
    
    
    def select_ratio_arm(self, i, j):
        
        if (i, j) not in self.p_dict:
            self.p_dict[(i, j)] = self.get_arm_ratio(i, j)
            
        return np.argmin(self.arm_counts/(self.p_dict[(i, j)]+1e-10))
    
    
    def pull(self, arm):
        
        if self.binary:
            r = np.random.binomial(1, arm.T@self.theta_star)
        else:
            r = arm.T@self.theta_star + np.random.randn()
        
        return r
    
        
    def select_direction(self):
        
        r_hat = (self.X@self.theta_hat).flatten()
        i = np.argmax(r_hat)
        ucb = (r_hat - r_hat[i]) + self.conf(i)
        ucb[i] = -np.inf
        j = np.argmax(ucb)
        B = np.max(ucb)
        
        return i, j, B
    
    
    def conf(self, i, det=True):
        
        if det:
            C = np.sqrt(2*np.log(self.K**2*(np.sqrt(self.det_A))/(self.delta*np.sqrt(self.hyper**(self.d))))) + np.sqrt(self.hyper)*self.S
        else:
            C = np.sqrt(self.d*np.log(self.K**2*(1 + self.N*self.L**2/self.hyper)/self.delta)) + np.sqrt(self.hyper)*self.S
        
        try:
            U,D,V = np.linalg.svd(self.A_inv)
        except:
            U,D,V = np.linalg.svd(np.linalg.pinv(self.A))            
            
        Ahalf = U@np.diag(np.sqrt(D))@V.T
        
        variance = np.sqrt(np.sum(((self.X-self.X[i, :, None].T)@Ahalf)**2, axis=1))  
        
        return variance*C
    
    
    def get_arm_ratio(self, i, j):

        arm = self.X[i, :, None]
        arm_prime = self.X[j, :, None]
        y = arm - arm_prime
        
        # Construct the problem.        
        w = cvx.Variable((1,self.K))
        objective = cvx.Minimize(cvx.norm(w[0,:], 1))
        constraints = [y.T == w@self.X]
        prob = cvx.Problem(objective, constraints)
        prob.solve(cvx.MOSEK)
        w = w.value.flatten()
        p = np.abs(w)/np.abs(w).sum()

        return p