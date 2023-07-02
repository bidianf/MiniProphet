# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
import logging
logger = logging.getLogger('pytho.models')


  
def get_changepoint_matrix(t, t_change, T, S):
    ''' Calculate changepoint matrix A with elements A_{t,j} = a_j(t) 
    defined on page 9 of "Forecasting at Scale"   
    
    Parameters
    ----------
    Explained in the docstring for "maximize_loglik". 
    
    Returns
    -------
    A : TxS matrix
    '''
    A = np.zeros(shape = (T,S))
    a_row = np.zeros(S)
    cp_idx = 0
    for i in range(T):
        while (cp_idx < S) and (t[i] >= t_change[cp_idx]):
            a_row[cp_idx] = 1
            cp_idx += 1
        A[i] = a_row
    return A

def logistic_gamma(k, m, delta, t_change, S):
    ''' Calculate vector gamma of offsets needed to preserve the 
    continuity of the logistic trend, with components gamma_j 
    defined on page 9 of "Forecasting at Scale"   
    
    Parameters
    ----------
    Explained in the docstring for "maximize_loglik". 
    
    Returns
    -------
    gamma :  np.array with S components
    '''
    gamma = np.zeros(S) 
    k_s = np.concatenate((np.atleast_1d(k),k + np.cumsum(delta)))
    m_pr = m
    for i in range(S):
        gamma[i] = (t_change[i] - m_pr) * (1 - k_s[i] / k_s[i + 1])
        m_pr += gamma[i] 
    return gamma
   
def logistic_trend(k, m, delta, t, cap, A, t_change, S):
    ''' Calculate the piecewise logistic trend
    in equation (3) on page 9 of "Forecasting at Scale"
    
    Parameters
    ----------
    Explained in the docstring for "maximize_loglik". 
    
    Returns
    -------
    np.array with T components
    '''   
    gamma = logistic_gamma(k, m, delta, t_change, S)
    inv_logit = (k + A @ delta) * (t - (m + A @ gamma))
    inv_logit = 1/(1+np.exp(-inv_logit))
    return cap * inv_logit 

def linear_trend(k, m, delta, t, A, t_change):
    ''' Calculate the piecewise linear trend
    in equation (4) on page 10 of "Forecasting at Scale"
    
    Parameters
    ----------
    Explained in the docstring for "maximize_loglik". 
    
    Returns
    -------
    np.array with T components
    '''   
    return (k + A @ delta) * t + (m + A @(-t_change * delta))
    
def flat_trend(m, T):
    ''' Generate a constant (flat)  trend
    
    Parameters
    ----------
    Explained in the docstring for "maximize_loglik". 
    
    Returns
    -------
    np.array with T components
    '''   
    return np.repeat(m, T)

    
def maximize_loglik(params, dat, **kwargs):
    """ Estimates the model parameters (MAP Maximum A Posteriori) directly
    via L-BFGS-B (without calling Stan). It does not perform sampling. 
    
    
    Parameters
    ----------
    params: A dictionary containing initial guess for model parameters 
        k (scalar): The initial trend slope/growth parameter.
        m (scalar): The initial intercept/offset parameter.
        delta (array of length S): Slope change at each of S changepoints.
        beta (array of length K): Coefficients for K seasonality features.
        sigma_obs (scalar): Noise level (standard deviation).
    
    dat: Dictionary, with keys and values as below
        T:      Number of time periods
        K:      Number of regressors
        t:      Time
        cap:    Capacities for logistic trend
        y:      Time series of interest
        S:      Number of changepoints
        t_change: The S times of trend changepoints
        X:      T x K matrix of Regressors
        sigmas: K x 1 Scale vector on seasonality prior
        tau:    Scale on changepoints prior
        trend_indicator: 0 for linear, 1 for logistic, 2 for flat
        s_a:    K x 1 vector of indicators of additive features
        s_m:    K x 1 vector of indicators of multiplicative features
        kwargs: Additional arguments passed to the optimization function
        
    Returns
    -------
    A dictionary containing the fitted model parameters, with the same
    structure as the input dictionary "params"
    """
  
    k = np.asarray(params['k']).flatten() 
    m = np.asarray(params['m']).flatten()
    delta = np.asarray(params['delta']).flatten()
    beta = np.asarray(params['beta']).flatten()
    sigma_obs = np.asarray(params['sigma_obs']).flatten()
    #To remove the positivity constraint on the standard deviation of 
    #errors and obtain an unconstrained problem, log-transform 
    log_sigma_obs = np.log(sigma_obs)
    
    T = dat['T']
    S = dat['S']
    K = dat['K']
    tau = dat['tau']
    trend_indicator = dat['trend_indicator']
    y = np.asarray(dat['y'])
    t = np.asarray(dat['t'])
    cap = np.asarray(dat['cap'])
    t_change = dat['t_change']
    s_a = dat['s_a']
    s_m = dat['s_m']
    X = np.array(dat['X'])
    sigmas = np.asarray(dat['sigmas'])
    
    params_vector_init = np.concatenate((k, m, delta, beta, log_sigma_obs))
       
    A = get_changepoint_matrix(t, t_change, T, S)
 
        
    def loglik(params_vector):
            """ Calculate the log-likelihood function, given parameters "params""
            and data "dat" 
            
            Parameters
            ----------
            params: An array that concatenates the following 1-dim arrays
                k   : The initial trend slope/growth parameter.
                m   : The initial intercept/offset parameter.
                delta: array of length S with slope change at each of S changepoints.
                beta : array of length K with coefficients for K seasonality features.
                sigma_obs (scalar): Noise level (standard deviation).
            Returns
            -------
            Scalar containing the value of the log likelihood
            """
            k = params_vector[0]
            m = params_vector[1]
            delta = params_vector[2:(2+S)]
            beta = params_vector[(2+S):(2+S+K)]
            log_sigma_obs = params_vector[-1]
            sigma_obs = np.exp(log_sigma_obs)
            
            if (trend_indicator == 0):
                trend = linear_trend(k, m, delta, t, A, t_change)
            elif (trend_indicator == 1):
                trend = logistic_trend(k, m, delta, t, cap, A, t_change, S)
            elif (trend_indicator == 2):
                trend = flat_trend(m, T);
             
            
            #Model y ~ normal(trend*(1+X*(beta*s_m))+X*(beta*s_a),sigma_obs)
            #Add contribution of priors to the posterior
            #k ~ normal(0, 5); Omit constant -np.log(2*np.pi)/2 -np.log(5)
            ll = -k**2/(2*5**2)
            #m ~ normal(0, 5); Omit constant -np.log(2*np.pi)/2 -np.log(5)
            ll += -m**2/(2*5^2)
            #delta ~ double_exponential(0, tau); Omit constant - np.log(2*tau)
            ll += -np.sum(np.abs(delta/tau))
            #sigma_obs ~ normal(0, 0.5); Omit -np.log(2*np.pi)/2 - np.log(.5)
            #MODIFY THE PRIOR< INCREASE FROM .5 to 15
            ll += -sigma_obs**2/(2*.5**2)
            #beta ~ normal(0, sigmas);
            #Omit - len(beta)* np.log(2*np.pi)/2 - np.sum(np.log(sigmas))
            ll += - np.sum(beta**2/(2*sigmas**2)) 
            
    
            #Add contribution of log likelihood conditional on parameters
            #Part of normal before exp. Omit -y.size * np.log((2*np.pi)**.5)
            ll += - log_sigma_obs * y.size
            #Create an auxiliary variable containing the mean of the series
            aux = trend * (1+X.dot(beta * s_m)) + X.dot(beta * s_a)
            #add the exponential part in the normal density
            ll += -np.sum((y-aux)**2)/(2*sigma_obs**2) 
            return -ll #scipy.optimize finds a minimum rather than a maximum
    
    
    logger.info("Start the minimization")
    if 'method' not in kwargs:
        kwargs.update({'method': 'BFGS'})
    if 'gtol' not in kwargs:
        kwargs.update({'gtol': 1e-5})
    if 'maxiter' not in kwargs:
        kwargs.update({'maxiter': 15000})
    if 'disp' not in kwargs:
        kwargs.update({'disp': True})
    
    try:
        res = minimize(loglik, params_vector_init,
                       method = kwargs['method'],
                       options = {'gtol': kwargs['gtol'],
                                  'maxiter': kwargs['maxiter'],
                                  'disp': kwargs['disp']                                  
                                   })
    
    except RuntimeError:
        logger.info(kwargs['method'] + ' failed. Use SLSQP as fallback.')
        res = minimize(loglik, params_vector_init, method = 'SLSQP')  
    
    logger.info("Minimization completed")
    logger.info("result: %s", res)     
    
    res_dict = {'k': res.x[0].reshape((1,-1)), 'm': res.x[1].reshape((1,-1)),
                'delta': res.x[2:(2+S)].reshape((1,-1)),
                'beta': res.x[(2+S):(2+S+K)].reshape((1,-1)),
                'sigma_obs': np.exp(res.x[-1].reshape((1,-1))) }
    return res_dict
    

    
