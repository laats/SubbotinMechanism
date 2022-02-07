# -*-Python-*-
################################################################################
#
# File:         SLDP.py
# RCS:          $Header: $
# Description:  Compute the smallest scale s such that the mechanism that adds
#               noise variable sX is (eps, del)-differentially private
#               for X being either a Subbotin(r) or Logistic variable.
#
#               For details see the paper
#               "Differential privacy for symmetric log-concave mechanisms"
#               by Staal A. Vinterbo presented at AISTATS 2022.
#
#               If you use this code, please cite the paper. 
#
#               Try $ python SLDP.py
#               to run a few tests.
#               
# Author:       Staal Vinterbo
# Created:      Wed Jul 14 12:39:19 2021
# Modified:     Mon Feb  7 17:01:56 2022 (Staal Vinterbo) staal@wiggly.local
# Language:     Python 3.5 or higher
# Package:      N/A
# License:      Licenced under the EUPL-1.2 or later.
#               The licence text and information about it can be found here:
#               https://ec.europa.eu/info/european-union-public-licence_en
#
# (c) Copyright 2022, Staal Vinterbo, all rights reserved.
#
################################################################################
'''(eps, del)-differentially private noise scales for Subbotin and Logistic mechanisms'''

__all__ = ['EDSubbotinScale', 'EDLaplaceScale', 'EDLogisticScale']

import sys
import numpy as np
from warnings import warn, catch_warnings, simplefilter
from scipy.special import gamma, gammaincc


#### (eps, del)-differential privacy distribution scales

def EDSubbotinScale(r : float, epsilon : float, delta : float, Delta :float = 1.0, tol : float = 1e-12) -> float:
    '''Subbotin_r scale needed for (epsilon, delta)-differential privacy

    Inputs:
      r              -- Subbotin parameter r >= 1 
      epsilon, delta -- privacy parameters, eps >= 0, delta in [0,1)
      Delta          -- global sensitivity, Delta > 0
      tol            -- numeric tolerance for root finding

    Output: real number > 0 : scale of Logistic distribution'''

    if r == 1:
        return EDLaplaceScale(epsilon, delta, Delta)

    if r < 1:
        raise RuntimeError('EDSubbotinScale: r < 1 not allowed.')

    checkParms('EDSubbotinScale', epsilon, delta, Delta)

    # Subbotin_r CDF
    uIGamma = lambda s, x : gamma(s) * gammaincc(s, x)
    den = (2*gamma(1/r))
    F = lambda x : 0.5 + np.sign(x) * (0.5 - uIGamma(1/r, (np.abs(x)**r)/r)/den)

    def f(s):
        tstar = t(r, s, epsilon, Delta, tol = tol)
        return g(s, tstar, epsilon, Delta, F) - delta

    s0, s1 = find_bracket(f, EDLaplaceScale(epsilon, delta, Delta))
    scale = bisection(f, bracket=(s0, s1), tol=tol)
    assert(np.abs(f(scale)) <= tol)
    return scale 


def EDLaplaceScale(epsilon : float, delta : float, Delta : float = 1.0) -> float:
    '''Laplace scale needed for (epsilon, delta)-differential privacy.

    Inputs:
      epsilon, delta -- eps >= 0, delta in [0,1)
      Delta          -- global sensitivity, Delta > 0

    Output: real number > 0 : scale of Laplace distribution'''

    checkParms('EDLaplaceScale', epsilon, delta, Delta, delta0=True)

    return Delta / (epsilon - 2.0 * np.log(1.0 - delta))


def EDLogisticScale(epsilon : float, delta : float, Delta : float = 1.0) -> float:
    '''Logistic scale needed for (epsilon, delta)-differential privacy.

    Inputs:
      epsilon, delta -- privacy parameters, eps >= 0, delta in [0,1)
      Delta          -- global sensitivity, Delta > 0

    Output: real number > 0 : scale of Logistic distribution'''

    checkParms('EDLogisticScale', epsilon, delta, Delta, delta0=True)
    
    numer : float = (np.exp(epsilon/2.0) +
                     np.sqrt(delta*(np.exp(epsilon) +
                                    delta - 1.0)))
    
    return Delta / (2.0 * np.log(numer / (1.0 - delta)))



### optimization code -- here might be dragons

def g(s, t, epsilon, Delta, F):
    '''the left hand side of the DP criterion for CDF F'''
    return F((Delta - t)/s) - np.exp(epsilon) * F(-t/s)


def t(r, s, epsilon, Delta, tol = 1e-16):
    '''compute t parameter to go into function g'''

    # want to find largest z such that psi_r(z) <= epsilon
    psi_r = lambda z : s ** (-r) * (z ** r - np.abs(-z + Delta) ** r) / r
    f  = lambda z : psi_r(z) - epsilon

    # using the root for r=2 as initial guess as it exists in closed form
    iniguess = lambda s : (2 * s ** 2 * epsilon + Delta ** 2) / Delta / 2

    # want to catch numerical problems in psi_r
    with catch_warnings(record=True) as w:
        simplefilter("always")
        xa, xc = find_bracket(lambda t : -f(t), iniguess(s))
        if len(w) > 0 or xa is None:
            raise RuntimeError('EDSubbotinScale find_bracket problem for '+
                             'r/s/f(xa)/f(xc) = {}/{}/{}/{}'.format(r,s,f(xa), 
                                                                    f(xc)))
        
    root = bisection(f, (xa, xc), tol=tol)
    if root is None:
        raise RuntimeError('EDSubbotinScale bisection problem for '+
                             'r/s/D = {}/{}/{}'.format(r,s,Delta))
    return root

## optmization is by bisection

def find_bracket(f, x0 = 1, N = 128):
    '''find bisection bracket for decreasing function f with postive root

    x0 -- starting point
    N -- maximal number of doublings/halvings to try
    '''

    y = f(x0)
    s0 = np.sign(y)
    if y == 0: return (x0, x0)
    
    stepf = 1/2 if s0 < 0 else 2

    x = x0
    for _ in range(N):
        x = x*stepf
        y = f(x)
        if np.sign(y) != s0:
            break
    if np.sign(y) == s0:
        return (None, None) # defer failure handling
    xprev = x/stepf
    return (xprev, x) if x > xprev else (x, xprev)


def bisection(f, bracket, tol, b = 0, N = 128):
    '''bisection algorithm that can guarantee signum(b) * f(found_root) >= 0

    parameters:
      f: continous and monotone function f(x) 
         that changes sign once in the bracket interval
      bracket: closed interval [bracket[0], bracket[1]]. Note: bisection returns
               bracket[0] if bracket[0] == bracket[1]
      tol: the width of tolerated final interval
      b: if 0 choose midpoint of last interval, if positive choose upper bound,
         if negative, choose lower bound
      N: maximum number of bisections to attempt

    return value: if b != 0 then return the endpoint of the last 
                  interval that yields signum(b) * f(found_root) >= 0 
                  else return midpoint of last interval
    '''
    assert(tol > 0 and N > 0)
    
    l, r = bracket[0], bracket[1]
    if l == r: return l
    lsign = np.sign(f(l))

    # check that bracket endpoint signs differ
    if not lsign*f(r) < 0:
        return None # defer error handling

    # so we can treat lsign*f as an increasing function
    lsign = -lsign 

    for _ in range(N):
        if r - l <= tol:
            break
        m = (l + r)/2
        fm = lsign * f(m)
        if fm == 0: return m
        if fm > 0:
            r = m
        else:
            l = m

    if b == 0:
        # return midpoint
        return (r + l)/2
            
    # return the appropriate endpoint if b != 0
    if b > 0: return r if lsign > 0 else l
    if b < 0: return l if lsign > 0 else r


def checkParms(name, epsilon, delta, Delta, delta0 = False):
    '''check for legal input parameters'''
    strs = []
    if epsilon < 0:
        strs += ['epsilon = {:e} < 0'.format(epsilon)]
    if delta0 and delta < 0:
        strs += ['delta {:e} < 0'.format(delta)]
    if (not delta0) and delta <= 0:
        strs += ['delta {:e} <= 0'.format(delta)]
    if Delta <= 0:
        strs += ['Delta {:e} <= 0'.format(Delta)]
    if len(strs) > 0:
        raise RuntimeError('{}: parameter violations:\n\t'.format(name) + 
                           '\n\t'.join(strs))



if __name__ == "__main__":

    import scipy.stats as ss
    Phi = ss.norm.cdf
    
    def gR(s, epsilon, delta, Delta=1):
        '''compute Phi(...) - exp(epsilon) Phi(...)'''
        d2s = Delta/(2*s)
        esd = epsilon * s / Delta
        return Phi(d2s - esd) - np.exp(epsilon)*Phi(-d2s - esd)
    
    print('''Performing tests of scale computations:

* check that a bad setting of parameters fails
* see how large Subbotin(r) parameter r can get before failure
* compare scales for the gaussian mechanism computed by our and 
  Balle and Wang's algorithm if that is available.
''')
    
    print('Bad parameters test (should give error):')
    try:
        print(EDSubbotinScale(3, -1, 0, 0))
    except RuntimeError as er:
        print('caught error:', er)

    print('---')
    print('Max r test for epsilon = Delta = 1 and delta = 1.e-8:')
    for i in range(50, 100):
        try:
            s = EDSubbotinScale(i, 1, 10.0**(-8))
        except Exception as er:
            print('Managed to compute Subbotin({}) scale {}'.format(i-1, s))
            print('failed for r = {}'.format(i))
            print('with ' + str(er))
            break
        if i % 5 == 0:
            print('r = {} ...'.format(i))
    print('recomputing {}...'.format(i-1))
    print(EDSubbotinScale(i-1, 1, 10.0**(-8)), 'ok.')

    try:
        from agmexample import calibrateAnalyticGaussianMechanism as scaleBW
    except:
        print('''
To compare with Balle and Wang's algorithm for computing the 
optimal Gaussian scale, please download agm-example.py from 
        https://github.com/BorjaBalle/analytic-gaussian-mechanism
and make it available as "agmexample"  
(e.g., download, rename to agmexample.py, and place in the same
directory as this file). 

Setting np.random.seed(100), the following output is produced 
on my machine after following the instructions above:

In the following, R and RBW are the right hand side of
Phi(...) - exp(epsilon)Phi(...) <= delta for scales 
computed by our and Balle and Wang's algorithms, respectively. 
These ratios, computed on random (epsilon, delta) 
pairs and Delta = 1, should ideally be 1 or at least not much smaller than 1.
Values below are rounded for readability.
(epsilon, delta):		 delta/R 	 delta/RBW
(0.543, 1.98e-09): 		 1.0000 	 0.9995
(0.278, 1.81e-15): 		 0.9937 	 2.7321
(0.425, 1.17e-12): 		 1.0000 	 1.3833
(0.845, 1.82e-05): 		 1.0000 	 1.0000
(0.005, 1.27e-04): 		 1.0000 	 1.0000
(0.122, 1.43e-11): 		 1.0000 	 1.0594
(0.671, 1.94e-17): 		 2.3864 	 3.8184
(0.826, 1.82e-10): 		 1.0000 	 1.0026
(0.137, 1.34e-16): 		 1.0434 	 0.2475
(0.575, 1.18e-06): 		 1.0000 	 1.0000
(0.891, 1.37e-04): 		 1.0000 	 1.0000
(0.209, 1.01e-12): 		 1.0000 	 1.7463
(0.185, 1.25e-06): 		 1.0000 	 1.0000
(0.108, 1.80e-09): 		 1.0000 	 0.9997
(0.220, 1.02e-14): 		 1.0008 	 0.0493
        ''')
    else:
        # np.random.seed(100) # used to create demo output
        n = 15 
        scale = lambda e,d : EDSubbotinScale(2, e, d, 1)
        reps = lambda n : np.random.random(n)
        rdel = lambda n : (
            (1 + np.random.random(n))
            * 10.**(-np.random.randint(4, 18, size=n)))

        epsdels = zip(reps(n), rdel(n))
        print('''---\nIn the following, R and RBW are the right hand side of
Phi(...) - exp(epsilon)Phi(...) <= delta, computed by our and Balle and Wang's 
algorithms, respectively. These ratios, computed on random (epsilon, delta)
pairs and Delta = 1, should ideally be 1 or at least not much smaller than 1.
Values below are rounded for readability.''')
        print('(epsilon, delta):\t\t delta/R \t delta/RBW')
        for eps, delta in epsdels:
            s = scale(eps, delta)
            sbw = scaleBW(eps, delta, 1)
            d = gR(s, eps, delta)
            dbw = gR(sbw, eps, delta)
            print('({:.3f}, {:.2e}): \t\t {:.4f} \t {:.4f}'.format(eps, delta,
                                                      delta/d, delta/dbw))

        

    
    
    
    
