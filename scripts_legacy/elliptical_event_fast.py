"""Deterministic three-dimensional elliptical CDF path integration.

This research implementation is initially a candidate accelerator. It must be
validated against independent high-precision randomized CDFs before integration
into the forecasting workflow. Finite t at R=I is NOT independence: its common
radial scale is integrated explicitly. No sample outcomes enter this module.
"""
from __future__ import annotations
from functools import lru_cache
import time
import numpy as np
from scipy.integrate import quad_vec
from scipy.special import ndtr, stdtr, gammaln
from scipy.stats import norm, t
from numpy.polynomial.legendre import leggauss

PAIRS = ((0,1,2),(0,2,1),(1,2,0))


@lru_cache(maxsize=12)
def nodes(order):
    x,w = leggauss(order)
    return (x+1)/2,w/2


def identity_cdfs(a,df,eps=1e-7):
    marginal = norm.cdf(a) if np.isinf(df) else t.cdf(a,df)
    if np.isinf(df):
        return np.column_stack([marginal[:,i]*marginal[:,j] for i,j,_ in PAIRS]+[marginal.prod(axis=1)]),0.
    # x=S/2 is Gamma(df/2,1). Normal thresholds share sqrt(S/df).
    shape = df/2
    def integrand(x):
        if x<=0: return np.zeros((len(a),4))
        weight = np.exp((shape-1)*np.log(x)-x-gammaln(shape))
        p = ndtr(a*np.sqrt(2*x/df))
        return weight*np.column_stack([p[:,i]*p[:,j] for i,j,_ in PAIRS]+[p.prod(axis=1)])
    result,error = quad_vec(integrand,0.,np.inf,epsabs=eps,epsrel=eps,norm='max',limit=1000)
    return result,float(error)


def path_corrections(a,R,df,order):
    """Integrate derivative along SPD path R(s)=I+s(R-I), s in[0,1].

    Normal derivative is bivariate density times conditional normal CDF.
    Integrating that derivative over the common chi-square scale gives the t
    kernel (1+Q/df)^(-df/2)/(2*pi*sqrt(1-rho^2)), with a t_df conditional CDF.
    That df is not the ordinary t_(df+2) conditional density formula: there is
    no scale-squared Jacobian in the CDF correlation derivative.
    """
    s,w = nodes(order)
    corrections = np.zeros((len(a),4))
    for slot,(i,j,k) in enumerate(PAIRS):
        rho = R[:,i,j,None]*s
        ri = R[:,i,k,None]*s
        rj = R[:,j,k,None]*s
        variance = 1-rho*rho
        ai,aj,ak = a[:,i,None],a[:,j,None],a[:,k,None]
        Q = np.maximum((ai*ai-2*rho*ai*aj+aj*aj)/variance,0.)
        if np.isinf(df): kernel = np.exp(-Q/2)/(2*np.pi*np.sqrt(variance))
        else: kernel = np.exp(-df/2*np.log1p(Q/df))/(2*np.pi*np.sqrt(variance))
        multiplier = R[:,i,j,None]*w
        corrections[:,slot] = np.sum(multiplier*kernel,axis=1)
        mean = ((ri-rho*rj)*ai+(rj-rho*ri)*aj)/variance
        conditional_variance = 1-(ri*ri-2*rho*ri*rj+rj*rj)/variance
        if np.any(conditional_variance<=0): raise ValueError('Non-SPD conditional correlation along path')
        z = (ak-mean)/np.sqrt(conditional_variance)
        if np.isinf(df): conditional_cdf = ndtr(z)
        else: conditional_cdf = stdtr(df,z/np.sqrt(1+Q/df))
        corrections[:,3] += np.sum(multiplier*kernel*conditional_cdf,axis=1)
    return corrections


def fast_four_cdfs(V,R,df=np.inf,tolerance=1e-5,initial_order=32,max_order=1024):
    V,R = np.asarray(V,float),np.asarray(R,float)
    if V.ndim!=2 or V.shape[1]!=3 or not np.isfinite(V).all() or np.any((V<=0)|(V>=1)):
        raise ValueError('Accelerator initially requires three interior finite CDF thresholds')
    if R.ndim==2: R=np.broadcast_to(R,(len(V),3,3))
    if R.shape!=(len(V),3,3) or not np.isfinite(R).all(): raise ValueError('Invalid R array')
    if not np.allclose(R,R.transpose(0,2,1),rtol=0,atol=1e-12) or not np.allclose(np.diagonal(R,axis1=1,axis2=2),1,rtol=0,atol=1e-12):
        raise ValueError('R must be a symmetric correlation matrix')
    if np.linalg.eigvalsh(R).min()<=0: raise ValueError('R must be SPD')
    if not (df == np.inf or (np.isfinite(df) and df>0)): raise ValueError('Positive finite df or positive infinity required')
    if not np.isfinite(tolerance) or not 0<tolerance<.01: raise ValueError('Tolerance must lie in (0,.01)')
    if (initial_order<4 or initial_order&(initial_order-1) or
            max_order<2*initial_order or max_order&(max_order-1)):
        raise ValueError('Require power-of-two initial order >=4 and maximum >= twice initial order')
    a = norm.ppf(V) if np.isinf(df) else t.ppf(V,df)
    base,error = identity_cdfs(a,df,eps=min(tolerance/20,1e-7))
    lower = path_corrections(a,R,df,initial_order)
    order = initial_order*2
    higher = path_corrections(a,R,df,order)
    discrepancy = np.max(np.abs(higher-lower),axis=1)
    orders = np.full(len(V),order,int)
    pending = discrepancy>tolerance/2
    while pending.any() and order<max_order:
        order *= 2
        replacement = path_corrections(a[pending],R[pending],df,order)
        discrepancy[pending] = np.max(np.abs(replacement-higher[pending]),axis=1)
        higher[pending] = replacement;orders[pending] = order
        pending = discrepancy>tolerance/2
    if pending.any(): raise RuntimeError(f'Path integration unresolved on {int(pending.sum())} rows')
    result = base+higher
    if np.any((result < -tolerance)|(result > 1+tolerance)):
        raise RuntimeError('CDF range violation; approximation is not silently accepted')
    return np.clip(result,0,1),discrepancy+error,orders


def fast_event_probabilities(V,R,df=np.inf,tolerance=1e-4,batch_size=512):
    from dependence import _event_result
    V,R = np.asarray(V,float),np.asarray(R,float)
    if R.ndim==2: R=np.broadcast_to(R,(len(V),3,3))
    if not isinstance(batch_size,int) or batch_size<1: raise ValueError('Positive integer batch size required')
    all_counts,all_errors,all_orders,all_corrections = [],[],[],[]
    for start in range(0,len(V),batch_size):
        values = V[start:start+batch_size]
        four,error,orders = fast_four_cdfs(values,R[start:start+batch_size],df,tolerance=tolerance/10)
        pairs = four[:,:3].sum(axis=1);triple = four[:,3]
        c0 = triple
        c1 = pairs-3*triple
        c2 = values.sum(axis=1)-2*pairs+3*triple
        c3 = 1-values.sum(axis=1)+pairs-triple
        count = np.column_stack([c0,c1,c2,c3])
        # Allow small floating integration cancellation only; keep its size.
        correction = max(0.,float(-count.min()))
        if correction>tolerance: raise RuntimeError(f'Count probability cancellation {correction} exceeds tolerance')
        corrected = np.any((count<0)|(count>1),axis=1)
        count = np.clip(count,0,1);count /= count.sum(axis=1)[:,None]
        # Exactly-two has absolute coefficient sum nine across the four CDFs.
        # This scales a convergence proxy; it is not a rigorous error bound.
        all_counts.append(count);all_errors.append(error*9);all_orders.append(orders)
        all_corrections.append(corrected)
    result = _event_result(np.vstack(all_counts),np.concatenate(all_errors),
                           corrections=np.concatenate(all_corrections))
    result['integration_order'] = np.concatenate(all_orders)
    result['numerical_method'] = 'elliptical_correlation_path_legendre'
    return result


if __name__=='__main__':
    from dependence import elliptical_event_probabilities,IntegrationSettings
    V=np.random.default_rng(4).uniform(.5,.99,(100,3))
    R=np.array([[1.,.6,.5],[.6,1.,.7],[.5,.7,1.]])
    for df in (np.inf,5):
        st=time.perf_counter();out=fast_event_probabilities(V,R,df)
        seconds=time.perf_counter()-st
        reference=elliptical_event_probabilities(V,R,df,IntegrationSettings(tolerance=1e-5,initial_maxpts=131072,max_maxpts=1048576))
        import json
        print(json.dumps({'df':str(df),'fast_seconds':seconds,'maximum_count_discrepancy':float(np.max(np.abs(out['count_probabilities']-reference['count_probabilities']))),
                          'maximum_event_discrepancy':float(np.max(np.abs(out['ge2']-reference['ge2']))),'reported_error_proxy_max':float(out['numerical_error'].max())}),flush=True)
