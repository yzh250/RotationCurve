################################################################################
# Import modules
#-------------------------------------------------------------------------------
import numpy as np

from scipy.optimize.optimize import _wrap_function, \
                                    OptimizeResult, \
                                    brent, \
                                    _line_for_search, \
                                    _minimize_scalar_bounded, \
                                    _linesearch_powell
                           
import warnings
################################################################################





################################################################################
# Standard status messages of optimizers
#-------------------------------------------------------------------------------
_status_message = {'success': 'Optimization terminated successfully.', 
                   'maxfev': 'Maximum number of function evaluations has been exceeded.', 
                   'maxiter': 'Maximum number of iteractions has been exceeded.', 
                   'pr_loss': 'Desired error not necessarily achieved due to precision loss.', 
                   'nan': 'NaN result encountered.', 
                   'out_of_bounds': 'The result is outside of the provided bounds.'}
################################################################################





################################################################################
# Custom version of scipy.optimize._linesearch_powell
#-------------------------------------------------------------------------------
def linesearch_powell(func, 
                      p, 
                      xi, 
                      tol=1e-3, 
                      lower_bound=None,
                      upper_bound=None,
                      fval=None):
    '''
    Line-search algorithm using fminbound.
    
    Find the minimum of the function `func(x0 + alpha*direc)`
    
    
    PARAMETERS
    ==========

    func : function handle
        Function to be minimized

    p : array-like
        Values of the parameters being fitted

    xi : ndarray
        `direc` - directional vector used in Powell minimization method
    
    lower_bound : ndarray
        The lower bounds for each parameter in `x0`.  If the `i`th parameter in 
        `x0` is unbounded below, then `lower_bound[i]` should be `-np.inf`.
        
        Note: `np.shape(lower_bound) == (n,)`
        
    upper_bound : ndarray
        The upper bounds for each parameter in `x0`.  If the `i`th parameter in 
        `x0` is unbounded above, then `upper_bound[i]` should be `np.inf`.
        
        Note: `np.shape(upper_bound) == (n,)`
        
    fval : int
        `fval` is equal to `func(p)`, the idea is just to avoid recomputing it 
        so that we can limit the `fevals`.
    '''
    
    def myfunc(alpha):
        return func(p + alpha*xi)
    
    
    
    if not np.any(xi):
        ########################################################################
        # If xi is zero, do not optimize
        #-----------------------------------------------------------------------
        return ((fval, p, xi) if fval is not None else (func(p), p, xi))
        ########################################################################
    
    elif lower_bound is None and upper_bound is None:
        ########################################################################
        # Non-bounded minimization
        #-----------------------------------------------------------------------
        alpha_min, fret, _, _ = brent(myfunc, full_output=1, tol=tol)
        
        xi = alpha_min*xi
        
        return np.squeeze(fret), p + xi, xi
        ########################################################################
    
    else:
        
        bound = _line_for_search(p, xi, lower_bound, upper_bound)

        print('bound:', bound)
        
        if np.isneginf(bound[0]) and np.isposinf(bound[1]):
            ####################################################################
            # Equivalent to unbounded
            #-------------------------------------------------------------------
            return linesearch_powell(func, p, xi, fval=fval, tol=tol)
            ####################################################################
        
        elif not np.isneginf(bound[0]) and not np.isposinf(bound[1]):
            ####################################################################
            # We can use a bounded scalar minimization
            #-------------------------------------------------------------------
            res = _minimize_scalar_bounded(myfunc, bound, xatol=tol/100)
            
            xi = res.x*xi
            
            return np.squeeze(res.fun), p + xi, xi
            ####################################################################
        
        else:
            ####################################################################
            # Only bounded on one side.  Use the tangent function to convert the 
            # infinity bound to a finite bound.  The new bounded region is a 
            # subregion of the region bounded by -np.pi/2 and np.pi/2
            #-------------------------------------------------------------------
            bound = np.arctan(bound[0]), np.arctan(bound[1])
            
            res = _minimize_scalar_bounded(lambda x: myfunc(np.tan(x)), 
                                           bound, 
                                           xatol=tol/100)
            
            xi = np.tan(res.x)*xi
            
            return np.squeeze(res.fun), p + xi, xi
            ####################################################################
    ############################################################################
    
    
    
################################################################################



################################################################################
# Custom version of scipy.optimize._minimize_powell
#-------------------------------------------------------------------------------
def minimize_powell(func, 
                    x0, 
                    args=(), 
                    callback=None, 
                    bounds=None, 
                    xtol=1e-4, 
                    ftol=1e-4, 
                    maxiter=None,
                    maxfev=None,
                    disp=False,
                    direc=None,
                    return_all=False,
                    **unknown_options):
    '''
    Minimization of scalar function of one or more variables using the modified 
    Powell algorithm.
    
    
    Options
    =======
    
    disp : bool
        Set to True to print convergence message
        
    xtol : float
        Relative error in solution `xopt` acceptable for convergence.
        
    ftol : float
        Relative error in `fun(opt)` acceptable for convergence.
        
    maxiter, maxfev : int
        Maximum allowed number of iterations and function evaluations.  Will 
        default to `N*1000`, where `N` is the number of variables, if neither 
        `maxiter` or `maxfev` is set.  If both `maxiter` and `maxfev` are set, 
        minimization will stop at the first reached.
        
    direc : ndarray
        Initial set of direction vectors for the Powell method
        
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the 
        iterations.
        
    bounds : Bounds
        If bounds are not provided, then an unbounded line search will be used.  
        If bounds are provided and the initial guess is within the bounds, then 
        every function evaluation throughout the minimization procedure will be 
        within the bounds.  If bounds are provided, the initial guess is outside 
        the bounds, and `direc` is full rank (or left to default), then some 
        function evaluations during the first iteration may be outside the 
        bounds, but every function evaluation after the first iteration will be 
        within the bounds.  If `direc` is not full rank, then some parameters 
        may not be optimized and the solution is not guaranteed to be within the 
        bounds.
    '''
    
    
    #_check_unknown_options(unknown_options)
    
    maxfun = maxfev
    retall = return_all
    
    # We need to use a mutable object here that we can update in the wrapper function
    fcalls, func = _wrap_function(func, args)
    
    x = np.asarray(x0).flatten()
    
    if retall:
        allvecs = [x]
    
    ############################################################################
    # Number of parameters to fit    
    #---------------------------------------------------------------------------
    N = len(x)
    ############################################################################
    
    
    ############################################################################
    # Initialize maxiter and maxfun
    #---------------------------------------------------------------------------
    # If neither are set, then set both to default
    if maxiter is None and maxfun is None:
        maxiter = N*1000
        maxfun = N*1000
    elif maxiter is None:
        # Convert remaining Nones to np.inf unless the other is np.inf, in 
        # which case use the default to avoid unbounded iteration
        if maxfun == np.inf:
            maxiter = N*1000
        else:
            maxiter = np.inf
    elif maxfun is None:
        if maxiter == np.inf:
            maxfun = N*1000
        else:
            maxfun = np.inf
    ############################################################################
    
    
    ############################################################################
    # Initialize direc
    #---------------------------------------------------------------------------
    if direc is None:
        direc = np.eye(N, dtype=float)
    else:
        direc = np.asarray(direc, dtype=float)
        
        if np.linalg.matrix_rank(direc) != direc.shape[0]:
            warnings.warn('direc input is not full rank, some parameters may ' 
                          'not be optimized', 
                          OptimizeWarning, 3)
    ############################################################################
    
    
    ############################################################################
    # Initialize bounds
    #---------------------------------------------------------------------------
    if bounds is None:
        # Don't make these arrays of all +/- inf, because _linesearch_powell 
        # will do an unnecessary check of all the elements.  Just keep them 
        # None, _linesearch_powell will not have to check all the elements.
        lower_bound, upper_bound = None, None
    else:
        # bounds is standardized in _minimize.py
        lower_bound, upper_bound = bounds.lb, bounds.ub
        
        if np.any(lower_bound > x0) or np.any(x0 > upper_bound):
            warnings.warn('Initial guess is not within the specified bounds', 
                          OptimizeWarning, 3)
    ############################################################################
    
    
    ############################################################################
    #---------------------------------------------------------------------------
    fval = np.squeeze(func(x))
    
    x1 = x.copy()
    
    iter = 0
    
    ilist = list(range(N))
    
    while True:
        
        print('direc:', direc)
        
        fx = fval
        
        bigind = 0
        
        delta = 0.0
        
        ########################################################################
        #-----------------------------------------------------------------------
        for i in ilist:
            
            print('Fitting for variable', i)
            
            direc1 = direc[i]
            
            fx2 = fval
            
            fval, x, direc1 = _linesearch_powell(func, 
                                                x, 
                                                direc1, 
                                                tol=xtol*100, 
                                                lower_bound=lower_bound,
                                                upper_bound=upper_bound, 
                                                fval=fval)
            
            if (fx2 - fval) > delta:
                
                delta = fx2 - fval
                
                bigind = i
        
        print('Finished fitting for variables')
        ########################################################################
        
        
        ########################################################################
        # Increase number of iterations
        #-----------------------------------------------------------------------
        iter += 1
        ########################################################################
        
        
        ########################################################################
        #-----------------------------------------------------------------------
        if callback is not None:
            callback(x)
        ########################################################################
        
        
        ########################################################################
        # Save current solution
        #-----------------------------------------------------------------------
        if retall:
            
            allvecs.append(x)
            
            print('Current parameter values:', x)
        ########################################################################
        
        
        ########################################################################
        # Stop iterating through solutions if difference between this solution 
        # and the last is small enough
        #-----------------------------------------------------------------------
        bnd = ftol*(np.abs(fx) + np.abs(fval)) + 1e-20
        
        if 2.0*(fx - fval) <= bnd:
            break
        ########################################################################
        
        
        ########################################################################
        # Stop iterating through solutions if we have evaluated the function too 
        # many times.
        #-----------------------------------------------------------------------
        if fcalls[0] >= maxfun:
            break
        ########################################################################
        
        
        ########################################################################
        # Stop iterating through solutions if we have tried too many times
        #-----------------------------------------------------------------------
        if iter >= maxiter:
            break
        ########################################################################
        
        
        ########################################################################
        # Stop iterating through solutions if we are in a nan-region of the 
        # function
        #-----------------------------------------------------------------------
        if np.isnan(fx) and np.isnan(fval):
            break
        ########################################################################
        
        
        ########################################################################
        # Construct the extrapolated point
        #-----------------------------------------------------------------------
        direc1 = x - x1
        
        #print('direc1', direc1)
        
        x2 = 2*x - x1

        #print(x2)
        '''
        #-----------------------------------------------------------------------
        # Check to see that x2 is within the bounds.  If not, adjust to the 
        # boundary values.
        #-----------------------------------------------------------------------
        for i in range(len(x2)):

            # Lower bound
            if x2[i] < lower_bound[i]:
                x2[i] = lower_bound[i]
                print('Adjusted x2[', i, '] to lower bound.')

            # Upper bound
            elif x2[i] > upper_bound[i]:
                x2[i] = upper_bound[i]
                print('Adjusted x2[', i, '] to upper bound.')
        #-----------------------------------------------------------------------
        '''
        x1 = x.copy() # Update x1 to current value of x
        
        fx2 = np.squeeze(func(x2))

        print('Constructed the extrapolated point.')
        ########################################################################
        
        
        ########################################################################
        # Update direc if this solution is better than the last solution
        #-----------------------------------------------------------------------
        if fx > fx2:
            
            t = 2.0*(fx + fx2 - 2.0*fval)
            
            temp = fx - fval - delta
            
            t *= temp*temp
            
            temp = fx - fx2
            
            t -= delta*temp*temp
            
            if t < 0.0:
                fval, x, direc1 = _linesearch_powell(func, 
                                                    x, 
                                                    direc1, 
                                                    tol=xtol*100, 
                                                    lower_bound=lower_bound,
                                                    upper_bound=upper_bound,
                                                    fval=fval)
                
                if np.any(direc1):
                    
                    direc[bigind] = direc[-1]
                    
                    direc[-1] = direc1
        ########################################################################
    ############################################################################
    
    
    ############################################################################
    # Determine why the solver terminated
    # 
    # Out of bounds is more urgent than exceeding function evals or iters, but I 
    # don't want to cause inconsistencies by changing the established warning 
    # flags for maxfev and maxiter, so the out of bounds warning flag becomes 3, 
    # but is checked for first.
    #---------------------------------------------------------------------------
    warnflag = 0
    
    if bounds and (np.any(lower_bound > x) or np.any(x > upper_bound)):
        warnflag = 4
        msg = _status_message['out_of_bounds']
        
        if disp:
            print('Warning:', msg)
        
    elif fcalls[0] >= maxfun:
        warnflag = 1
        msg = _status_message['maxfev']
        
        if disp:
            print('Warning:', msg)
    
    elif iter >= maxiter:
        warnflag = 2
        msg = _status_message['maxiter']
        
        if disp:
            print('Warning:', msg)
    
    elif np.isnan(fval) or np.isnan(x).any():
        warnflag = 3
        msg = _status_message['nan']
        
        if disp:
            print('Warning:', msg)
    
    else:
        msg = _status_message['success']
        
        if disp:
            print(msg)
            print('        Current function value: %f' % fval)
            print('        Iterations: %d' % iter)
            print('        Function evaluations: %d' % fcalls[0])
    ############################################################################
    
    
    ############################################################################
    # Create result dictionary
    #---------------------------------------------------------------------------
    result = OptimizeResult(fun=fval, 
                            direc=direc, 
                            nit=iter, 
                            nfev=fcalls[0], 
                            status=warnflag, 
                            success=(warnflag == 0), 
                            message=msg, 
                            x=x)
    ############################################################################
    
    return result
################################################################################
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    