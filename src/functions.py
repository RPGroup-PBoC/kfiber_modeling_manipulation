import numpy as np
import numba

import emcee
import pandas as pd



@numba.njit
def contour_fn(x, y):
    """
    Given the (x,y) coordinates of points on a curve,
    calculates the corresponding arc lengths, as well
    as the contour length of the whole curve.
    
    Parameters
    ----------
    x : numpy array
        x-coordinates.

    y : numpy array
        y-coordinates.
    
    
    Returns
    -------
    s : numpy array
        arc lengths corresponding to the (x,y) data.
    
    L_contour : float
        Contour length of the curve specified by the (x,y) data.
    """
    
    # Arc length array
    s = np.zeros(len(x))
    
    for i in range(1,len(s)):
        dx = x[i] - x[i-1]
        dy = y[i] - y[i-1]
        
        ds = np.sqrt(dx**2 + dy**2)
        s[i] = s[i-1] + ds
        
    # Contour length
    L_contour = s[-1]
    
    return s, L_contour


@numba.njit
def mesh_jit(x_arr, y_arr):
    """
    An `njit`-compatible function that creates meshgrids using two 1-d arrays
    """
    nrow = len(y_arr)
    ncol = len(x_arr)
    mesh1 = np.zeros((nrow, ncol))
    mesh1[:,:] = x_arr

    mesh2 = np.zeros((ncol, nrow))
    mesh2[:,:] = y_arr
    mesh2 = np.transpose(mesh2)
    
    return mesh1, mesh2


@numba.njit
def min_over_columns(arr):
    """
    An `njit`-compatible function that finds the lowest values in 
    each row of a 2-d array (minimum over columns).
    """
    n, _ = arr.shape
    min_vals = np.empty(n)
    for i in range(n):
        min_vals[i] = np.min(arr[i,:])
    return min_vals


@numba.njit
def distances_pair_min_new(x_points, y_points, x_curve, y_curve):
    """
    For each point specified by {`x_points`, `y_points`}, finds the minimal 
    distance to the piece-wise linear curve specified by {`x_curve`, `y_curve`}.
    Perpendicular projections are used in finding the distances.
    
    Parameters
    ----------
    x_points : array_like
        x-coordinates of points for which we calculate the minimal distances.
        
    y_points : array_like
        y-coordinates of points for which we calculate the minimal distances.
        
    x_curve : array_like
        x-coordinates of the reference set of points.
        
    y_curve : array_like
        y-coordinates of the reference set of points.
    
    Returns
    -------
    d_min_overall : numpy array
        Array of the minimal distances that the pairs {x_points, y_points}
        have from the reference pairs {x_curve, y_curve}.
    """
    
    x_curve_mat, x_points_mat = mesh_jit(x_curve, x_points)
    y_curve_mat, y_points_mat = mesh_jit(y_curve, y_points)

    d_pairwise_mat = np.sqrt((x_curve_mat-x_points_mat)**2 + (y_curve_mat-y_points_mat)**2)
    
    d1_mat = d_pairwise_mat[:,0:-1]
    d2_mat = d_pairwise_mat[:,1:]
    ell = np.sqrt((y_curve[1]-y_curve[0])**2 + (x_curve[1]-x_curve[0])**2)
    
    cos1_mat = (d2_mat**2+ell**2-d1_mat**2)/(2*d2_mat*ell)
    cos2_mat = (d1_mat**2+ell**2-d2_mat**2)/(2*d1_mat*ell)
    
    sign = np.sign(cos1_mat*cos2_mat)
    is_mid = (1+sign)/2
    is_end = 1-is_mid
    
    d_perp = d2_mat*np.sqrt(np.abs(1-cos1_mat**2))
    
    comp = d1_mat < d2_mat
    d_lower = d1_mat*comp + d2_mat*(1-comp)
    
    d_min_segments = is_mid*d_perp + is_end*d_lower
    
    d_min_overall = min_over_columns(d_min_segments)
    d_min_overall[np.isnan(d_min_overall)]=0
    
    return d_min_overall


def sampler_to_dataframe(sampler, columns=None):
    """
    Convert output of an emcee sampler to a Pandas DataFrame.
    Source: BeBi103 course materials (Justin Bois, Caltech)

    Parameters
    ----------
    sampler : emcee.EnsembleSampler or emcee.PTSampler instance
        Sampler instance form which MCMC has already been run.

    Returns
    -------
    output : DataFrame
        Pandas DataFrame containing the samples. Each column is
        a variable, except: 'lnprob' and 'chain' for an
        EnsembleSampler, and 'lnlike', 'lnprob', 'beta_ind',
        'beta', and 'chain' for a PTSampler. These contain obvious
        values.
    """
    invalid_column_names = ['lnprob', 'chain', 'lnlike', 'beta',
                            'beta_ind']
    if np.any([x in columns for x in invalid_column_names]):
            raise RuntimeError('You cannot name columns with any of these: '
                                    + '  '.join(invalid_column_names))

    if columns is None:
        columns = list(range(sampler.chain.shape[-1]))

    if isinstance(sampler, emcee.EnsembleSampler):
        n_walkers, n_steps, n_dim = sampler.chain.shape

        df = pd.DataFrame(data=sampler.flatchain, columns=columns)
        df['lnprob'] = sampler.flatlnprobability
        df['chain'] = np.concatenate([i * np.ones(n_steps, dtype=int)
                                                for i in range(n_walkers)])
    elif isinstance(sampler, emcee.PTSampler):
        n_temps, n_walkers, n_steps, n_dim = sampler.chain.shape

        df = pd.DataFrame(
            data=sampler.flatchain.reshape(
                (n_temps * n_walkers * n_steps, n_dim)),
            columns=columns)
        df['lnlike'] = sampler.lnlikelihood.flatten()
        df['lnprob'] = sampler.lnprobability.flatten()

        beta_inds = [i * np.ones(n_steps * n_walkers, dtype=int)
                     for i, _ in enumerate(sampler.betas)]
        df['beta_ind'] = np.concatenate(beta_inds)

        df['beta'] = sampler.betas[df['beta_ind']]

        chain_inds = [j * np.ones(n_steps, dtype=int)
                      for i, _ in enumerate(sampler.betas)
                      for j in range(n_walkers)]
        df['chain'] = np.concatenate(chain_inds)
    else:
        raise RuntimeError('Invalid sample input.')

    return df


@numba.njit
def distances_pair_min(x_points, y_points, x_curve, y_curve):
    """
    Finds the lowest pairwise distances between two sets of points.
    
    Parameters
    ----------
    x_points : array_like
        x-coordinates of points for which we calculate the minimal distances.
        
    y_points : array_like
        y-coordinates of points for which we calculate the minimal distances.
        
    x_curve : array_like
        x-coordinates of the reference set of points.
        
    y_curve : array_like
        y-coordinates of the reference set of points.
    
    Returns
    -------
    d_pairwise_min : numpy array
        Array of the minimal distances that the pairs {x_points, y_points}
        have from the reference pairs {x_curve, y_curve}.
        
    ind_pairs : numpy array
        Indices of the points from the {x_curve, y_curve} set that yield
        the minimal distances.
    """
    
    x_curve_mat, x_points_mat = mesh_jit(x_curve, x_points)
    y_curve_mat, y_points_mat = mesh_jit(y_curve, y_points)
    
    d_pairwise_mat = np.sqrt((x_curve_mat-x_points_mat)**2 + (y_curve_mat-y_points_mat)**2)
    
    d_pairwise_min, ind_pairs = min_axis1_jit(d_pairwise_mat)
    
    return d_pairwise_min, ind_pairs


@numba.njit
def interpolate_two_pt(x_out, x1, x2, y1, y2):
    """
    Perform linear interpolation at `x_out` given two endpoint coordinates.
    """
    m = (y2-y1)/(x2-x1)
    c = y1 - m*x1
    y_out = m*x_out + c
    return y_out













@numba.njit
def rotate(x, y, angle):
    """
    Gives the coordinates in a Cartesian reference frame that 
    is rotated around the origin (0,0) in the clockwise direction 
    by the specified amount.
    
    Parameters
    ----------
    x : float or numpy array
        x-coordinates in the original reference frame.

    y : float or numpy array
        y-coordinates in the original reference frame.
        
    angle : float
        The rotation angle in radians.
        
        
    Returns
    -------
    x_rot : float or numpy array
        x-coordinates in the rotated reference frame.
    
    y_rot : float or numpy array
        y-coordinates in the rotated reference frame.
    """
    
    x_rot =  x*np.cos(angle) + y*np.sin(angle)
    y_rot = -x*np.sin(angle) + y*np.cos(angle)
    
    return x_rot, y_rot