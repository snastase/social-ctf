import numpy as np


def resample_windows(windows, width, n_samples=4501, collapse=np.nanmax):
    """Resample Windows 
    
    Resamples window values onto timepoints according to a given collapsing function 
    (e.g nanmax, nanmean, nanmedian). Consider using np.nanmax for binary window values,
    np.nanmedian for integer labels, np.nanmean for continuous values. 
    
    Parameters
    ----------
    windows: np.array 
    width: int
    n_samples: int, default: 4501
    collapse: callable, default: np.nanmax
    
    Returns
    -------
    times: np.array
    
    """
    
    # Check that there are less windows than samples 
    assert len(windows) < n_samples
    
    # Stack windows on timepoint grid 
    times = []
    for onset, window in enumerate(windows):
        time = np.full(n_samples, np.nan)
        time[onset:onset+width] = window 
        times.append(time)
    
    # Collapse window values onto timepoint grid 
    times = collapse(times, axis=0)
    
    return times 

