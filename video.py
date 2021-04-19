from os.path import join
import numpy as np
import matplotlib.pyplot as plt

# Load helper function(s) for interacting with CTF dataset
from ctf_dataset.load import create_wrapped_dataset


# Function to resample windows to time points
def resample_windows(windows, width, n_samples=4501, collapse=np.nanmax):
    """Resample Windows 
    
    Resamples window values onto timepoints according to a given collapsing
    function (e.g nanmax, nanmean, nanmedian). Consider using np.nanmax for
    binary window values, np.nanmedian for integer labels, np.nanmean for
    continuous values. 
    
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


# Function to pull a subset of video frames based on time IDs
def get_frames(wrap_f, map_id, matchup_id, repeat_id,
               time_ids, margins=(0, 0), view='pov'):
    
    assert view in ['pov', 'top']
    assert len(time_ids) == 4501
    
    boundaries = np.where(np.diff(time_ids))[0] + 1
    boundaries = np.split(boundaries, len(boundaries) // 2)
    
    for onset, offset in boundaries:
        time_slice = slice(onset - margins[0], offset + margins[1])
    
        if view == 'pov':
            # This works for range() but not np.arange()... no idea why
            frames = np.stack([wrap_f['map/matchup/repeat/player/time/pov'][
                map_id, matchup_id, repeat_id, p, time_slice]
                               for p in range(4)],
                              axis=0)
        elif view == 'top':
            frames = wrap_f["map/matchup/repeat/time/top"][
                map_id, matchup_id, repeat_id, time_slice][np.newaxis, ...]
    
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    axes.imshow(wrap_f["map/matchup/repeat/time/top"][map_id,
                                                      matchup_id,
                                                      repeat_id,
                                                      time_id])
    axes.xaxis.set_visible(False)
    axes.yaxis.set_visible(False)


if __name__ == '__main__':
    
    base_dir = '/mnt/bucket/labs/hasson/snastase/social-ctf'
    data_dir = join(base_dir, 'data')

    # Create wrapped CTF dataset
    wrap_f = create_wrapped_dataset(data_dir,
                                    output_dataset_name="virtual.hdf5")
