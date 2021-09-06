import numpy as np
from brainiak.utils.utils import p_from_null
from statsmodels.stats.multitest import multipletests
from scipy.spatial.distance import squareform


# Helper function for Fisher-transformed average
def fisher_mean(correlations, axis=None):
    return np.tanh(np.nanmean(np.arctanh(correlations), axis=axis))


# Nonparametric bootstrap hypothesis test
def bootstrap_test(data, bootstrap_axis=0, n_bootstraps=1000,
                   estimator=fisher_mean, ci_percentile=95,
                   side='two-sided'):

    n_samples = data.shape[bootstrap_axis]
    observed = estimator(data, axis=bootstrap_axis)

    # Loop through bootstrap samples to populate distribution
    distribution = []
    for i in np.arange(n_bootstraps):
        
        # Random sample with replacement
        bootstrap_ids = np.random.choice(np.arange(n_samples),
                                         size=n_samples)
        bootstrap_data = np.take(data, bootstrap_ids,
                                 axis=bootstrap_axis)
        
        # Apply estimator and append to distribution
        distribution.append(estimator(bootstrap_data,
                                      axis=bootstrap_axis))
        
    distribution = np.stack(distribution, axis=bootstrap_axis)
        
    
    # Compute CIs from bootstrap distribution
    ci = (np.percentile(distribution,
                        (100 - ci_percentile)/2, axis=0),
          np.percentile(distribution,
                        ci_percentile + (100 - ci_percentile)/2,
                        axis=0))
    
    # Shift bootstrap distribution to 0 for hypothesis test
    shifted = distribution - observed
    
    # Get p-value for actual median from shifted distribution
    p = p_from_null(observed, shifted,
                    side=side, exact=False,
                    axis=bootstrap_axis)

    return observed, ci, p, distribution
    

# Compute FDR on a redundant square matrix of p-values
def squareform_fdr(square_p, include_diagonal=True, alpha=.05,
                   method='fdr_bh'):
    triangle_p = squareform(square_p, checks=False)
    
    if include_diagonal:
        diagonal_p = np.diagonal(square_p)
        n_diagonal = len(diagonal_p)
        p = np.concatenate([diagonal_p, triangle_p])
        
    else:
        p = triangle_p
        
    _, fdr_p, _, _ = multipletests(p, alpha=alpha, method=method)
    
    if include_diagonal:
        fdr_square = squareform(fdr_p[n_diagonal:])
        np.fill_diagonal(fdr_square, fdr_p[:n_diagonal])
        
    else:
        fdr_square = squareform(fdr_p[n_diagonal:])
        np.fill_diagonal(fdr_square, np.nan)
    
    return fdr_square


# Compute mean ISCF (i.e. ISC) based on an indicator variable
def block_iscf(iscfs, indicator, min_block=0, precede=None):
    
    # Get blocks from indicator
    block_ids = indicator.nonzero()[0]
    blocks = np.split(block_ids, np.where(np.diff(block_ids) != 1)[0] + 1)
    
    # Get time points preceding blocks if requested
    if precede:
        blocks = [block[0] - precede for block in blocks]
        
        # Ensure we don't end up with negative time points
        blocks = [block[block >= 0] for block in blocks]
        
    # Compute the mean ISCR within each block
    block_iscfs = []
    for block in blocks:
        if len(block) >= min_block:
            block_iscfs.append(np.mean(iscfs[block]))
    block_iscfs = fisher_mean(block_iscfs)
    
    return block_iscfs


# Nonparametric test for shuffling blocks in a time series
def block_randomization(iscfs, indicator, pad=0, min_block=0,
                        n_randomizations=1000, side='two-sided'):
    
    assert len(iscfs) == len(indicator)
    n_samples = len(indicator)
    
    # Find blocks and nonblock time points
    block_ids = indicator.nonzero()[0]
    nonblock_ids = (~indicator).nonzero()[0]
    
    # Compute block edges accounting for first and last time point
    block_edges = np.where(np.diff(np.r_[False, indicator, False])
                           != 0)[0]
    assert len(block_edges) % 2 == 0
    
    block_onsets, block_offsets = block_edges[::2], block_edges[1::2]
    blocks = np.split(block_ids, np.where(np.diff(block_ids) != 1)[0] + 1)
    n_blocks = len(blocks)
    block_durs = [len(block) for block in blocks]
    
    # Compute mean ISCF within blocks
    observed = []
    for block in blocks:
        if len(block) >= min_block:
            observed.append(np.mean(iscfs[block]))
    observed = fisher_mean(observed)
    
    # Pad around blocks if requested
    # THIS DOESN'T WORK YET BECAUSE BLOCKS AT BEGINNING/END 
    # WILL WRECK NONBLOCK_IDS
    if pad:
        len_block_ids = len(block_ids)
        len_nonblock_ids = len(nonblock_ids)
        for onset, offset in zip(block_onsets, block_offsets):
            block_ids = np.insert(block_ids,
                                  np.where(block_ids == onset)[0],
                                  -(np.arange(pad)[::-1] + 1) + onset)
            block_ids = np.insert(block_ids,
                                  np.where(block_ids == offset - 1)[0] + 1,
                                  np.arange(pad) + offset)
            nonblock_ids = np.delete(nonblock_ids,
                                     -(np.arange(pad) - 
                                       np.where(nonblock_ids ==
                                                onset - 1)[0]))
            nonblock_ids = np.delete(nonblock_ids,
                                     np.arange(pad) + 
                                     np.where(nonblock_ids == offset)[0])
        assert len(block_ids) == len_block_ids + n_blocks * pad * 2
        assert len(nonblock_ids) == len_nonblock_ids - n_blocks * pad * 2

    # Loop through randomizations to create distribution
    distribution = []
    for i in np.arange(n_randomizations):
        random_rs = []
        for block_dur in block_durs:
            random_onsets = np.copy(nonblock_ids)
            block_gaps = np.where(np.diff(nonblock_ids) != 1)[0]
            prev_gap = 0
            overlap_ids = []
            for gap in block_gaps:
                overlap_id = gap - block_dur + 1
                if overlap_id <= prev_gap:
                    overlap_id = prev_gap + 1
                overlap_ids.extend(np.arange(overlap_id, gap + 1))
                prev_gap = gap
            random_onsets = np.delete(random_onsets, overlap_ids)

            # Exclude final time points so we don't run out of samples
            trim_end = nonblock_ids[nonblock_ids >= (n_samples - block_dur)]
            if len(trim_end) > 0:
                for trim_id in trim_end:
                    random_onsets = random_onsets[random_onsets != trim_id]

            random_onset = np.random.choice(random_onsets, size=1)
            random_block = np.arange(random_onset, random_onset + block_dur)

            assert len(np.intersect1d(block_ids, random_block)) == 0
            random_rs.append(np.mean(iscfs[random_block]))
        
        distribution.append(fisher_mean(random_rs))
                                 
    distribution = np.array(distribution)
    
    # Get p-value for observed from null distribution
    p = p_from_null(observed, distribution,
                    side=side, exact=False,
                    axis=0)
    
    # Get z-value based on null distribution
    z = (observed - fisher_mean(distribution)) / np.std(distribution)

    return observed, z, p, distribution