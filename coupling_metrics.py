import numpy as np
from itertools import combinations
from scipy.stats import zscore
from scipy.spatial.distance import squareform
from scipy.signal import hilbert
from brainiak.isc import isc
from sklearn.metrics import mutual_info_score


# Function to compute "unwrapped" correlation
def cofluctuation(a, b):
    return zscore(a) * zscore(b)


# Function to compute co-fluctuation across players
def iscf(data, vectorize_iscf=True):

    iscf_off, iscf_diag = [], []
    for pair in combinations(np.arange(data.shape[-1]), 2):
        
        # Special case of only one unit/player:
        if data.ndim == 2:
            cf = cofluctuation(data[:, pair[0]],
                               data[:, pair[1]])
            iscf_diag.append(cf)

        # Get ISCF for diagonal and off-diagonal units    
        elif data.ndim == 3:
            
            pair_diag = []
            for unit in np.arange(data.shape[1]):
                cf = cofluctuation(data[:, unit, pair[0]],
                                    data[:, unit, pair[1]])
                pair_diag.append(cf)

            iscf_diag.append(pair_diag)

            pair_off = []
            for units in combinations(np.arange(data.shape[1]), 2):
                cf = cofluctuation(data[:, units[0], pair[0]],
                                   data[:, units[1], pair[1]])
                pair_off.append(cf)

            iscf_off.append(pair_off)
            
    if data.ndim == 2:
        return np.array(iscf_diag)
        
    elif data.ndim > 2 and not vectorize_iscf:
        iscf_sq = []
        for pair_off, pair_diag in zip(iscf_off, iscf_diag):
            pair_sq = []
            for off, diag in zip(np.transpose(pair_off),
                                 np.transpose(pair_diag)):
                off_sq = squareform(off, checks=False)
                np.fill_diagonal(off_sq, diag)
                pair_sq.append(off_sq)
            iscf_sq.append(pair_sq)
        return np.array(iscf_sq)
    
    elif data.ndim > 2 and vectorize_iscf:
        return np.array(iscf_off), np.array(iscf_diag)


# Function to compute instantaneous phase synchrony
def phase_synchrony(a, b):
    a = np.angle(hilbert(a), deg=False)
    b = np.angle(hilbert(b), deg=False)
    ips = 1 - np.sin(np.abs(a - b) / 2)
    return ips


# Function to run pairwise intersubject phase synchrony (ISPS)
def isps(data, vectorize_isps=True):
    
    isps_off, isps_diag = [], []
    for pair in combinations(np.arange(data.shape[-1]), 2):
        
        # Special case of only one unit/player:
        if data.ndim == 2:
            ips = phase_synchrony(data[:, pair[0]],
                                  data[:, pair[1]])
            isps_diag.append(ips)
        
        # Get ISPS for diagonal and off-diagonal units    
        elif data.ndim == 3:
            
            pair_diag = []
            for unit in np.arange(data.shape[1]):
                ips = phase_synchrony(data[:, unit, pair[0]],
                                      data[:, unit, pair[1]])
                pair_diag.append(ips)

            isps_diag.append(pair_diag)

            pair_off = []
            for units in combinations(np.arange(data.shape[1]), 2):
                ips = phase_synchrony(data[:, units[0], pair[0]],
                                      data[:, units[1], pair[1]])
                pair_off.append(ips)

            isps_off.append(pair_off)

    if data.ndim == 2:
        return np.array(isps_diag)
        
    elif data.ndim > 2 and not vectorize_isps:
        isps_sq = []
        for pair_off, pair_diag in zip(isps_off, isps_diag):
            pair_sq = []
            for off, diag in zip(np.transpose(pair_off),
                                 np.transpose(pair_diag)):
                off_sq = squareform(off, checks=False)
                np.fill_diagonal(off_sq, diag)
                pair_sq.append(off_sq)
            isps_sq.append(pair_sq)
        return np.array(isps_sq)
    
    elif data.ndim > 2 and vectorize_isps:
        return np.array(isps_off), np.array(isps_diag)


# Function to compute sliding-window ISC
def window_isc(data, width=150, pairwise=True):

    # We expect time (i.e. samples) in the first dimension
    if data.ndim == 2:
        data = data[:, np.newaxis, :]

    n_samples = data.shape[0]
    n_units = data.shape[1]
    n_pairs = data.shape[2] * (data.shape[2] - 1) // 2
    onsets = np.arange(n_samples - width)
    n_windows = len(onsets)

    win_iscs = np.zeros((n_windows, n_pairs, n_units))
    for onset in onsets:
        win_data = data[onset:onset + width, ...]
        win_iscs[onset, ...] = isc(win_data, pairwise=True)
        if onset > 0 and onset % 500 == 0:
                print(f"Finished computing ISC for {onset} windows")

    if n_units == 1:
        win_iscs = win_iscs[..., 0]

    return win_iscs


# Function to compute sliding-window ISFC
def window_isfc(data, width=150, pairwise=True, vectorize_isfcs=False):
    
    # We expect time (i.e. samples) in the first dimension
    n_samples = data.shape[0]
    n_units = data.shape[1]
    n_pairs = data.shape[2] * (data.shape[2] - 1) // 2
    onsets = np.arange(n_samples - width)
    n_windows = len(onsets)
    
    win_isfcs = np.zeros((n_windows, n_pairs, n_units, n_units))
    for onset in onsets:
        win_data = data[onset:onset + width, ...]
        win_isfcs[onset, ...] = isfc(win_data, pairwise=pairwise,
                                     vectorize_isfcs=vectorize_isfcs)
        if onset > 0 and onset % 500 == 0:
            print(f"Finished computing ISFC for {onset} windows")
        
    return win_isfcs


# Function for computing ISCs at varying lags
def lagged_isc(data, n_lags=150, circular=True):
    
    # If lag is integer, get positive and negative range around zero
    if type(n_lags) is int:
        lags = np.arange(-n_lags, n_lags + 1)

    # Get number of pairs
    n_players = data.shape[-1]
        
    # Iterate through lags to populate lagged ISCs
    lagged_iscs = []
    for lag in lags:

        pairwise_iscs = []
        for pair in combinations(np.arange(n_players), 2):
            
            lagged_player1 = data[..., pair[0]]
            lagged_player2 = data[..., pair[1]]
            
            # If circular, loop excess elements to beginning
            if circular:
                if lag != 0:
                    lagged_player1 = np.concatenate((lagged_player1[-lag:],
                                                     lagged_player1[:-lag]))
                    
            # If not circular, trim non-overlapping elements
            # Shifts y with respect to x
            else:
                print('second')
                if lag < 0:
                    lagged_player1 = lagged_player1[:lag]
                    lagged_player2 = lagged_player2[-lag:]
                elif lag > 0:
                    lagged_player1 = lagged_player1[lag:]
                    lagged_player2 = lagged_player2[:-lag]

            pairwise_isc = isc(np.stack((lagged_player1, lagged_player2),
                                        axis=-1))
            pairwise_iscs.append(pairwise_isc)
            
        pairwise_iscs = np.array(pairwise_iscs)
        lagged_iscs.append(pairwise_iscs)
    
    lagged_iscs = np.stack(lagged_iscs, axis=-1)

    if lagged_iscs.shape[1] == 1:
        lagged_iscs = np.squeeze(lagged_iscs)
    
    return lagged_iscs, lags


# Function for computing mutual information on (binned) continuous variables
def mutual_info_binned(x, y, bins=None):
    if not bins:
        bins = np.floor(np.sqrt(4501 / 5))
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


if __name__ == 'main':
    
    # Load helper function(s) for interacting with CTF dataset
    from ctf_dataset.load import create_wrapped_dataset
    from os.path import join

    base_dir = '/mnt/bucket/labs/hasson/snastase/social-ctf'
    data_dir = join(base_dir, 'data')


    # Create wrapped CTF dataset
    wrap_f = create_wrapped_dataset(data_dir, output_dataset_name="virtual.hdf5")

    n_lstms = 512
    n_repeats = 8
    n_players = 4
    n_pairs = n_players * (n_players - 1) // 2

    map_id = 0 # 0
    matchup_id = 0 # 0-54
    repeat_id = 0 # 0-7
    player_id = 0 # 0-3


    # Get matchups with all same agents (e.g. AA vs AA)
    agent_ids = wrap_f['map/matchup/repeat/player/agent_id'][0, :, :, :, 0]
    matchup_ids = np.all(agent_ids[:, 0, :] == 
                         agent_ids[:, 0, 0][:, np.newaxis], axis=1)
    n_matchups = np.sum(matchup_ids) # 0, 34, 49, 54


    # Extract LSTMs for one map and matchup
    lstm = 'lstm'

    lstms_matched = wrap_f[f'map/matchup/repeat/player/time/{lstm}'][
        map_id, matchup_ids, ...].astype(np.float32)
    print("Loaded LSTMs for within-population matchups")

    # Apply tanh to LSTMs
    if lstm == 'lstm':
        lstms_matched = np.tanh(lstms_matched)


    # Load pre-saved PCA's
    n_matchups = 4
    n_repeats = 8
    n_players = 4
    n_pairs = n_players * (n_players - 1) // 2
    n_samples = 4501
    k = 100
    lstms_pca = np.load(f'results/lstms_tanh-z_pca-k{k}.npy')
    
    
    # Loop through matchups and repeats and compute ISCF
    iscf_results = np.zeros((n_matchups, n_repeats, n_pairs, n_samples, k, k))
    for matchup in np.arange(n_matchups):
        for repeat in np.arange(n_repeats):

            lstms = lstms_pca[matchup, repeat, ...]
            lstms = np.rollaxis(lstms, 0, 3)

            # Compute ISCs between each pair for 4 agents
            iscfs = iscf(lstms, vectorize_iscf=False)
            iscf_results[matchup, repeat, ...] = iscfs

            print(f"Computed ISCF for matchup {matchup} (repeat {repeat})")

    np.save(f'results/iscf_lstm_tanh-z_pca-k{k}.npy', iscf_results)


    # Loop through matchups and repeats
    isps_results = np.zeros((n_matchups, n_repeats, n_pairs, n_samples, k, k))
    for matchup in np.arange(n_matchups):
        for repeat in np.arange(n_repeats):

            lstms = lstms_pca[matchup, repeat, ...]
            lstms = np.rollaxis(lstms, 0, 3)

            # Compute ISCs between each pair for 4 agents
            ispss = isps(lstms, vectorize_isps=False)
            isps_results[matchup, repeat, ...] = ispss

            print(f"Computed ISPS for matchup {matchup} (repeat {repeat})")

    np.save(f'results/isps_lstm_tanh-z_pca-k{k}.npy', isps_results)

    
    # Load in confound-regression PCs and run ISCF
    reg = 'com' # 'pre', 'hud', 'act', or 'com'
    lstms_pca_reg = np.load(f'results/lstms_tanh-z_pca-k{k}_reg-{reg}.npy')
    
    # Loop through matchups and repeats and compute ISCF
    for matchup in np.arange(n_matchups):
        for repeat in np.arange(n_repeats):

            lstms = lstms_pca_reg[matchup, repeat, ...]
            lstms = np.rollaxis(lstms, 0, 3)

            # Compute ISCs between each pair for 4 agents
            iscfs = iscf(lstms, vectorize_iscf=False)

            print(f"Computed ISCF for matchup {matchup} (repeat {repeat})")
    
            np.save(f'results/iscf_lstm_tanh-z_pca-k{k}_reg-{reg}'
                    f'_m{matchup}_r{repeat}.npy', iscfs)
