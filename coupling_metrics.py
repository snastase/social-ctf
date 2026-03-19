import numpy as np
from numpy.linalg import lstsq
from itertools import combinations
from scipy.stats import pearsonr, zscore
from scipy.spatial.distance import squareform
from scipy.signal import hilbert
from brainiak.isc import isc
from sklearn.metrics import mutual_info_score, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


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


# Function to compute cross-validated ISC
def cvisc_sklearn(samples_train, samples_test, targets_train, targets_test,
                  lag=None, n_maps=32):
    
    n_targets = targets_train.shape[-1]
    
    # Multioutput regression model for many targets (PCs)
    model = MultiOutputRegressor(LinearRegression())
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
    
    if not lag or lag == 0:
        pass

    elif lag > 0:
        samples_train = samples_train[..., :-lag, :]
        targets_train = targets_train[..., lag:, :]
        samples_test = samples_test[..., :-lag, :]
        targets_test = targets_test[..., lag:, :]

    elif lag < 0:
        samples_train = samples_train[..., -lag:, :]
        targets_train = targets_train[..., :lag, :]
        samples_test = samples_test[..., -lag:, :]
        targets_test = targets_test[..., :lag, :]
        
    # Split and stack runs (will need another split/stack for repeats)
    samples_train = np.concatenate(np.split(
        samples_train, n_maps - 1, axis=0), axis=1)[0]
    targets_train = np.concatenate(np.split(
        targets_train, n_maps - 1, axis=0), axis=1)[0]

    # Fit scaler and regression pipeline and predict
    pipeline.fit(samples_train, targets_train)
    targets_pred = pipeline.predict(samples_test)

    # Compute balanced accuracy scores per target
    target_scores = []
    for t in np.arange(n_targets):
        target_scores.append(pearsonr(targets_test[:, t], targets_pred[:, t])[0])
    
    return target_scores


# Joint intersubject neural encoding/decoding
def cvisc(train_samples, test_samples,
          train_targets, test_targets,
          n_targets=142, scorer=pearsonr):
    
    scaler = StandardScaler()
    train_samples_z = scaler.fit_transform(train_samples)
    test_samples_z = scaler.transform(test_samples)
    
    train_samples_int = np.column_stack((
        train_samples, np.ones(train_samples.shape[0])))
    test_samples_int = np.column_stack((
        test_samples, np.ones(test_samples.shape[0])))
    
    W, _, _, _ = lstsq(train_samples_int, train_targets, rcond=None)

    test_pred = test_samples_int @ W
    
    def get_score(actual, predicted):
        if scorer is pearsonr:
            score = pearsonr(actual, predicted)[0]
        else:
            score = scorer(actual, predicted)
        return score

    scores = []
    for t in np.arange(n_targets):
        scores.append(
            get_score(test_targets[:, t],
                      test_pred[:, t]))
    
    return scores


# Joint intersubject neural encoding/decoding
def joint_cvisc(train_samples, test_samples,
                train_targets, test_targets,
                teammate_width=142, opponent_width=284,
                n_targets=142, scorer=pearsonr):
    
    scaler = StandardScaler()
    train_samples_z = scaler.fit_transform(train_samples)
    test_samples_z = scaler.transform(test_samples)
    
    train_samples_int = np.column_stack((
        train_samples_z, np.ones(train_samples.shape[0])))
    test_samples_int = np.column_stack((
        test_samples_z, np.ones(test_samples.shape[0])))
    
    W, _, _, _ = lstsq(train_samples_int, train_targets, rcond=None)
    
    assert W.shape[0] == np.sum((teammate_width,
                                 opponent_width)) + 1
    W_teammate = np.zeros(W.shape)
    W_teammate[:teammate_width] = W[:teammate_width]
    W_teammate[-1] = W[-1]
    W_opponent = np.zeros(W.shape)
    W_opponent[teammate_width:teammate_width + opponent_width] = \
        W[teammate_width:teammate_width + opponent_width]
    W_opponent[-1] = W[-1]

    pred_joint = test_samples_int @ W
    pred_teammate = test_samples_int @ W_teammate
    pred_opponent = test_samples_int @ W_opponent
    
    def get_score(actual, predicted):
        if scorer is pearsonr:
            score = pearsonr(actual, predicted)[0]
        else:
            score = scorer(actual, predicted)
        return score

    joint_scores = []
    teammate_scores = []
    opponent_scores = []
    for t in np.arange(n_targets):
        joint_scores.append(
            get_score(test_targets[:, t],
                      pred_joint[:, t]))
        teammate_scores.append(
            get_score(test_targets[:, t],
                      pred_teammate[:, t]))
        opponent_scores.append(
            get_score(test_targets[:, t],
                      pred_opponent[:, t]))
    
    return (joint_scores, teammate_scores, opponent_scores)


# Nested intersubject neural encoding/decoding
def nested_cvisc(train_samples, test_samples,
                 train_targets, test_targets,
                 teammate_width=142,
                 opponent_width=284,
                 n_targets=142,
                 scorer=r2_score):
    
    scaler = StandardScaler()
    train_samples_z = scaler.fit_transform(train_samples)
    test_samples_z = scaler.transform(test_samples)
    
    train_samples_int = np.column_stack((
        train_samples_z, np.ones(train_samples.shape[0])))
    test_samples_int = np.column_stack((
        test_samples_z, np.ones(test_samples.shape[0])))
    
    W, _, _, _ = lstsq(train_samples_int, train_targets, rcond=None)

    assert W.shape[0] == np.sum((teammate_width,
                                 opponent_width)) + 1
    pred_joint = test_samples_int @ W
    
    train_nested = np.column_stack((
        train_samples_z[..., teammate_width:teammate_width + opponent_width],
        np.ones(train_samples.shape[0])))
    test_nested = np.column_stack((
        test_samples_z[..., teammate_width:teammate_width + opponent_width],
        np.ones(test_samples.shape[0])))

    W, _, _, _ = lstsq(train_nested, train_targets, rcond=None)
    assert W.shape[0] == opponent_width + 1
    pred_nested = test_nested @ W
    
    def get_score(actual, predicted):
        if scorer is pearsonr:
            score = pearsonr(actual, predicted)[0]
        else:
            score = scorer(actual, predicted)
        return score

    joint_scores = []
    nested_scores = []
    for t in np.arange(n_targets):
        joint_scores.append(
            get_score(test_targets[:, t],
                      pred_joint[:, t]))
        nested_scores.append(
            get_score(test_targets[:, t],
                      pred_nested[:, t]))
    
    return (joint_scores, nested_scores)

# PC-wise behavior encoding
def behavior_encoding(train_behav, test_behav,
                      train_lstms, test_lstms,
                      player_width=30, teammate_width=30,
                      opponent_width=60, n_pcs=142,
                      scorer=pearsonr):    
    W, _, _, _ = lstsq(train_behav, train_lstms)
    
    assert W.shape[0] == np.sum((player_width,
                                 teammate_width,
                                 opponent_width))
    W_player = np.zeros(W.shape)
    W_player[:player_width] = W[:player_width]
    W_teammate = np.zeros(W.shape)
    W_teammate[player_width:player_width + teammate_width] = \
        W[player_width:player_width + teammate_width]
    W_opponent = np.zeros(W.shape)
    W_opponent[player_width + teammate_width:] = \
        W[player_width + teammate_width:]

    pred_joint = test_behav @ W
    pred_player = test_behav @ W_player
    pred_teammate = test_behav @ W_teammate
    pred_opponent = test_behav @ W_opponent
    
    def get_score(actual, predicted):
        if scorer is pearsonr:
            score = pearsonr(actual, predicted)[0]
        else:
            score = scorer(actual, predicted)
        return score

    joint_scores = []
    player_scores = []
    teammate_scores = []
    opponent_scores = []
    for t in np.arange(n_pcs):
        joint_scores.append(
            get_score(test_lstms[:, t],
                   pred_joint[:, t]))
        player_scores.append(
            get_score(test_lstms[:, t],
                   pred_player[:, t]))
        teammate_scores.append(
            get_score(test_lstms[:, t],
                   pred_teammate[:, t]))
        opponent_scores.append(
            get_score(test_lstms[:, t],
                   pred_opponent[:, t]))
    
    return (joint_scores, player_scores,
            teammate_scores, opponent_scores)


# Behavior-to-behavior prediction
def behavior_regression(train_behav, test_behav,
                        train_targets, test_targets,
                        teammate_width=30, opponent_width=60,
                        scorer=r2_score):
    
    train_behav = np.column_stack((train_behav, np.ones(train_behav.shape[0])))
    test_behav = np.column_stack((test_behav, np.ones(test_behav.shape[0])))
    
    W, _, _, _ = lstsq(train_behav, train_targets, rcond=None)
    assert W.shape[0] == teammate_width + opponent_width + 1
    pred_joint = test_behav @ W
    
    W_teammate = np.zeros(W.shape)
    W_teammate[:teammate_width] = W[:teammate_width]
    W_teammate[-1] = W[-1]
    assert np.all(np.sum(W_teammate != 0, axis=0) == teammate_width + 1)
    pred_teammate = test_behav @ W_teammate
    
    W_opponent = np.zeros(W.shape)
    W_opponent[teammate_width:] = W[teammate_width:]
    assert np.all(np.sum(W_opponent != 0, axis=0) == opponent_width + 1)
    pred_opponent = test_behav @ W_opponent
    
    train_nested = train_behav[:, teammate_width:]
    test_nested = test_behav[:, teammate_width:]
    W, _, _, _ = lstsq(train_nested, train_targets, rcond=None)
    assert W.shape[0] == opponent_width + 1
    pred_nested = test_nested @ W
    
    def get_score(actual, predicted):
        if scorer is pearsonr:
            score = pearsonr(actual, predicted)[0]
        else:
            score = scorer(actual, predicted)
        return score

    joint_scores = []
    nested_scores = []
    teammate_scores = []
    opponent_scores = []
    for t in np.arange(test_targets.shape[1]):
        joint_scores.append(
            get_score(test_targets[:, t],
                      pred_joint[:, t]))
        nested_scores.append(
            get_score(test_targets[:, t],
                      pred_nested[:, t]))
        teammate_scores.append(
            get_score(test_targets[:, t],
                      pred_teammate[:, t]))
        opponent_scores.append(
            get_score(test_targets[:, t],
                      pred_opponent[:, t]))
    
    return (joint_scores, nested_scores, teammate_scores, opponent_scores)


# Nested behavior encoding
def nested_encoding(train_behav, test_behav,
                    train_lstms, test_lstms,
                    player_width=30, teammate_width=30,
                    opponent_width=60, n_pcs=142,
                    scorer=r2_score):

    # WE NEED TO ADD INTERCEPT HERE!!!
    W, _, _, _ = lstsq(train_behav, train_lstms, rcond=None)
    assert W.shape[0] == player_width + teammate_width + opponent_width
    pred_joint = test_behav @ W
    
    train_nested = train_behav[
        ..., np.concatenate((
            np.arange(player_width), 
            np.arange(player_width + teammate_width,
                      player_width + teammate_width + opponent_width)))]
    test_nested = test_behav[
        ..., np.concatenate((
            np.arange(player_width), 
            np.arange(player_width + teammate_width,
                      player_width + teammate_width + opponent_width)))]
    W, _, _, _ = lstsq(train_nested, train_lstms, rcond=None)
    assert W.shape[0] == player_width + opponent_width
    pred_nested = test_nested @ W
    
    def get_score(actual, predicted):
        if scorer is pearsonr:
            score = pearsonr(actual, predicted)[0]
        else:
            score = scorer(actual, predicted)
        return score

    joint_scores = []
    nested_scores = []
    for t in np.arange(n_pcs):
        joint_scores.append(
            get_score(test_lstms[:, t],
                   pred_joint[:, t]))
        nested_scores.append(
            get_score(test_lstms[:, t],
                   pred_nested[:, t]))
    
    return (joint_scores, nested_scores)


# Regression-based mediation analysis
def mediation_encoding(train_X, test_X,
                       train_Y, test_Y,
                       train_M, test_M,
                       scorer=r2_score):
    
    X_width = train_X.shape[-1]
    
    train_X_int = np.column_stack((train_X, np.ones(train_X.shape[0])))
    test_X_int = np.column_stack((test_X, np.ones(test_X.shape[0])))
    
    W, _, _, _ = lstsq(train_X_int, train_Y, rcond=None)
    pred_total = test_X_int @ W
    
    W, _, _, _ = lstsq(train_X_int, train_M, rcond=None)
    pred_M = test_X_int @ W
    
    train_XM = np.column_stack((train_X, train_M, np.ones(train_X.shape[0])))
    test_XM = np.column_stack((test_X, test_M, np.ones(test_X.shape[0])))
    
    W, _, _, _ = lstsq(train_XM, train_Y, rcond=None)
    pred_medi = test_XM @ W
    
    W_direct = np.zeros(W.shape)
    W_direct[:X_width] = W[:X_width]
    W_direct[-1] = W[-1]
    pred_direct = test_XM @ W_direct
    
    def get_score(actual, predicted):
        if scorer is pearsonr:
            score = pearsonr(actual, predicted)[0]
        else:
            score = scorer(actual, predicted)
        return score

    scores_total, scores_M, scores_medi = [], [], []
    scores_direct = []
    for t in np.arange(test_Y.shape[1]):
        scores_total.append(
            get_score(pred_total[:, t], 
                      test_Y[:, t]))
        scores_M.append(
            get_score(pred_M, 
                      test_M))
        scores_medi.append(
            get_score(pred_medi[:, t], 
                      test_Y[:, t]))
        scores_direct.append(
            get_score(pred_direct[:, t], 
                      test_Y[:, t]))
    
    return scores_total, scores_M, scores_medi, scores_direct


# Autoregressive "causal" modeling analysis
def autoreg_encoding(train_X, test_X,
                     train_Y, test_Y,
                     train_future, test_future,
                     scorer=r2_score):
    
    train_auto = np.column_stack((train_Y, np.ones(train_Y.shape[0])))
    test_auto = np.column_stack((test_Y, np.ones(test_Y.shape[0])))
    
    W, _, _, _ = lstsq(train_auto, train_future, rcond=None)
    pred_auto = test_auto @ W
    
    train_causal = np.column_stack((train_X, train_Y))
    test_causal = np.column_stack((test_X, test_Y))
    
    train_causal = np.column_stack((train_causal, np.ones(train_causal.shape[0])))
    test_causal = np.column_stack((test_causal, np.ones(test_causal.shape[0])))
    
    W, _, _, _ = lstsq(train_causal, train_future, rcond=None)
    pred_causal = test_causal @ W
    
    def get_score(actual, predicted):
        if scorer is pearsonr:
            score = pearsonr(actual, predicted)[0]
        else:
            score = scorer(actual, predicted)
        return score

    scores_auto, scores_causal = [], []
    for t in np.arange(test_future.shape[1]):
        scores_auto.append(
            get_score(pred_auto[:, t], 
                      test_future[:, t]))
        scores_causal.append(
            get_score(pred_causal[:, t], 
                      test_future[:, t]))
    
    return scores_auto, scores_causal

    
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

    base_dir = '/mnt/cup/labs/hasson/snastase/social-ctf'
    data_dir = join(base_dir, 'data_v1')


    # Create wrapped CTF dataset
    wrap_f = create_wrapped_dataset(data_dir, output_dataset_name="virtual.hdf5")

    # Load pre-saved PCA's
    matchup_id = 0
    n_maps = 32
    n_repeats = 32
    n_players = 4
    n_pairs = n_players * (n_players - 1) // 2
    n_samples = 4501
    n_lstms = 512
    k = 142    
    
    confounds = 'reg-pre' # None
    
    # Loop through matchups and repeats and compute ISCF
    for map_id in np.arange(n_maps):
        for repeat_id in np.arange(n_repeats):
            if confounds:
                lstms_pca = np.load(f'results/lstms-pca_{confounds}_'
                                    f'matchup-{matchup_id}_map-{map_id}_'
                                    f'repeat-{repeat_id}.npy')
            else:
                lstms_pca = np.load(f'results/lstms-pca_'
                                    f'matchup-{matchup_id}_map-{map_id}_'
                                    f'repeat-{repeat_id}.npy')
            lstms_pca = lstms_pca[..., :k]
            lstms_pca = np.moveaxis(lstms_pca, 0, 2)
            iscfs = iscf(lstms_pca, vectorize_iscf=False)
            print(f"Computed ISCF for map {map_id} (repeat {repeat_id})")
            if confounds:
                np.save(f'results/iscfs_{confounds}_matchup-{matchup_id}_'
                        f'map-{map_id}_repeat-{repeat_id}.npy', iscfs)
            else:
                np.save(f'results/iscfs_matchup-{matchup_id}_'
                        f'map-{map_id}_repeat-{repeat_id}.npy', iscfs)

    #--- OLD ---

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
