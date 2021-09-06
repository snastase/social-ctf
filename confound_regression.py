from os.path import join
import numpy as np


# Simple function to compute least-squares residuals
def confound_regression(confounds, signal):
    
    # Manually add intercept
    confounds = np.column_stack([confounds, np.ones(confounds.shape[0])])
    
    # Estimate coefficients via least-squares fit
    coef, _, _, _ = np.linalg.lstsq(confounds, signal, rcond=None)
    
    # Compute residuals based on least-squares fit
    residuals = signal - np.dot(confounds, coef)
    
    return residuals


if __name__ == 'main':

    # Load helper function(s) for interacting with CTF dataset
    from ctf_dataset.load import create_wrapped_dataset


    base_dir = '/mnt/bucket/labs/hasson/snastase/social-ctf'
    data_dir = join(base_dir, 'data')


    # Create wrapped CTF dataset
    wrap_f = create_wrapped_dataset(data_dir,
                                    output_dataset_name="virtual.hdf5")

    n_lstms = 512
    n_repeats = 8
    n_players = 4
    n_pairs = n_players * (n_players - 1) // 2
    n_samples = 4501

    map_id = 0 # 0
    matchup_id = 0 # 0-54
    repeat_id = 0 # 0-7
    player_id = 0 # 0-3


    # Load pre-saved PCA's
    n_pairs = n_players * (n_players - 1) // 2
    n_matchups = 4
    k = 100
    lstms_pca = np.load(f'results/lstms_tanh-z_pca-k{k}.npy')


    # Get matchups with all same agents (e.g. AA vs AA)
    agent_ids = wrap_f['map/matchup/repeat/player/agent_id'][0, :, :, :, 0]
    matchup_ids = np.all(agent_ids[:, 0, :] == 
                         agent_ids[:, 0, 0][:, np.newaxis], axis=1)
    n_matchups = np.sum(matchup_ids) # 0, 34, 49, 54


    # Extract pre-LSTMs for relevant matchups
    prelstms = wrap_f[f'map/matchup/repeat/player/time/pre_lstm'][
        map_id, matchup_ids, ...].astype(np.float32)
    print("Loaded pre-LSTMs for within-population matchups")

    # Apply log(x + 1) to pre-LSTMs
    prelstms = np.log1p(prelstms)

    # Loop through matchups and regress confounds out of PCs
    lstms_res = np.full(lstms_pca.shape, np.nan)
    for matchup in np.arange(n_matchups):
        for repeat in np.arange(n_repeats):
            for player in np.arange(n_players):

                lstms = lstms_pca[matchup, repeat, player, ...]
                confounds = prelstms[matchup, repeat, player, ...]

                residuals = confound_regression(confounds, lstms)
                lstms_res[matchup, repeat, player] = residuals

            print("Finished confound regression for "
                  f"matchup {matchup} repeat {repeat}")

    np.save(f'results/lstms_tanh-z_pca-k{k}_reg-pre.npy', lstms_res)


    # Regress out HUD only
    hud = prelstms[..., n_lstms:]

    # Loop through matchups and regress confounds out of PCs
    lstms_res = np.full(lstms_pca.shape, np.nan)
    for matchup in np.arange(n_matchups):
        for repeat in np.arange(n_repeats):
            for player in np.arange(n_players):

                lstms = lstms_pca[matchup, repeat, player, ...]
                confounds = hud[matchup, repeat, player, ...]

                residuals = confound_regression(confounds, lstms)
                lstms_res[matchup, repeat, player] = residuals

            print("Finished confound regression for "
                  f"matchup {matchup} repeat {repeat}")

    np.save(f'results/lstms_tanh-z_pca-k{k}_reg-hud.npy', lstms_res)


    # Extract behavioral output (i.e. actions) for relevant matchups
    actions = wrap_f[f'map/matchup/repeat/player/time/action'][
        map_id, matchup_ids, ...].astype(np.float32)
    print("Loaded actions for within-population matchups")
    
    # Loop through matchups and regress confounds out of PCs
    lstms_res = np.full(lstms_pca.shape, np.nan)
    for matchup in np.arange(n_matchups):
        for repeat in np.arange(n_repeats):
            for player in np.arange(n_players):

                lstms = lstms_pca[matchup, repeat, player, ...]
                confounds = actions[matchup, repeat, player, ...]

                residuals = confound_regression(confounds, lstms)
                lstms_res[matchup, repeat, player] = residuals

            print("Finished confound regression for "
                  f"matchup {matchup} repeat {repeat}")

    np.save(f'results/lstms_tanh-z_pca-k{k}_reg-act.npy', lstms_res)


    # Combine pre-LSTMs and actions into single confound matrix
    combined = np.concatenate([prelstms, actions], axis=4)

    # Loop through matchups and regress confounds out of PCs
    lstms_res = np.full(lstms_pca.shape, np.nan)
    for matchup in np.arange(n_matchups):
        for repeat in np.arange(n_repeats):
            for player in np.arange(n_players):

                lstms = lstms_pca[matchup, repeat, player, ...]
                confounds = combined[matchup, repeat, player, ...]

                residuals = confound_regression(confounds, lstms)
                lstms_res[matchup, repeat, player] = residuals

            print("Finished confound regression for "
                  f"matchup {matchup} repeat {repeat}")

    np.save(f'results/lstms_tanh-z_pca-k{k}_reg-com.npy', lstms_res)