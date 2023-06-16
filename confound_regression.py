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


    base_dir = '/jukebox/hasson/snastase/social-ctf'
    data_dir = join(base_dir, 'data_v1')


    # Create wrapped CTF dataset
    wrap_f = create_wrapped_dataset(data_dir,
                                    output_dataset_name="virtual.hdf5")

    n_lstms = 512
    n_repeats = 32
    n_maps = 32
    n_players = 4
    n_pairs = n_players * (n_players - 1) // 2
    n_samples = 4501

    matchup_id = 0 # 0-54 or 0-3


    # Loop through maps and regress pre-LSTM confounds out of PCs
    for map_id in np.arange(n_maps):
        for repeat_id in np.arange(n_repeats):
            
            # Load pre-saved PCA-reduced data            
            lstms_pca = np.load(f'results/lstms-pca_matchup-{matchup_id}'
                                f'_map-{map_id}_repeat-{repeat_id}.npy')

            # Extract pre-LSTMs for relevant matchups
            prelstms = wrap_f[f'map/matchup/repeat/player/time/pre_lstm'][
                map_id, matchup_id, repeat_id, ...].astype(np.float32)
            print(f"Loaded pre-LSTMs for map {map_id} repeat {repeat_id}")

            # Apply log(x + 1) to pre-LSTMs
            prelstms = np.log1p(prelstms)
            
            # Loop through players and perform confound regression
            lstms_res = np.full(lstms_pca.shape, np.nan)
            for player_id in np.arange(n_players):

                lstms = lstms_pca[player_id, ...]
                confounds = prelstms[player_id, ...]

                residuals = confound_regression(confounds, lstms)
                lstms_res[player_id] = residuals

            print("Finished confound regression for "
                  f"map {map_id} repeat {repeat_id}")

            np.save(f'results/lstms-pca_reg-pre_matchup-{matchup_id}'
                    f'_map-{map_id}_repeat-{repeat_id}.npy', lstms_res)


    # Loop through maps and regress HUD-only confounds out of PCs
    for map_id in np.arange(n_maps):
        for repeat_id in np.arange(n_repeats):

            # Load pre-saved PCA-reduced data            
            lstms_pca = np.load(f'results/lstms-pca_matchup-{matchup_id}'
                                f'_map-{map_id}_repeat-{repeat_id}.npy')

            # Regress out HUD only
            prelstms = wrap_f[f'map/matchup/repeat/player/time/pre_lstm'][
                map_id, matchup_id, repeat_id, ...].astype(np.float32)
            hud = prelstms[..., n_lstms:]
            print(f"Loaded HUD for map {map_id} repeat {repeat_id}")

            # Loop through players and perform confound regression
            lstms_res = np.full(lstms_pca.shape, np.nan)
            for player_id in np.arange(n_players):

                lstms = lstms_pca[player_id, ...]
                confounds = hud[player_id, ...]

                residuals = confound_regression(confounds, lstms)
                lstms_res[player_id] = residuals

            print("Finished confound regression for "
                  f"map {map_id} repeat {repeat_id}")

            np.save(f'results/lstms-pca_reg-hud_matchup-{matchup_id}'
                    f'_map-{map_id}_repeat-{repeat_id}.npy', lstms_res)


    # Loop through maps and regress action confounds out of PCs
    for map_id in np.arange(n_maps):
        for repeat_id in np.arange(n_repeats):

            # Load pre-saved PCA-reduced data            
            lstms_pca = np.load(f'results/lstms-pca_matchup-{matchup_id}'
                                f'_map-{map_id}_repeat-{repeat_id}.npy')

            # Extract behavioral output (i.e. actions) for relevant matchups
            actions = wrap_f[f'map/matchup/repeat/player/time/action'][
                map_id, matchup_id, repeat_id, ...].astype(np.float32)
            print(f"Loaded actions for map {map_id} repeat {repeat_id}")
    
            # Loop through players and perform confound regression
            lstms_res = np.full(lstms_pca.shape, np.nan)
            for player_id in np.arange(n_players):

                lstms = lstms_pca[player_id, ...]
                confounds = actions[player_id, ...]

                residuals = confound_regression(confounds, lstms)
                lstms_res[player_id] = residuals

            print("Finished confound regression for "
                  f"map {map_id} repeat {repeat_id}")
            
            np.save(f'results/lstms-pca_reg-act_matchup-{matchup_id}'
                    f'_map-{map_id}_repeat-{repeat_id}.npy', lstms_res)


    # Loop through maps and regress combined input/output confounds out of PCs
    for map_id in np.arange(n_maps):
        for repeat_id in np.arange(n_repeats):

            # Load pre-saved PCA-reduced data            
            lstms_pca = np.load(f'results/lstms-pca_matchup-{matchup_id}'
                                f'_map-{map_id}_repeat-{repeat_id}.npy')
            
            # Extract pre-LSTMs for relevant matchups
            prelstms = wrap_f[f'map/matchup/repeat/player/time/pre_lstm'][
                map_id, matchup_id, repeat_id, ...].astype(np.float32)
            
            # Apply log(x + 1) to pre-LSTMs
            prelstms = np.log1p(prelstms)
            
            # Combine pre-LSTMs and actions into single confound matrix
            actions = wrap_f[f'map/matchup/repeat/player/time/action'][
                map_id, matchup_id, repeat_id, ...].astype(np.float32)

            combined = np.concatenate((prelstms, actions), axis=-1)
            print(f"Loaded pre-LSTMs and actions "
                  f"for map {map_id} repeat {repeat_id}")
            
            # Loop through players and perform confound regression
            lstms_res = np.full(lstms_pca.shape, np.nan)
            for player_id in np.arange(n_players):

                lstms = lstms_pca[player_id, ...]
                confounds = combined[player_id, ...]

                residuals = confound_regression(confounds, lstms)
                lstms_res[player_id] = residuals

            print("Finished confound regression for "
                  f"map {map_id} repeat {repeat_id}")
            
            np.save(f'results/lstms-pca_reg-com_matchup-{matchup_id}'
                    f'_map-{map_id}_repeat-{repeat_id}.npy', lstms_res)


    # Loop through maps and regress reward confounds out of PCs
    for map_id in np.arange(n_maps):
        for repeat_id in np.arange(n_repeats):
            
            # Load pre-saved PCA-reduced data            
            lstms_pca = np.load(f'results/lstms-pca_matchup-{matchup_id}'
                                f'_map-{map_id}_repeat-{repeat_id}.npy')

            # Extract rewards for relevant matchups
            rewards = wrap_f[f'map/matchup/repeat/player/time/reward'][
                map_id, matchup_id, repeat_id, ...].astype(np.float32)
            print(f"Loaded reward for map {map_id} repeat {repeat_id}")
            
            # Loop through players and perform confound regression
            lstms_res = np.full(lstms_pca.shape, np.nan)
            for player_id in np.arange(n_players):

                lstms = lstms_pca[player_id, ...]
                confounds = rewards[player_id, ...]

                residuals = confound_regression(confounds, lstms)
                lstms_res[player_id] = residuals

            print("Finished confound regression for "
                  f"map {map_id} repeat {repeat_id}")

            np.save(f'results/lstms-pca_reg-rew_matchup-{matchup_id}'
                    f'_map-{map_id}_repeat-{repeat_id}.npy', lstms_res)


    # Loop through maps and regress all input/output/reward confounds out of PCs
    for map_id in np.arange(n_maps):
        for repeat_id in np.arange(n_repeats):

            # Load pre-saved PCA-reduced data            
            lstms_pca = np.load(f'results/lstms-pca_matchup-{matchup_id}'
                                f'_map-{map_id}_repeat-{repeat_id}.npy')
            
            # Extract pre-LSTMs for relevant matchups
            prelstms = wrap_f[f'map/matchup/repeat/player/time/pre_lstm'][
                map_id, matchup_id, repeat_id, ...].astype(np.float32)
            
            # Apply log(x + 1) to pre-LSTMs
            prelstms = np.log1p(prelstms)
            
            # Extract behavioral output (i.e. actions) for relevant matchups
            actions = wrap_f[f'map/matchup/repeat/player/time/action'][
                map_id, matchup_id, repeat_id, ...].astype(np.float32)
            
            # Extract rewards for relevant matchups
            rewards = wrap_f[f'map/matchup/repeat/player/time/reward'][
                map_id, matchup_id, repeat_id, ...].astype(np.float32)
            print(f"Loaded reward for map {map_id} repeat {repeat_id}")

            # Combine pre-LSTMs, actions, rewards into single confound matrix
            combined = np.concatenate((prelstms, actions, rewards), axis=-1)
            print(f"Loaded pre-LSTMs and actions "
                  f"for map {map_id} repeat {repeat_id}")
            
            # Loop through players and perform confound regression
            lstms_res = np.full(lstms_pca.shape, np.nan)
            for player_id in np.arange(n_players):

                lstms = lstms_pca[player_id, ...]
                confounds = combined[player_id, ...]

                residuals = confound_regression(confounds, lstms)
                lstms_res[player_id] = residuals

            print("Finished confound regression for "
                  f"map {map_id} repeat {repeat_id}")
            
            np.save(f'results/lstms-pca_reg-all_matchup-{matchup_id}'
                    f'_map-{map_id}_repeat-{repeat_id}.npy', lstms_res)