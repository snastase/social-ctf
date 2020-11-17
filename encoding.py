from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from ctf_dataset.load import create_wrapped_dataset
from features import get_features
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


base_dir = '/mnt/bucket/labs/hasson/snastase/social-ctf'
data_dir = join(base_dir, 'data')


# Create wrapped CTF dataset
wrap_f = create_wrapped_dataset(data_dir, output_dataset_name="virtual.hdf5")

map_id = 0 # 0
matchup_id = 0 # 0-54
repeat_id = 0 # 0-7
player_id = 0 # 0-3

n_lstms = 512
n_repeats = 8
n_players = 4

# Extract LSTMs for one map and matchup (y variable)
lstms = wrap_f['map/matchup/repeat/player/time/post_lstm'][
    map_id, matchup_id, repeat_id, player_id].astype(np.float32)


# Load in feature matrix for this game (x variable)
features, feature_labels = get_features(wrap_f, map_id=map_id,
                                        matchup_id=matchup_id,
                                        repeat_id=repeat_id,
                                        player_id=player_id)


# Fit linear regression model to full dataset
lstm_model = LinearRegression().fit(features, lstms)
lstm_r2 = r2_score(lstms, lstm_model.predict(features),
                   multioutput='raw_values')
lstm_resid = lstms - lstm_model.predict(features)


# Cross-validated ridge regression
from sklearn.linear_model import Ridge

# Custom cross-validation function for scoring multiple output variables
def cross_validate(estimator, X, y, score=r2_score, cv=KFold):

    # Split into cross-validation folds
    scores, models = [], []
    for train, test in cv.split(X):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        # Fit the regression model
        model = estimator.fit(X_train, y_train)
        models.append(model)

        # Predict output(s) for test set
        y_pred = model.predict(X_test)

        # Evaluate prediction performance for all output variables
        scores.append(score(y_test, y_pred, multioutput='raw_values'))
        
    return scores, models

# 5-fold cross-validation across time points within a game
n_splits = 10
cv = KFold(n_splits)

ridge = Ridge()

# Get cross-validated R-squared for ridge encoding model
scores, models = cross_validate(ridge, features, lstms, score=r2_score, cv=cv)


# Cross-validate ridge regression across repeats
n_repeats = 8

lstms = wrap_f['map/matchup/repeat/player/time/post_lstm'][
    map_id, matchup_id, slice(n_repeats), player_id].astype(np.float32)

features, feature_labels = get_features(wrap_f, map_id=map_id,
                                        matchup_id=matchup_id,
                                        repeat_id=slice(n_repeats),
                                        player_id=player_id)

# Stack repeats along the time dimension
lstm_stack = np.vstack([r for r in lstms])
feature_stack = np.vstack([r for r in features])

# We could use LeaveOneGroupOut here, but KFold seems to work fine
cv = KFold(n_repeats)

ridge = Ridge()

scores, models = cross_validate(ridge, feature_stack, lstm_stack,
                                score=r2_score, cv=cv)


# Convenience function for plotting grid of LSTM values
def plot_lstm_grid(lstms, n_rows=16, n_cols=32, title=None, **kwargs):

    lstm_grid = lstms.reshape(n_rows, n_cols)
    ratio = lstm_grid.shape[0] / lstm_grid.shape[1]

    fig, ax = plt.subplots(figsize=(8, 6))

    m = ax.matshow(lstm_grid, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('LSTM units', loc='left')
    ax.set_title(title)
    fig.colorbar(m, ax=ax, fraction=0.047 * ratio, pad=0.04)
    plt.show()

plot_lstm_grid(np.mean(scores, axis=0),
               title='R-squared', vmin=0, vmax=1)
