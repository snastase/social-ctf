import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from behavior_heuristics import (following_teammate, escort, cycling,
                                 assist, spawn_camping, mobbing)


# Function for reduced-memory incremental eigendecomposition
def incremental_cvpca(train_data, test_data, n_components=None,
                      train_filename=None, test_filename=None):

    # If number of samples isn't specified, infer from data
    n_features = train_data.shape[1]
    n_total = train_data.shape[0]
    
    # Incrementally populate covariance matrix
    cov = np.zeros((n_features, n_features))
    for i, row in zip(np.arange(n_total), train_data):
        outer = np.outer(row, row)
        cov += outer / (n_total - 1)
        if i % 10000 == 0:
            print(f"Finished computing game {i}th covariance", flush=True)

    # Recover eignvalues and eigenvectors
    vals, vecs = eigh(cov)

    # Reorder values, vectors and extract column eigenvectors
    vals, vecs = vals[::-1], vecs[:, ::-1]
    if n_components:
        vecs = vecs[:, :n_components]

    # Project training data onto the eigenvectors
    train_proj = np.memmap(train_filename, dtype='float64',
                           mode='w+', shape=(train_data.shape[0],
                                             n_components))
    test_proj = np.memmap(test_filename, dtype='float64',
                          mode='w+', shape=(test_data.shape[0],
                                            n_components))

    train_proj[:] = train_data @ vecs
    train_proj.flush()
    
    test_proj[:] = test_data @ vecs
    test_proj.flush()

    return vals, vecs


matchup_id = 0
n_pcs = 142
n_isfcs = n_pcs * (n_pcs - 1) // 2 + n_pcs
team_pairs = {0: {'coop': 0, 'comp': [1, 2]},
              1: {'coop': 0, 'comp': [3, 4]},
              2: {'coop': 5, 'comp': [1, 3]},
              3: {'coop': 5, 'comp': [2, 4]}}
coop_pairs = [0, 0, 5, 5]
player_id = slice(None)

#f = f'results/decode-behav_isfcs_matchup-{matchup_id}_map-save.npy'
#iscf_stack = np.load(f, mmap_mode='r')
label_stack = np.load(f'results/decode-behav_labels_matchup-{matchup_id}.npy')
map_stack = np.load(f'results/decode-behav_maps_matchup-{matchup_id}.npy')
#assert iscf_stack.shape[0] == label_stack.shape[0] == map_stack.shape[0]
#print(iscf_stack.shape)


cv = PredefinedSplit(map_stack)

for f, (train, test) in enumerate(cv.split()):
    
    train_shape = (len(train), n_isfcs)
    test_shape = (len(test), n_isfcs)
    print(f"Training shape: {train_shape}", flush=True)
    
    test_map_id = np.unique(map_stack[test])[0]
    train_f = (f'results/decode-behav_isfcs_'
               f'matchup-{matchup_id}_map-{test_map_id}_train.npy')
    test_f = (f'results/decode-behav_isfcs_'
              f'matchup-{matchup_id}_map-{test_map_id}_test.npy')
    train_pca_f = (f'results/decode-behav_isfcs_'
                   f'matchup-{matchup_id}_map-{test_map_id}_cvpca-train.npy')
    test_pca_f = (f'results/decode-behav_isfcs_'
                  f'matchup-{matchup_id}_map-{test_map_id}_cvpca-test.npy')
    
    iscf_train = np.memmap(train_f, dtype='float64',
                       mode='w+', shape=train_shape)
    iscf_test = np.memmap(test_f, dtype='float64',
                          mode='w+', shape=test_shape)
    
    train_start = 0
    for map_id in np.unique(map_stack[train])[:2]:
        iscf_train_map = np.load(
            f'results/decode-behav_isfcs_matchup-{matchup_id}_map-{map_id}.npy',
            mmap_mode='r')
        iscf_map_mean = np.mean(iscf_train_map, axis=0)
        iscf_map_std = np.std(iscf_train_map, axis=0)
        
        train_len = iscf_train_map.shape[0]
        iscf_train[train_start : train_start + train_len] = (
            (iscf_train_map - iscf_map_mean) / iscf_map_std)
        train_start += train_len
        print(f"Mem-mapped training map {map_id}", flush=True)
        
    iscf_test[:] = np.load(
        f'results/decode-behav_isfcs_matchup-{matchup_id}_map-{test_map_id}.npy',
        mmap_mode='r')
    print(f"Finished mem-mapping fold {f}", flush=True)
    raise

    vals, vecs = incremental_cvpca(
        iscf_train[:train_start], iscf_test, n_components=n_pcs,
        train_filename=train_pca_f, test_filename=test_pca_f)
    raise


train_f = f'results/decode-behav_isfcs_matchup-{matchup_id}_train.npy'
test_f = f'results/decode-behav_isfcs_matchup-{matchup_id}_test.npy'

scaler = StandardScaler(copy=False)
#pca = PCA(n_components=n_pcs)
#model = SGDClassifier(loss='log', penalty='none')
model = LogisticRegression(penalty='none', class_weight='balanced',
                           solver='saga', n_jobs=6)

scores, label_preds = [], []
for f, (train, test) in enumerate(cv.split()):
    
    train_shape = (len(train), n_isfcs)
    test_shape = (len(test), n_isfcs)
    print(f"Training shape: {train_shape}", flush=True)
    
    iscf_train = np.memmap(train_f, dtype='float64',
                           mode='w+', shape=train_shape)
    iscf_test = np.memmap(test_f, dtype='float64',
                          mode='w+', shape=test_shape)
    raise
    
    train_start = 0
    for map_id in np.unique(map_stack[train]):
        iscf_train_map = np.load(
            f'results/decode-behav_isfcs_matchup-{matchup_id}_map-{map_id}.npy',
            mmap_mode='r')
        train_len = iscf_train_map.shape[0]
        iscf_train[train_start : train_start + train_len] = iscf_train_map
        train_start += train_len
        print(f"Mem-mapped training map {map_id}")
    
    map_id = np.unique(map_stack[test])[0]
    iscf_test[:] = np.load(
            f'results/decode-behav_isfcs_matchup-{matchup_id}_map-{map_id}.npy',
            mmap_mode='r')
    
    #iscf_train[:] = iscf_stack[train]
    #iscf_test[:] = iscf_stack[test]
    print(f"Finished mem-mapping fold {f}", flush=True)
    
    scaler.fit_transform(iscf_train)
    scaler.transform(iscf_test)
    print(f"Finished scaling fold {f}", flush=True)
    
    model.fit(iscf_train, label_stack[train])
    #model.fit(iscf_train, label_perm[train])
    label_pred = model.predict(iscf_test)
    score = balanced_accuracy_score(label_stack[test], label_pred)
    #score = balanced_accuracy_score(label_perm[test], label_pred)
    scores.append(score)
    label_preds.append(label_pred)
    print(f"Finished cross-validation fold {f}", flush=True)
    
np.save(f'results/decode-behav_isfc-scores_matchup-{matchup_id}.npy', scores)
np.save(f'results/decode-behav_isfc-preds_matchup-{matchup_id}.npy', label_preds)
