#!/jukebox/hasson/snastase/miniconda3/envs/ctf/bin/python

import os
import numpy as np
from sklearn import manifold
import pickle
import time

def filter_pcs(pca_structure,select_pcs,map_id=slice(None), matchup_id=slice(None), repeat_id=slice(None),
    player_id=slice(None), time=slice(None, None, 500),melt = True):
    pcs = np.squeeze(pca_structure[map_id, matchup_id, repeat_id, player_id,time,select_pcs])

    if melt:
        if len(select_pcs) > 1:
            pcs = np.reshape(np.moveaxis(pcs,0,-1),(-1,len(select_pcs)))                 
    return pcs

filter_kwargs_tsne = dict(
    map_id=slice(None),
    matchup_id=0,
    repeat_id=slice(None),
    player_id=slice(None),
    time=slice(None),
    melt = True
)

n_repeats = 32 # 32 total
n_maps = 32
n_pcs = 142

print(n_pcs)

lstms_pca_maps = np.memmap('results/lstms_pca_stack.npy',dtype='float64',mode='w+',shape=(n_maps,1,n_repeats,4,4501,150))
for m in np.arange(n_maps):
    for r in np.arange(n_repeats):
        loaded = np.load(f"/jukebox/hasson/snastase/social-ctf/results/lstms-pca_matchup-0_map-{m}_repeat-{r}.npy")
        # print(loaded[:,:,0:150].shape)
        lstms_pca_maps[m][0][r] = loaded[:,:,0:150]
print(lstms_pca_maps.shape)

filtered_pcs = filter_pcs(lstms_pca_maps,np.arange(n_pcs),**filter_kwargs_tsne)
print(filtered_pcs.shape)


def get_tsne_embedding_parallel(filtered_pcs, n_components=2, random_state=0,n_jobs=24,learning_rate = 'auto'):
    embedding = manifold.TSNE(n_components, random_state=random_state,n_jobs=n_jobs,learning_rate = learning_rate)
    filtered_pcs_tsne = embedding.fit_transform(filtered_pcs)
    return filtered_pcs_tsne, embedding

start = time.time()

filtered_pcs_tsne, embedding = get_tsne_embedding_parallel(filtered_pcs, n_components=2)
print(filtered_pcs_tsne.shape)
print(embedding.learning_rate)

end = time.time()
print(end - start)

np.save('results/filtered_pcs_tsne_32maps.npy',filtered_pcs_tsne)

print("saved tsne matrix successfully")

# NOTE: save fitted embedding object with pickle 

with open('results/filtered_pcs_tsne_32maps.pkl','wb') as f: 
    pickle.dump(embedding, f)
    

print("saved tsne embedding successfully")