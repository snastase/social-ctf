#!/jukebox/hasson/snastase/miniconda3/envs/ctf/bin/python

import os
import numpy as np
from sklearn import manifold
import pickle
import time

n_maps = 8
n_repeats = 32
n_players = 4
n_samples = 4501

n_total = n_maps * n_repeats * n_players * n_samples

print(n_total)

# filtered_pcs = np.memmap(f'/tmp/snastase/lstms-stack_tanh-z_pca-proj_matchup-0.npy', mode='r',shape = (n_total,512))[:, :142]

filtered_pcs = np.memmap(f'/jukebox/hasson/snastase/social-ctf/results/lstms-stack_tanh-z_pca-proj_matchup-0.npy', mode='r',dtype='float64',shape = (n_total,512))[:,:142]
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

# NOTE: save fitted embedding object with pickle 

#with open('results/maps4_filtered_pcs_tsne.pkl','wb') as f: 
    #pickle.dump(embedding, f)
    
#np.save('results/maps4_filtered_pcs_tsne.npy',filtered_pcs_tsne)
