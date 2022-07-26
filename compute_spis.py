#!/usr/bin/env python3

from itertools import product
from sys import argv
import numpy as np
from pyspi.calculator import Calculator


matchup_id = 0
n_maps = 32
n_repeats = 32
n_pcs = 142
params = {i: p for i, p in enumerate(product(np.arange(n_maps),
                                             np.arange(n_pcs)))}

map_id, pc_id = params[int(argv[1])]


for repeat_id in np.arange(n_repeats):
    lstms = np.load(f'results/lstms-pca_matchup-{matchup_id}_'
                    f'map-{map_id}_repeat-{repeat_id}.npy')[..., pc_id]

    # Initialize and run PySPI calculator
    print(f"Running PySPI on map {map_id}, "
          f"repeat {repeat_id}, PC{pc_id + 1}", flush=True)
    calc = Calculator(dataset=lstms, fast=True)
    calc.compute()

    table_f = (f'results/spis-fast_pc-{pc_id}_matchup-{matchup_id}_'
               f'map-{map_id}_repeat-{repeat_id}.csv')
    calc.table.to_csv(table_f)

    print(f"Finished running PySPI on map {map_id}, "
          f"repeat {repeat_id}, PC{pc_id + 1}")
