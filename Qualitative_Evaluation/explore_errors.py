import glob
import numpy as np
import os
import shutil

path = 'predictions/bfm-data/code-all-params/outs/output-5k-40-80pc-20e-landmark-background_output-40k-5-80pc-20e-landmark-background/seperate-all-losses-80-pc-20-exp-0.00001-lr-0.25-train-mse-loss-5-cam-params/3D/results/_computed_distances_.npy'

errors = np.load(path, allow_pickle=True, encoding="latin1").item()['computed_distances']

print(errors[0])

print(len(errors))

print(np.mean(errors[0]))