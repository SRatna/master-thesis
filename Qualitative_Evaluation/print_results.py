import glob
import numpy as np
from operator import itemgetter
import pprint
pp = pprint.PrettyPrinter(indent=2)

r = [] 
p = glob.glob('./predictions/camera-all-semi-supervised-bfm/outs/output*/*/*/supervised/*/*-normalized-*/3D/results/*.txt')
# p = glob.glob('./predictions/camera-all-semi-supervised-bfm/outs/output-5k-40-80pc-20e-landmark-background_output-40k-5-80pc-20e-landmark-background/seperate-all-losses-80-pc-20-exp-0.00001-lr-0.25-train-mse-loss-5-cam-params/*/supervised/*/adam-1e-05-ep-2-pllr-1e-06-resnet-frozen-cfov-cpp-cam-params-frozen-shade-frozen-used-2-models-round*/3D/results/*.txt')
# p = glob.glob('./predictions/camera-all-semi-supervised-bfm/outs/output-5k-40-80pc-20e-landmark-background_output-40k-5-80pc-20e-landmark-background/seperate-all-losses-80-pc-20-exp-0.00001-lr-0.25-train-mse-loss-5-cam-params/*/supervised/*/*/3D/results/*.txt')
for i in p:
  # if '/cosine_1/' not in i:
  #   continue
  # if '-parent-loaded' not in i:
  #   continue
  # if ('/output-40k' not in i) or ('pllr' in i):
  # if ('2-cam-params' not in i) or ('-round-' not in i):
  # if ('5-cam-params' not in i) or ('-cam-params-frozen' not in i):
  #   continue
  e = np.loadtxt(i, dtype=str)
  c_r = {
    'path': i,
    'median': e[3] if e.ndim == 1 else e[0][3]
  }
  r.append(c_r)

# r = []
p = glob.glob('./predictions/*/*/3D/results/*.txt')
for i in p:
  if ('5-cam-params' not in i):
    continue
  e = np.loadtxt(i, dtype=str)
  c_r = {
    'path': i,
    'median': e[3] if e.ndim == 1 else e[0][3]
  }
  # r.append(c_r)

# r = []
p = glob.glob('./predictions/bfm-data/bfm-code-new/outs/output-*/*/3D/results/*.txt')
for i in p:
  e = np.loadtxt(i, dtype=str)
  c_r = {
    'path': i,
    'median': e[3] if e.ndim == 1 else e[0][3]
  }
  # r.append(c_r)

n_r = sorted(r, key=itemgetter('median'))
# n_r = sorted(r, key=itemgetter('path'))
c = 1
for i in n_r[:10]:
  print(c)
  pp.pprint(i)
  print()
  print()
  c += 1