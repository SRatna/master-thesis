import glob
import numpy as np
import os
import shutil

# p = glob.glob('./predictions/*/*/3D/*/*/*.npy')
# p = glob.glob('./predictions/*/*/3D/results/*.txt')
p = glob.glob('./predictions/bfm-data/bfm-code-new/outs/output-*/*/3D/')
# p = glob.glob('./predictions/camera-all-semi-supervised-bfm/outs/output*/*/*/supervised/*/*/3D/results/*.txt')
# p = glob.glob('./predictions/camera-all-semi-supervised-bfm/outs/output*/*/*/supervised/*/*/3D/*/*/*.npy')
# p = glob.glob('./predictions/camera-all-semi-supervised-bfm/outs/output*/*/*/supervised/*/*/3D/results')

# for i in p:
#   print(i)
#   # print(os.path.exists(f'{i}results/'))
#   # print()
#   # # break
#   e = np.loadtxt(i, dtype=str)
#   print(e[0])
#   print()
#   # break

# t = '/home/sadhikari/now_evaluation/predictions/camera-all-semi-supervised-bfm/outs/output-40k-5-80pc-20e-landmark-no-background_output-20k-10-80pc-20e-landmark-no-background_output-20k-10-80pc-20e-landmark-background_output-40k-5-80pc-20e-landmark-background/seperate-all-losses-80-pc-20-exp-0.00001-lr-0.5-train-mse-loss/ffhq-all/supervised/photo_loss_cosine_1/adam-1e-06-ep-2-frozen-True-cfov-True/3D/results/semi-supervised-cuke-errors.txt'
# e = np.loadtxt(t, dtype=str)
# print(e)
# print(os.path.getsize(t))

print(len(p), p[0])
# for landmarks_path in p:
#   shutil.copyfile('./bfm_reconstruction/seven_landmarks_corrected.npy', landmarks_path)

for i in p:
  shutil.move(i, f'{i}-with-reversed-landmarks')