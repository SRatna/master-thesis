import glob
from PIL import Image
import os

path = '/work/ws-tmp/g051151-3dmm/bfm/output-20000-5-160pc/'

for img_path in sorted(glob.glob(path + '*/img/*/*')):
    try:
        img = Image.open(img_path).convert('RGB') # open the image file
    except Exception as e:
        print('Bad file:', img_path) # print out the names of corrupt files
        os.remove(img_path)
