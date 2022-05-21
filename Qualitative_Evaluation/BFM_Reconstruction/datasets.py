from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import glob
import numpy as np

IMAGE_SIZE = 224
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class CelebADataset(Dataset):
    def __init__(self, path=''):
        face = []

        for img in sorted(glob.glob(path + '*')):
            face.append(img)

        self.face = face
        self.dataset_len = len(self.face)

    def __getitem__(self, index):
        face = transform(Image.open(self.face[index]).convert('RGB'))
        return face, self.face[index]

    def __len__(self):
        return self.dataset_len

class TDMMDataset(Dataset):
    def __init__(self, path=''):
        face = []

        for img in sorted(glob.glob(path + '*.png')):
            face.append(img)

        self.face = face
        self.dataset_len = len(self.face)

    def __getitem__(self, index):
        face = transform(Image.open(self.face[index]).convert('RGB'))
        return face

    def __len__(self):
        return self.dataset_len

class SfSNetDataset(Dataset):
    def __init__(self, path=''):
        face = []

        for img in sorted(glob.glob(path + '*/*_face_*')):
            face.append(img)

        self.face = face
        self.dataset_len = len(self.face)

    def __getitem__(self, index):
        face = transform(Image.open(self.face[index]).convert('RGB'))
        return face

    def __len__(self):
        return self.dataset_len

class NowDataset(Dataset):
    def __init__(self, root_path, val_file):
        self.root_img_path = f'{root_path}/iphone_pictures/'
        if val_file:
            self.val_paths = np.loadtxt(val_file, dtype=str)
        else:
            # this is for train set
            self.val_paths = sorted(glob.glob(f'{self.root_img_path}*/*/*.jpg'))
            self.root_img_path = ''
        self.dataset_len = len(self.val_paths)

    def __getitem__(self, index):
        val_path = self.val_paths[index]
        img_path = self.root_img_path + val_path
        bb_path = img_path.replace('iphone_pictures', 'detected_face').replace('.jpg', '.npy') #bounding box npy file path
        img = np.array(Image.open(img_path))
        bbox = np.load(bb_path, allow_pickle=True, encoding='bytes')
        # crop face with bb (bounding box)
        x, y, _ = img.shape
        top = bbox.item()[b'top']
        bottom = bbox.item()[b'bottom']
        right = bbox.item()[b'right']
        left = bbox.item()[b'left']
        # add third part of bb to top and rest to bottom
        diff = max(bottom - top, right - left)
        top = top - (4/10)*diff
        bottom = bottom + (2/10)*diff
        left = left - (3/10)*diff
        right = right + (3/10)*diff
        
        top = 0 if top < 0 else int(top)
        bottom = x if bottom > x else int(bottom)
        left = 0 if left < 0 else int(left)
        right = y if right > y else int(right)
        # make square face
        img = img[top:bottom, left:right]
        s = max(img.shape[0:2])
        #Creating a dark square with NUMPY  
        face = np.zeros((s,s,3), np.uint8)
        #Getting the centering position
        ax, ay = (s - img.shape[1])//2,(s - img.shape[0])//2
        #Pasting the 'image' in a centering position
        face[ay:img.shape[0]+ay, ax:ax+img.shape[1]] = img
        face = transform(Image.fromarray(face))
        return face, val_path

    def __len__(self):
        return self.dataset_len
