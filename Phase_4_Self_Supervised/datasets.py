from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import glob

IMAGE_SIZE = 224
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
normalized_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class CelebADataset(Dataset):
    def __init__(self, path=''):
        face = []

        for img in sorted(glob.glob(path + '*_face*')):
            face.append(img)

        self.face = face
        self.dataset_len = len(self.face)

    def __getitem__(self, index):
        face = transform(Image.open(self.face[index]).convert('RGB'))
        return face

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

class FFHQDataset(Dataset):
    def __init__(self, path='', normalize=False):
        face = []

        for img in sorted(glob.glob(path + '*/*.png')):
            face.append(img)

        self.face = face
        self.normalize = normalize
        self.dataset_len = len(self.face)

    def __getitem__(self, index):
        if self.normalize:
            face = normalized_transform(Image.open(self.face[index]).convert('RGB'))
        else: face = transform(Image.open(self.face[index]).convert('RGB'))
        return face

    def __len__(self):
        return self.dataset_len
