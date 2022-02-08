import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class TrainSet(Dataset):
    
    def __init__(self, path, coeff_count=40, train_size=0.9):
        super().__init__()
        self.path = path
        data = np.loadtxt(self.path + 'coeff.dat', delimiter=',', dtype=np.float32)
        data = np.unique(data, axis=0)
        split = int(np.floor(train_size * len(data)))
        train_data = data[:split]            
        self.labels = {}
        for item in train_data:
            index = int(item[0])
            theta = item[1]
            phi = item[2]
            shape_coeff = item[3:coeff_count+3][:coeff_count]
            texture_coeff = item[coeff_count+3:][:coeff_count]
            label = np.hstack((theta, phi, shape_coeff, texture_coeff))
            self.labels[index] = label
        del data
        del train_data
        self.transform = transforms.Compose([
                            transforms.Resize(224),
#                             transforms.CenterCrop(224),
#                             transforms.RandomHorizontalFlip(), # randomly flip and rotate
#                             transforms.RandomRotation(10),
#                             transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.5, hue=0.15),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = Image.open(self.path + 'img' + str(idx).zfill(6) + '.png')
        image = image.convert('RGB')
        label = self.labels[idx]
        return self.transform(image), label
    
class TestSet(Dataset):
    
    def __init__(self, path, coeff_count=40, train_size=0.9):
        super().__init__()
        self.path = path
        data = np.loadtxt(self.path + 'coeff.dat', delimiter=',', dtype=np.float32)
        data = np.unique(data, axis=0)
        self.offset = int(np.floor(train_size * len(data)))
        test_data = data[self.offset:]
        self.labels = {}
        for item in test_data:
            index = int(item[0])
            theta = item[1]
            phi = item[2]
            shape_coeff = item[3:coeff_count+3][:coeff_count]
            texture_coeff = item[coeff_count+3:][:coeff_count]
            label = np.hstack((theta, phi, shape_coeff, texture_coeff))
            self.labels[index] = label
        del data
        del test_data
        self.transform = transforms.Compose([
                            transforms.Resize(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        idx = self.offset + idx
        image_path = self.path + 'img' + str(idx).zfill(6) + '.png'
        image = Image.open(image_path)
        image = image.convert('RGB')
        label = self.labels[idx]
        return self.transform(image), label
