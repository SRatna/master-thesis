from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim import lr_scheduler
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import copy
import glob
import json
import torch
import random
from log import *
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torch.utils.data import ConcatDataset

import matplotlib
matplotlib.use('Agg')


class NewBranch(nn.Module):
    def __init__(self, input_size, args):
        super(NewBranch, self).__init__()
        self.fc_camera_params = nn.Linear(input_size, args.cam_params)  # pitch, yaw, pp, fov
        self.fc_shape = nn.Linear(input_size, args.coeff_count)
        self.fc_texture = nn.Linear(input_size, args.coeff_count)
        self.fc_exp = nn.Linear(input_size, args.exp_count)
        self.fc_shade = nn.Linear(input_size, 27)

    def forward(self, inputs):
        _, x = inputs
        x = x.squeeze(2)
        x = x.squeeze(2)
        out_camera_params = self.fc_camera_params(torch.relu(x))
        out_shape = self.fc_shape(torch.relu(x))
        out_texture = self.fc_texture(torch.relu(x))
        out_exp = self.fc_exp(torch.relu(x))
        out_shade = self.fc_shade(torch.relu(x))
        return out_camera_params, out_shape, out_texture, out_exp, out_shade


class BFMDataset(Dataset):
    def __init__(self, path, coeff_count, exp_count, train_size=1):
        super().__init__()
        self.path = path
        self.coeff_count = coeff_count
        self.exp_count = exp_count
        faces = []
        for img in sorted(glob.glob(path + 'img/*/*')):
            faces.append(img)
        num_samples = int(np.floor(train_size * len(faces)))
        self.faces = random.sample(faces, num_samples)
        # for label in sorted(glob.glob(path + 'rps/*/*')):
        #    self.labels.append(label)
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
        return len(self.faces)

    def __getitem__(self, idx):
        rps_path = self.faces[idx].replace(
            '/img/', '/rps/').replace('.png', '.rps')
        image = Image.open(self.faces[idx]).convert('RGB')
        with open(rps_path) as json_file:
            data = json.load(json_file)
        pitch = [data['pose']['pitch']]
        yaw = [data['pose']['yaw']]
        principal_point = data['camera']['principalPoint']
        focal_length = data['camera']['focalLength']
        # 7.5 is half of sensor size which is 15, 15
        fov = [np.arctan(7.5/focal_length)*2]
        shape_coeff = data['momo']['shape'][:self.coeff_count]
        texture_coeff = data['momo']['color'][:self.coeff_count]
        exp_coeff = data['momo']['expression'][:self.exp_count]
        shading = np.array(data['environmentMap']['coefficients']).reshape(-1)
        label = np.hstack(
            (pitch, yaw, principal_point, fov, shape_coeff, texture_coeff, exp_coeff, shading))
        return self.transform(image), label


def get_data_loaders(args):
    # number of subprocesses to use for data loading
    num_workers = 8
    # percentage of training set to use as validation
    data_dirs = args.data_dir.split('/')[-1].split('_')
    root_dir = '/'.join(args.data_dir.split('/')[:-1])
    train_sets = []
    for d in data_dirs:
        train_data = BFMDataset(f'{root_dir}/{d}/train/',
                                args.coeff_count, args.exp_count, args.train_size)
        train_sets.append(train_data)
    telegram_logger(args.data_dir)
    train_data = ConcatDataset(train_sets)
    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(args.valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    samplers = {
        'train': SubsetRandomSampler(train_idx),
        'val': SubsetRandomSampler(valid_idx)
    }

    dataloaders = {x: DataLoader(train_data, batch_size=args.batch_size,
                                 sampler=samplers[x], num_workers=num_workers)
                   for x in ['train', 'val']}
    dataset_sizes = {'train': len(train_idx), 'val': len(valid_idx)}
    telegram_logger(f'Train size: {len(train_idx)}')
    return dataloaders, dataset_sizes


def train(model, dataloaders, dataset_sizes, args):
    telegram_logger('Training Started')
    if args.loss == 'mse':
        criterion = nn.MSELoss()
    if args.loss == 'l1':
        criterion = nn.L1Loss()
    if args.loss == 'smooth_l1':
        criterion = nn.SmoothL1Loss()
    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.parameters(), args.lr)
    # Decay LR by a factor of 0.6 every 5 epochs
    scheduler = lr_scheduler.StepLR(optimizer, args.step_size, args.gamma)
    device = torch.device(args.device)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    lowest_loss = np.inf
    loss_dict = {'train': [], 'val': []}
    early_stopping_count = 0
    # lambda = nn.Parameter(torch.Tensor([0.1])).to(device)
    lambda_camera_params = 0.1
    lambda_shape = 0.5
    lambda_texture = 0.5
    lambda_exp = 0.25
    lambda_shade = 0.5
    coeff_index = args.coeff_count + 5  # 5 for angles, pp, fov
    coeff_index_2 = 2*args.coeff_count + 5
    coeff_index_3 = 2*args.coeff_count + 5 + args.exp_count
    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss_camera_params = 0.0
            running_loss_shape = 0.0
            running_loss_texture = 0.0
            running_loss_exp = 0.0
            running_loss_shade = 0.0
            running_loss_total = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.float().to(device)
                labels = labels.float().to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    out_camera_params, out_shape, out_texture, out_exp, out_shade = model(
                        inputs)
                    loss_camera_params = criterion(
                        out_camera_params, labels[:, :args.cam_params]) # labels includes all but select what you need as per the cam_params number
                    loss_shape = criterion(
                        out_shape, labels[:, 5:coeff_index])
                    loss_texture = criterion(
                        out_texture, labels[:, coeff_index:coeff_index_2])
                    loss_exp = criterion(
                        out_exp, labels[:, coeff_index_2:coeff_index_3])
                    loss_shade = criterion(
                        out_shade, labels[:, coeff_index_3:])
                    loss_total = lambda_camera_params*loss_camera_params + lambda_shape*loss_shape + \
                        lambda_texture*loss_texture + lambda_exp*loss_exp + lambda_shade*loss_shade
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_total.backward()
                        optimizer.step()

                # statistics
                running_loss_camera_params += loss_camera_params.item() * inputs.size(0)
                running_loss_shape += loss_shape.item() * inputs.size(0)
                running_loss_texture += loss_texture.item() * inputs.size(0)
                running_loss_exp += loss_exp.item() * inputs.size(0)
                running_loss_shade += loss_shade.item() * inputs.size(0)
                running_loss_total += loss_total.item() * inputs.size(0)

            if phase == 'train' and scheduler != None:
                scheduler.step()

            epoch_loss_camera_params = running_loss_camera_params / \
                dataset_sizes[phase]
            epoch_loss_shape = running_loss_shape / dataset_sizes[phase]
            epoch_loss_texture = running_loss_texture / dataset_sizes[phase]
            epoch_loss_exp = running_loss_exp / dataset_sizes[phase]
            epoch_loss_shade = running_loss_shade / dataset_sizes[phase]
            epoch_loss_total = running_loss_total / dataset_sizes[phase]
            loss_dict[phase].append({
                'camera_params': epoch_loss_camera_params,
                'shape': epoch_loss_shape,
                'texture': epoch_loss_texture,
                'expression': epoch_loss_exp,
                'shade': epoch_loss_shade,
                'total': epoch_loss_total
            })
            print('{} Camera Params Loss: {:.4f} Shape Loss: {:.4f} Texture Loss: {:.4f} Expression Loss: {:.4f} Shade Loss: {:.4f} Total Loss: {:.4f}'.format(
                phase, epoch_loss_camera_params, epoch_loss_shape, epoch_loss_texture, epoch_loss_exp, epoch_loss_shade, epoch_loss_total))

            if phase == 'val':
                if (epoch_loss_total - lowest_loss) < (-1e-3):
                    lowest_loss = epoch_loss_total
                    best_model_wts = copy.deepcopy(model.state_dict())
                    early_stopping_count = 0
                else:
                    early_stopping_count += 1
                    print(
                        'Early Stopping {}/{}'.format(early_stopping_count, args.patience))

        if early_stopping_count >= args.patience:
            telegram_logger(f'Early Stopped! Done till epoch: {epoch}')
            break
        print()

    time_elapsed = time.time() - since
    telegram_logger('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    telegram_logger('Lowest val loss: {:4f}'.format(lowest_loss))
    telegram_logger(f'Done till epoch: {epoch}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, loss_dict


def save_loss_fig(loss_dict, path):
    for loss_type in ['camera_params', 'shape', 'texture', 'expression', 'shade', 'total']:
        loss = {'train': [], 'val': []}
        for phase in ['train', 'val']:
            for item in loss_dict[phase]:
                loss[phase].append(item[loss_type])
        plt.figure(figsize=(10, 6))
        data = pd.DataFrame(loss)
        sns.lineplot(data=data)
        plt.title(f'{loss_type.title()} loss: {path}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.savefig(f'{path}/{loss_type}-loss.png')
        telegram_upload_image(f'{path}/{loss_type}-loss.png')


def print_test_loss(model, args):
    device = torch.device(args.device)
    data_dirs = args.data_dir.split('/')[-1].split('_')
    root_dir = '/'.join(args.data_dir.split('/')[:-1])
    test_sets = []
    for d in data_dirs:
        test_data = BFMDataset(f'{root_dir}/{d}/test/',
                               args.coeff_count, args.exp_count)
        test_sets.append(test_data)
    test_data = ConcatDataset(test_sets)
    test_loader = DataLoader(test_data)
    telegram_logger(f'Test size: {len(test_loader)}')
    coeff_index = args.coeff_count + 5  # 5 for camera parameters
    coeff_index_2 = 2*args.coeff_count + 5  # 5 for camera parameters
    coeff_index_3 = 2*args.coeff_count + 5 + \
        args.exp_count  # 5 for camera parameters
    theta_loss = 0.
    phi_loss = 0.
    pp_loss = 0.
    fov_loss = 0.
    shape_coeff_loss = 0.
    texture_coeff_loss = 0.
    exp_coeff_loss = 0.
    shading_loss = 0.
    model.eval()
    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            prediction1, p_shape_coeff, p_texture_coeff, p_exp_coeff, p_shading = model(
                image)
            shape_coeff = label[0][5:coeff_index]
            texture_coeff = label[0][coeff_index:coeff_index_2]
            exp_coeff = label[0][coeff_index_2:coeff_index_3]
            shading = label[0][coeff_index_3:]
            theta_loss += (prediction1[0][0] - label[0][0]).pow(2).item()
            phi_loss += (prediction1[0][1] - label[0][1]).pow(2).item()
            if args.cam_params == 5:
                pp_loss += (prediction1[0][3:5] - label[0]
                            [3:5]).pow(2).mean().item()
                fov_loss += (prediction1[0][4] - label[0][4]).pow(2).item()
            shape_coeff_loss += (p_shape_coeff -
                                 shape_coeff).pow(2).mean().item()
            texture_coeff_loss += (p_texture_coeff -
                                   texture_coeff).pow(2).mean().item()
            exp_coeff_loss += (p_exp_coeff -
                               exp_coeff).pow(2).mean().item()
            shading_loss += (p_shading -
                             shading).pow(2).mean().item()
    theta_loss /= len(test_loader)
    phi_loss /= len(test_loader)
    fov_loss /= len(test_loader)
    pp_loss /= len(test_loader)
    shape_coeff_loss /= len(test_loader)
    texture_coeff_loss /= len(test_loader)
    exp_coeff_loss /= len(test_loader)
    shading_loss /= len(test_loader)

    telegram_logger(f'theta loss: {theta_loss}')
    telegram_logger(f'phi loss: {phi_loss}')
    telegram_logger(f'fov loss: {fov_loss}')
    telegram_logger(f'principal point loss: {pp_loss}')
    telegram_logger(f'shape coeff loss: {shape_coeff_loss}')
    telegram_logger(
        f'texture coeff loss: {texture_coeff_loss}')
    telegram_logger(
        f'expression coeff loss: {exp_coeff_loss}')
    telegram_logger(f'shading loss: {shading_loss}')

    return {
        'theta': theta_loss,
        'phi': phi_loss,
        'fov': fov_loss,
        'pp': pp_loss,
        'shape': shape_coeff_loss,
        'texture': texture_coeff_loss,
        'expression': exp_coeff_loss,
        'shading': shading_loss
    }
