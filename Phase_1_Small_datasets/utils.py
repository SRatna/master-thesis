import time
import copy
import torch
import yaml
from log import *
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from datasets import TrainSet, TestSet

class dotdict(dict):
    __getattr__ = dict.get

def loadArgs():
    config_file = open('config.yml', 'r')
    args = yaml.load(config_file, Loader=yaml.FullLoader)
    return dotdict(args)

class NewBranch(nn.Module):
    def __init__(self, input_size, coeff_count):
        super(NewBranch, self).__init__()
        self.fc_poses = nn.Linear(input_size, 2)
        self.fc_shape = nn.Linear(input_size, coeff_count)
        self.fc_texture = nn.Linear(input_size, coeff_count)

    def forward(self, inputs):
        _, x = inputs
        x = x.squeeze(2)
        x = x.squeeze(2)
        out_poses = self.fc_poses(torch.relu(x))
        out_shape = self.fc_shape(torch.relu(x))
        out_texture = self.fc_texture(torch.relu(x))
        return out_poses, out_shape, out_texture

def get_data_loaders(args):
    # number of subprocesses to use for data loading
    num_workers = 8
    # percentage of training set to use as validation
    valid_size = 0.2
    if args.concat:
        train_data1 = TrainSet(f'{args.data_dirs[0]}/', args.coeff_count)
        train_data2 = TrainSet(f'{args.data_dirs[1]}/', args.coeff_count)
        train_data3 = TrainSet(f'{args.data_dirs[2]}/', args.coeff_count)
        train_data = ConcatDataset([train_data1, train_data2, train_data3])
        telegram_logger('All concatinated')
    else:
        train_data = TrainSet(f'{args.data_dirs[0]}/', args.coeff_count)
        telegram_logger(f'{args.data_dirs[0]}/')
    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
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
    telegram_logger(f'Train size: {len(train_idx)}, Val size: {len(valid_idx)}')
    return dataloaders, dataset_sizes


def train(model, dataloaders, dataset_sizes, args):
    telegram_logger('Training Started')
    criterion = nn.MSELoss()
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
    lambda_poses = 0.1
    lambda_shape = 0.5
    lambda_texture = 0.5
    coeff_index = args.coeff_count + 2  # 2 for angles
    coeff_index_2 = 2*args.coeff_count + 2  # 2 for angles
    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss_poses = 0.0
            running_loss_shape = 0.0
            running_loss_texture = 0.0
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
                    out_poses, out_shape, out_texture = model(inputs)
                    loss_poses = criterion(out_poses, labels[:, :2])
                    loss_shape = criterion(
                        out_shape, labels[:, 2:coeff_index])
                    loss_texture = criterion(
                        out_texture, labels[:, coeff_index:coeff_index_2])
                    loss_total = lambda_poses*loss_poses + lambda_shape*loss_shape + \
                        lambda_texture*loss_texture
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_total.backward()
                        optimizer.step()

                # statistics
                running_loss_poses += loss_poses.item() * inputs.size(0)
                running_loss_shape += loss_shape.item() * inputs.size(0)
                running_loss_texture += loss_texture.item() * inputs.size(0)
                running_loss_total += loss_total.item() * inputs.size(0)

            if phase == 'train' and scheduler != None:
                scheduler.step()

            epoch_loss_poses = running_loss_poses / dataset_sizes[phase]
            epoch_loss_shape = running_loss_shape / dataset_sizes[phase]
            epoch_loss_texture = running_loss_texture / dataset_sizes[phase]
            epoch_loss_total = running_loss_total / dataset_sizes[phase]
            loss_dict[phase].append({
                'poses': epoch_loss_poses,
                'shape': epoch_loss_shape,
                'texture': epoch_loss_texture,
                'total': epoch_loss_total
            })
            print('{} Poses Loss: {:.4f} Shape Loss: {:.4f} Texture Loss: {:.4f} Total Loss: {:.4f}'.format(
                phase, epoch_loss_poses, epoch_loss_shape, epoch_loss_texture, epoch_loss_total))

            if phase == 'val':
                if (epoch_loss_total - lowest_loss) < (-1e-3):
                    lowest_loss = epoch_loss_total
                    best_model_wts = copy.deepcopy(model.state_dict())
                    early_stopping_count = 0
                else:
                    early_stopping_count += 1
                    print('Early Stopping {}/{}'.format(early_stopping_count, args.patience))

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
    for loss_type in ['poses', 'shape', 'texture', 'total']:
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
        plt.savefig(f'{path}-{loss_type}-loss.png')
        telegram_upload_image(f'{path}-{loss_type}-loss.png')


def print_test_loss(model, args):
    if args.concat:
        test_loader = DataLoader(ConcatDataset([
            TestSet(f'{args.data_dirs[0]}/', args.coeff_count),
            TestSet(f'{args.data_dirs[1]}/', args.coeff_count),
            TestSet(f'{args.data_dirs[2]}/', args.coeff_count)
        ]))
    else:
        test_loader = DataLoader(TestSet(f'{args.data_dirs[0]}/', args.coeff_count))
    coeff_index = args.coeff_count + 2  # 2 for angles
    coeff_index_2 = 2*args.coeff_count + 2  # 2 for angles
    theta_loss = 0.
    phi_loss = 0.
    shape_coeff_loss = 0.
    texture_coeff_loss = 0.
    model.eval()
    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(args.device), label.to(args.device)
            prediction1, p_shape_coeff, p_texture_coeff = model(image)
            shape_coeff = label[0][2:coeff_index]
            texture_coeff = label[0][coeff_index:coeff_index_2]
            theta_loss += (prediction1[0][0] - label[0][0]).pow(2).item()
            phi_loss += (prediction1[0][1] - label[0][1]).pow(2).item()
            shape_coeff_loss += (p_shape_coeff -
                                 shape_coeff).pow(2).mean().item()
            texture_coeff_loss += (p_texture_coeff -
                                   texture_coeff).pow(2).mean().item()
    theta_loss /= len(test_loader)
    phi_loss /= len(test_loader)
    shape_coeff_loss /= len(test_loader)
    texture_coeff_loss /= len(test_loader)

    telegram_logger(f'theta loss: {theta_loss}')
    telegram_logger(f'phi loss: {phi_loss}')
    telegram_logger(f'shape coeff loss: {shape_coeff_loss}')
    telegram_logger(
        f'texture coeff loss: {texture_coeff_loss}')
    
    return {
        'theta': theta_loss,
        'phi': phi_loss,
        'shape': shape_coeff_loss,
        'texture': texture_coeff_loss,
    }