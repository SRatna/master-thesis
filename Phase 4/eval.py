import os
import torch
import yaml
import pickle
import torch.nn as nn
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datasets import CelebADataset, TDMMDataset, SfSNetDataset, FFHQDataset
from utils import save_image, BFM2017MeshRenderer, dotdict
import resnet50_ft_dims_2048 as resnet50_model
from loss import ImageNetVGG19Loss, VGGFaceResNet50Loss, photo_loss, PerceptualLoss
from model import MixedModels
from log import telegram_logger as tl

if __name__ == '__main__':
    config_file = open('config.yml', 'r')
    args = yaml.load(config_file, Loader=yaml.FullLoader)
    args = dotdict(args)
    
    print(args)
    device = torch.device(args.device)
    tl("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    if args.data == 'celeba':
        full_dataset = CelebADataset(args.data_dir)
    if args.data == '3dmm':
        full_dataset = TDMMDataset(args.data_dir)
    if args.data == 'sfsnet':
        full_dataset = SfSNetDataset(args.data_dir)
    if args.data == 'ffhq':
        full_dataset = FFHQDataset(args.data_dir)
    if args.data == 'bfm':
        full_dataset = TDMMDataset(args.data_dir + 'train/img/*/')
    dataset_size = len(full_dataset)
    validation_count = int(args.val_split * dataset_size / 100)
    train_count = dataset_size - validation_count
    tl(f'Train count {train_count}')
    train_dataset, val_dataset = random_split(
        full_dataset, [train_count, validation_count])
    syn_train_dl  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    print(train_count)
    
    args['constant_pose'] = False # false during evaluation
    
    bfm_model = BFM2017MeshRenderer(args)
    
    mixed_models = MixedModels(args, bfm_model)
    mixed_models.eval()
    out_dir = 'tmp'
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        for bix, data in enumerate(syn_train_dl):
            face = data
            face   = face.to(device)
            out_recon, masks = mixed_models(face)
            save_image(face, f'{out_dir}/face_{bix}_eval.png')
            save_image(out_recon, f'{out_dir}/recon_face_{bix}_eval.png')
            save_image((face * masks), f'{out_dir}/masked_recon_face_{bix}_eval.png')
            if bix > 10: break
    tl('done')






