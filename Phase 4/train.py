import os
import sys
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
from model import NewBranch
from log import telegram_logger as tl

if __name__ == '__main__':
    config_file = open('config.yml', 'r')
    args = yaml.load(config_file, Loader=yaml.FullLoader)
    args = dotdict(args)
    
    if len(sys.argv) == 2:
        args['round'] = sys.argv[1]
        
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
        full_dataset = FFHQDataset(args.data_dir, args.normalize)
    if args.data == 'bfm':
        full_dataset = TDMMDataset(args.data_dir + 'train/img/*/')
    dataset_size = len(full_dataset)
    validation_count = int(args.val_split * dataset_size / 100)
    train_count = dataset_size - validation_count
    tl(f'Train count {train_count}, LR {args.lr}, Optim {args.optim} loss type {args.loss} round {args.round} model_dir {args.model_dir}')
    train_dataset, val_dataset = random_split(
        full_dataset, [train_count, validation_count])
    syn_train_dl  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    print(train_count)
    
    bfm_model = BFM2017MeshRenderer(args)
    
    if args.supervise:
        model_ft = resnet50_model.resnet50_ft()
    else:
        model_ft = resnet50_model.resnet50_ft(weights_path=args.resnet_model_dir)
    model_ft = resnet50_model.resnet50_ft()
    if args.freeze:
        for param in model_ft.parameters():
            param.require_grad = False
    num_ftrs = model_ft.classifier.in_channels
    model_branched = nn.Sequential(model_ft, NewBranch(num_ftrs, args.coeff_count, args.exp_count, bfm_model, args.cam_params))
    if args.supervise:
        state_dict = torch.load(args.model_dir, map_location=device)
        model_branched.load_state_dict(state_dict)
        if args.parent_model_dir:
            tl(f'Loading from parent model: {args.parent_model_dir}')
            parent_state_dict = torch.load(args.parent_model_dir, map_location=device)
            fc_camera_params = {
                'weight': parent_state_dict['1.fc_camera_params.weight'],
                'bias': parent_state_dict['1.fc_camera_params.bias'],
            }
            fc_shade = {
                'weight': parent_state_dict['1.fc_shade.weight'],
                'bias': parent_state_dict['1.fc_shade.bias'],
            }
            model_branched[1].fc_camera_params.load_state_dict(fc_camera_params)
            model_branched[1].fc_shade.load_state_dict(fc_shade)
        if args.freeze_cam_params:
            for param in model_branched[1].fc_camera_params.parameters():
                param.require_grad = False
        if args.freeze_shade:
            for param in model_branched[1].fc_shade.parameters():
                param.require_grad = False
    model_branched = model_branched.to(device)

    if args.per_layer_lr:
        model_parameters = [
            {'params': model_branched[0].parameters(), 'lr': args.lr},
            {'params': model_branched[1].fc_camera_params.parameters(), 'lr': args.pllr},
            {'params': model_branched[1].fc_shape.parameters(), 'lr': args.lr},
            {'params': model_branched[1].fc_texture.parameters(), 'lr': args.lr},
            {'params': model_branched[1].fc_exp.parameters(), 'lr': args.lr},
            {'params': model_branched[1].fc_shade.parameters(), 'lr': args.pllr},
        ]
        if args.optim == 'adam':
            optimizer = torch.optim.Adam(model_parameters)
        if args.optim == 'sgd':
            optimizer = torch.optim.SGD(model_parameters)
        # if not args.freeze:
        #     model_parameters.append({'params': model_branched[0].parameters()})
        # if not args.freeze_cam_params:
        #     model_parameters.append({'params': model_branched[1].fc_camera_params.parameters(), 'lr': args.pllr})
        # if not args.freeze_shade:
        #     model_parameters.append({'params': model_branched[1].fc_shade.parameters(), 'lr': args.pllr})
    else:
        model_parameters = model_branched.parameters()
        if args.optim == 'adam':
            optimizer = torch.optim.Adam(model_parameters, lr=args.lr)
        if args.optim == 'sgd':
            optimizer = torch.optim.SGD(model_parameters, lr=args.lr)

    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    recon_loss = nn.MSELoss()
    recon_lambda = 1
    if args.loss == 'l1':
        recon_loss  = nn.L1Loss()
    if args.loss == 'vgg':
        recon_loss  = ImageNetVGG19Loss().to(args.device)
    if args.loss == 'resnet':
        recon_loss  = VGGFaceResNet50Loss(args.resnet_model_dir).to(args.device)
    if 'cosine' in args.loss:
        recon_lambda = float(args.loss.split('_')[-1])
        recon_loss  = PerceptualLoss(args.resnet_model_dir).to(args.device)

    model_branched.train()

    num_epochs = args.epochs
    lamda_recon  = 1
    syn_train_len  = len(syn_train_dl)
    loss_dict = {
        'recon': []
    }
    
    # to log images and models
    train_type = 'supervised' if args.supervise else 'unsupervised'
    out_dir = '/'.join(args.model_dir.split('/')[5:-1]) + '/' + args.data_dir.split('/')[-2] + '/' + train_type + '/' + args.loss + '/'
    path = f"{args.optim}-{args.lr}-ep-{num_epochs}-landmarks-fixed"
    if args.per_layer_lr:
        path += f'-pllr-{args.pllr}'
    if args.normalize:
        path += '-normalized'
    if args.freeze:
        path += '-resnet-frozen'
    if args.constant_fov:
        path += '-cfov'
    if args.constant_pp:
        path += '-cpp'
    if args.constant_pose:
        path += '-cpose'
    if args.use_mask:
        path += '-mask-used'
    if args.freeze_cam_params:
        path += '-cam-params-frozen'
    if args.freeze_shade:
        path += '-shade-frozen'
    if args.parent_model_dir:
        path += f'-{args.parent_type}-parent'
        if args.parent_type == 'finetuned' and args.load_parent:
            path += '-loaded'
    path += f'-round-{args.round}'
    out_dir += path
    os.makedirs(out_dir, exist_ok=True)
    tl(path)
    for epoch in range(1, num_epochs+1):
        tl(f'Current epoch {epoch}')
        rloss = 0 # Reconstruction loss

        for bix, data in enumerate(syn_train_dl):
            face = data
            face   = face.to(device)
            
            # Apply Mask on input image
            #face = face * mask
            out_recon, masks = model_branched(face)
            if args.use_mask:
                masked_face = face * masks
            else:
                masked_face = face
            
            if 'photo_loss' in args.loss:
                current_recon_loss = recon_lambda * photo_loss(out_recon, face, masks)
                if 'cosine' in args.loss:
                    current_recon_loss += recon_loss(out_recon, masked_face)
            else:
                current_recon_loss  = recon_loss(out_recon, masked_face)


            optimizer.zero_grad()
            current_recon_loss.backward()
            optimizer.step()
            exp_lr_scheduler.step()


            # Logging for display and debugging purposes
            rloss += current_recon_loss.item()
            
        save_image(face, f'{out_dir}/face_{epoch}.png', denorm=args.normalize)
        save_image(out_recon, f'{out_dir}/recon_face_{epoch}.png')
        # save_image((face * masks), f'{out_dir}/masked_recon_face_{epoch}.png')

        cur_rloss = rloss / syn_train_len
        loss_dict['recon'].append(cur_rloss)
        tl(f'Is Semi?: {args.supervise} Mask: {args.use_mask} Optim: {args.optim} LR: {args.lr} -> Is fov constant? : {args.constant_fov} -> Epoch : {epoch} -> Coeff : {args.coeff_count} -> Out path : {out_dir} -> Type of Loss : {args.loss} -> Training set results: Recon Loss: {cur_rloss}')
    if args.save_model:
        # load parent's model before saving
        if args.parent_model_dir and args.parent_type == 'finetuned' and args.load_parent:
            parent_state_dict = torch.load(args.parent_model_dir, map_location=device)
            fc_camera_params = {
                'weight': parent_state_dict['1.fc_camera_params.weight'],
                'bias': parent_state_dict['1.fc_camera_params.bias'],
            }
            fc_shade = {
                'weight': parent_state_dict['1.fc_shade.weight'],
                'bias': parent_state_dict['1.fc_shade.bias'],
            }
            model_branched[1].fc_camera_params.load_state_dict(fc_camera_params)
            model_branched[1].fc_shade.load_state_dict(fc_shade)
            model_branched.eval()
            args['constant_pose'] = False # false during evaluation
            with torch.no_grad():
                for bix, data in enumerate(syn_train_dl):
                    face = data
                    face   = face.to(device)
                    out_recon, masks = model_branched(face)
                    save_image(face, f'{out_dir}/face_{bix}_eval.png')
                    save_image(out_recon, f'{out_dir}/recon_face_{bix}_eval.png')
                    # save_image((face * masks), f'{out_dir}/masked_recon_face_{bix}_eval.png')
                    if bix > 2: break
        torch.save(model_branched.state_dict(), f'{out_dir}/model.pth')
        with open(f'{out_dir}/output.pickle', "wb") as output_file:
            output = {
                'loss': loss_dict
            }
            pickle.dump(output, output_file)
    tl('done')






