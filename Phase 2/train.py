import torch
import pickle
import argparse
import json
from utils import *
import torch.nn as nn
import resnet50_ft_dims_2048 as resnet50_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--coeff-count', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--step-size', type=int, default=4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gamma', type=float, default=0.4)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--finetune', type=str, default='no')
    parser.add_argument('--freeze-vgg', type=str, default='yes')

    args = parser.parse_args()

    device = torch.device(args.device)
    print("Using device {}.".format(device))
    print(args)
    torch.manual_seed(args.seed)

    dataloaders, dataset_sizes = get_data_loaders(
        args.batch_size, args.data_dir, args.coeff_count)

    path = args.data_dir.split("/")[-1]

    if args.finetune == 'yes':
        model_ft = resnet50_model.resnet50_ft()
    else:
        model_ft = resnet50_model.resnet50_ft(
            weights_path='resnet50_ft_dims_2048.pth')
        if args.freeze_vgg == 'yes':
            for param in model_ft.parameters():
                param.require_grad = False
        else: path += 'not-freezed-vgg'
    num_ftrs = model_ft.classifier.in_channels
    # replace last layer
    # input_size_branched_layers = 256
    # model_ft.classifier = nn.Conv2d(
    #     num_ftrs, input_size_branched_layers, kernel_size=[1, 1], stride=(1, 1))
    model_branched = nn.Sequential(model_ft, NewBranch(
        num_ftrs, args.coeff_count))
    if args.finetune == 'yes':
        state_dict = torch.load(path+'-model.pth')
        model_branched.load_state_dict(state_dict)
        path += '-ft'
    model_branched = model_branched.to(device)
    model, loss = train(model_branched, dataloaders, dataset_sizes, args)
    path += f'-lr-{args.lr}'
    print(path)
    torch.save(model.state_dict(), f'{path}-model.pth')
    save_loss_fig(loss, path)
    test_loss = print_test_loss(model, args.data_dir, args.coeff_count, args.device)
    # save loss as pickle
    with open(f'{path}-loss.pickle', "wb") as output_file:
        output = {
            'train_val_loss': loss,
            'args': json.dumps(args),
            'test_loss': test_loss
        }
        pickle.dump(output, output_file)
    print('done')


