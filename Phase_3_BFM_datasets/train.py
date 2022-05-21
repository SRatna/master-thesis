import torch
import pickle
import argparse
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
    parser.add_argument('--exp-count', type=int, default=10)
    parser.add_argument('--cam-params', type=int, default=5)
    parser.add_argument('--valid-size', type=float, default=0.10)
    parser.add_argument('--train-size', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--step-size', type=int, default=4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.4)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--out-dir', type=str, default='.')
    parser.add_argument('--save-model', type=str, default='yes')
    parser.add_argument('--finetune', type=str, default='no')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--loss', type=str, default='mse')

    args = parser.parse_args()
    print(args)
    device = torch.device(args.device)
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    dataloaders, dataset_sizes = get_data_loaders(args)

    if args.finetune == 'yes':
        model_ft = resnet50_model.resnet50_ft()
    else:
        model_ft = resnet50_model.resnet50_ft(
            weights_path='resnet50_ft_dims_2048.pth')
        for param in model_ft.parameters():
            param.require_grad = False
    num_ftrs = model_ft.classifier.in_channels
    input_size_branched_layers = num_ftrs
    model_branched = nn.Sequential(model_ft, NewBranch(
        input_size_branched_layers, args))
    if args.finetune == 'yes':
        state_dict = torch.load(args.out_dir.replace('-ft', '') + '/model.pth')
        model_branched.load_state_dict(state_dict)
    model_branched = model_branched.to(device)

    model, loss = train(model_branched, dataloaders, dataset_sizes, args)
    path = f'{args.out_dir}/'
    if args.save_model == 'yes':
        torch.save(model.state_dict(), f'{args.out_dir}/model.pth')
    save_loss_fig(loss, args.out_dir)
    test_loss = print_test_loss(model, args)
    # save loss as pickle
    with open(f'{args.out_dir}/output.pickle', "wb") as output_file:
        output = {
            'train_val_loss': loss,
            'args': args,
            'test_loss': test_loss
        }
        pickle.dump(output, output_file)
    print('done')
