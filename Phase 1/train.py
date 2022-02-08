import torch
import pickle
import json
from utils import *
from log import *
import torch.nn as nn
import resnet50_ft_dims_2048 as resnet50_model

if __name__ == '__main__':
    args = loadArgs()

    device = torch.device(args.device)
    telegram_logger("Using device {}.".format(device))
    telegram_logger(json.dumps(args))
    torch.manual_seed(args.seed)

    dataloaders, dataset_sizes = get_data_loaders(args)

    if args.concat:
        path = 'all'
    else:
        path = args.data_dirs[0].split("/")[-1]

    if args.finetune:
        model_ft = resnet50_model.resnet50_ft()
    else:
        model_ft = resnet50_model.resnet50_ft(
            weights_path=args.resnet_model_dir)
        if args.freeze_vgg:
            for param in model_ft.parameters():
                param.require_grad = False
        else: path += '-not-freezed-vgg'
    num_ftrs = model_ft.classifier.in_channels
    # replace last layer
    model_branched = nn.Sequential(model_ft, NewBranch(
        num_ftrs, args.coeff_count))
    if args.finetune:
        state_dict = torch.load(args.pretrained_model_dir)
        model_branched.load_state_dict(state_dict)
        path += '-ft-' + args.pretrained_model_dir.replace('.pth', '')
    model_branched = model_branched.to(device)
    model, loss = train(model_branched, dataloaders, dataset_sizes, args)
    telegram_logger(path)
    torch.save(model.state_dict(), f'{path}.pth')
    save_loss_fig(loss, path)
    # save loss as pickle
    test_loss = print_test_loss(model, args)
    with open(f'{path}-output.pickle', "wb") as output_file:
        output = {
            'train_val_loss': loss,
            'args': args,
            'test_loss': test_loss
        }
        pickle.dump(output, output_file)
    telegram_logger('done')


