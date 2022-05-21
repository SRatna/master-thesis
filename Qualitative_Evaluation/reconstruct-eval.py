import torch
import yaml
import torch.nn as nn
from bfm_reconstruction.model import NewBranch
from torch.utils.data import DataLoader
from bfm_reconstruction.datasets import NowDataset
from bfm_reconstruction.utils import save_image, BFM2017MeshRenderer, dotdict, save_mesh
import resnet50_ft_dims_2048 as resnet50_model
from pathlib import Path
import glob
from log import telegram_logger
from compute_error import metric_computation
from cumulative_errors import generating_cumulative_error_plots

config_file = open('bfm_reconstruction/config.yml', 'r')
args = yaml.load(config_file, Loader=yaml.FullLoader)
args = dotdict(args)

# full_dataset = CelebADataset('../images-yolact/')
val_file = './data/imagepathsvalidation.txt'
root_path = './data/final_release_version'
full_dataset = NowDataset(root_path, val_file)
test_dl = DataLoader(full_dataset, batch_size=args.batch_size,
                     shuffle=False, num_workers=8)
print(len(test_dl))

args['bfm_dir'] = '../BFM-generator/data/bfm2017/model2017-1_bfm_nomouth.h5'
bfm_model = BFM2017MeshRenderer(args)

models_path = '../camera-all-semi-supervised-bfm/outs/output*/*/*/supervised/*/*-landmarks-fixed-*/*.pth'
# models_path = '../camera-all-semi-supervised-bfm/outs/output-5k-40-80pc-20e-landmark-background_output-40k-5-80pc-20e-landmark-background/seperate-all-losses-80-pc-20-exp-0.00001-lr-0.25-train-mse-loss-5-cam-params/ffhq-all/supervised/cosine_1/adam-1e-05-ep-6-pllr-1e-06-resnet-frozen-cfov-cpp-cam-params-frozen-*/*.pth'
# models_path = '../bfm-data/code-all-params/outs/output-*/*/*.pth'
# models_path = '../bfm-data/code-all-params/outs/output-5k-40-80pc-20e-landmark-background_output-40k-5-80pc-20e-landmark-background/*/*.pth'
for model_path in sorted(glob.glob(models_path)):
    print(model_path)
    if '-normalized-' in model_path:
        continue
    if '2-cam-params' in model_path:
        args['cam_params'] = 2
    else:
        args['cam_params'] = 5
    path = '/'.join(model_path.split('/')[1:-1])
    output_path = f'./predictions/{path}'
    output_path_2d = f'{output_path}/2D'
    if Path(f'{output_path_2d}/face0.png').exists():
        print('already done')
        continue
    Path(output_path_2d).mkdir(parents=True, exist_ok=True)
    output_path_3d = f'{output_path}/3D'
    Path(output_path_3d).mkdir(parents=True, exist_ok=True)
    # load model
    model_ft = resnet50_model.resnet50_ft()
    num_ftrs = model_ft.classifier.in_channels
    model = nn.Sequential(model_ft, NewBranch(
        num_ftrs, args.coeff_count, args.exp_count, bfm_model, args.cam_params))
    state_dict = torch.load(model_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    # start prediction
    model.eval()
    with torch.no_grad():
        for bix, data in enumerate(test_dl):
            face, val_path = data
            out_recon, out_mesh = model(face.to(args.device))
    #         save_image(face)
    #         save_image(out_recon)
            save_image(face, f'{output_path_2d}/face{bix}.png')
            save_image(out_recon, f'{output_path_2d}/face{bix}_r.png')
            save_mesh(output_path_3d, out_mesh, val_path)
    #         if bix == 1:
    #             break
    # break

    # start evaluation
    dataset_folder = './data/'
    telegram_logger(f'{output_path_3d} started to compute mesh loss')
    # valid_challenges = ['', 'multiview_neutral', 'multiview_expressions', 'multiview_occlusions', 'selfie']
    valid_challenges = ['']
    for challenge in valid_challenges:
        metric_computation(dataset_folder, output_path_3d, challenge=challenge)

    # print cumulative error plots
    base_path = f'{output_path_3d}/results/'
    # List of method identifiers, used as method name within the polot
    method_identifiers = []
    # List of paths to the error files (must be of same order than the method identifiers)
    method_error_fnames = []
    # File name of the output error plot
    out_fname = f'{base_path}semi-supervised-cuke-errors'
    # valid_challenges = ['', 'multiview_neutral', 'multiview_expressions', 'multiview_occlusions', 'selfie']
    valid_challenges = ['']
    for challenge in valid_challenges:
        method_identifiers.append(f'semi-supervised-cuke-{challenge}')
        method_error_fnames.append(f'{base_path}_computed_distances_{challenge}.npy')
    generating_cumulative_error_plots(method_error_fnames, method_identifiers, out_fname)
