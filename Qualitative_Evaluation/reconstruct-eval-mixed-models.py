import torch
import yaml
from bfm_reconstruction.model import MixedModels
from torch.utils.data import DataLoader
from bfm_reconstruction.datasets import NowDataset, CelebADataset
from bfm_reconstruction.utils import save_image, BFM2017MeshRenderer, dotdict, save_mesh
from pathlib import Path
from log import telegram_logger
from compute_error import metric_computation
from cumulative_errors import generating_cumulative_error_plots

torch.manual_seed(0)
config_file = open('bfm_reconstruction/config.yml', 'r')
args = yaml.load(config_file, Loader=yaml.FullLoader)
args = dotdict(args)

# full_dataset = CelebADataset('../images-yolact/')
if args.type == 'val':
    val_file = './data/imagepathsvalidation.txt'
else:
    val_file = './data/imagepathstest.txt'
root_path = './data/final_release_version'
full_dataset = NowDataset(root_path, val_file)
if args.dataset == 'genova':
    full_dataset = CelebADataset('./data/genova/')
test_dl = DataLoader(full_dataset, batch_size=args.batch_size,
                     shuffle=args.shuffle, num_workers=8)
print(len(test_dl))

# load model
args['bfm_dir'] = '../BFM-generator/data/bfm2017/model2017-1_bfm_nomouth.h5'
bfm_model = BFM2017MeshRenderer(args)
model = MixedModels(args, bfm_model)
model = model.to(args.device)

# create output dirs
model_path = args.model_face
print(model_path)
path = '/'.join(model_path.split('/')[2:-1])
output_path = f'./predictions/mixed-models/{args.dataset}/{path}/round-{args.round}/{args.type}'
output_path_2d = f'{output_path}/2D'
Path(output_path_2d).mkdir(parents=True, exist_ok=True)
output_path_3d = f'{output_path}/3D'
Path(output_path_3d).mkdir(parents=True, exist_ok=True)

# start prediction
model.eval()
with torch.no_grad():
    for bix, data in enumerate(test_dl):
        face, val_path = data
        out_recon, out_mesh = model(face.to(args.device))
        save_image(face, f'{output_path_2d}/face{bix}.png')
        save_image(out_recon, f'{output_path_2d}/face{bix}_r.png')
        if args.save_mesh:
            save_mesh(output_path_3d, out_mesh, val_path)
        # if bix == 10:
        #     break
# break

# start evaluation
if args.type == 'val' and args.eval:
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