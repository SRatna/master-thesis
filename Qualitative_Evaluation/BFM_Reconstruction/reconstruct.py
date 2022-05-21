import torch
import yaml
import torch.nn as nn
from model import NewBranch
from torch.utils.data import DataLoader
from datasets import NowDataset
from utils import save_image, BFM2017MeshRenderer, dotdict, save_mesh
import resnet50_ft_dims_2048 as resnet50_model
from pathlib import Path
import glob

config_file = open('config.yml', 'r')
args = yaml.load(config_file, Loader=yaml.FullLoader)
args = dotdict(args)

# full_dataset = CelebADataset('../images-yolact/')
val_file = '../data/imagepathsvalidation.txt'
root_path = '../data/final_release_version'
full_dataset = NowDataset(root_path, val_file)
test_dl = DataLoader(full_dataset, batch_size=args.batch_size,
                     shuffle=False, num_workers=8)
print(len(test_dl))

bfm_model = BFM2017MeshRenderer(args)

models_path = '../../bfm-data/code-all-params/outs/output*/*/model.pth'


for model_path in sorted(glob.glob(models_path)):
    print(model_path)
    model_name, model_type = model_path.split('/')[-3: -1]
    output_path = f'../predictions/{model_name}/{model_type}'
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
