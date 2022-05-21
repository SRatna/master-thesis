import glob
from pathlib import Path

# models_path = '../../camera-all-semi-supervised-bfm/outs/output*/*/*/supervised/*/*/*.pth'
models_path = '../../bfm-data/bfm-code-new/outs/output*/*/*.pth'
# models_path = '../camera-all-semi-supervised-bfm/'
print(models_path)
for model_path in sorted(glob.glob(models_path)):
    # print(model_path)
    path = '/'.join(model_path.split('/')[2:-1])
    output_path = f'../predictions/{path}'
    output_path_2d = f'{output_path}/2D'
    if not Path(output_path).exists():
        print(output_path_2d)
        print()
    # break
    # model_name, model_type = model_path.split('/')[-3: -1]
    # print(model_type.split('-')[3])