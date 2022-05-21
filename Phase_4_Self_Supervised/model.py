import torch
import torch.nn as nn
import resnet50_ft_dims_2048 as resnet50_model


class NewBranch(nn.Module):
    def __init__(self, input_size, coeff_count, exp_count, bfm_model, cam_params):
        super(NewBranch, self).__init__()
        self.bfm_model = bfm_model
        self.fc_camera_params = nn.Linear(input_size, cam_params)
        self.fc_shape = nn.Linear(input_size, coeff_count)
        self.fc_texture = nn.Linear(input_size, coeff_count)
        self.fc_exp = nn.Linear(input_size, exp_count)
        self.fc_shade = nn.Linear(input_size, 27)

    def forward(self, inputs):
        _, x = inputs
        x = x.squeeze(2)
        x = x.squeeze(2)
        out_camera_params = self.fc_camera_params(torch.relu(x))
        out_shape = self.fc_shape(torch.relu(x))
        out_texture = self.fc_texture(torch.relu(x))
        out_exp = self.fc_exp(torch.relu(x))
        out_shade = self.fc_shade(torch.relu(x))
        faces, masks = self.bfm_model.reconstruct_face(out_shape, out_texture, out_exp, out_camera_params, out_shade)
        return faces, masks

class NewBranchWithoutRenderer(nn.Module):
    def __init__(self, input_size, args):
        super(NewBranchWithoutRenderer, self).__init__()
        self.fc_camera_params = nn.Linear(input_size, args.cam_params)
        self.fc_shape = nn.Linear(input_size, args.coeff_count)
        self.fc_texture = nn.Linear(input_size, args.coeff_count)
        self.fc_exp = nn.Linear(input_size, args.exp_count)
        self.fc_shade = nn.Linear(input_size, 27)

    def forward(self, inputs):
        _, x = inputs
        x = x.squeeze(2)
        x = x.squeeze(2)
        out_camera_params = self.fc_camera_params(torch.relu(x))
        out_shape = self.fc_shape(torch.relu(x))
        out_texture = self.fc_texture(torch.relu(x))
        out_exp = self.fc_exp(torch.relu(x))
        out_shade = self.fc_shade(torch.relu(x))
        return out_shape, out_texture, out_exp, out_camera_params, out_shade

class MixedModels(nn.Module):
    def __init__(self, args, bfm_model):
        super(MixedModels, self).__init__()
        model_face = resnet50_model.resnet50_ft()
        num_ftrs = model_face.classifier.in_channels
        self.model_face = nn.Sequential(model_face, NewBranchWithoutRenderer(num_ftrs, args))
        state_dict = torch.load(args.model_face, map_location=args.device)
        self.model_face.load_state_dict(state_dict)
        self.model_face.to(args.device)

        model_cam_shade = resnet50_model.resnet50_ft()
        num_ftrs = model_cam_shade.classifier.in_channels
        self.model_cam_shade = nn.Sequential(model_cam_shade, NewBranchWithoutRenderer(num_ftrs, args))
        state_dict = torch.load(args.model_cam_shade, map_location=args.device)
        self.model_cam_shade.load_state_dict(state_dict)
        self.model_cam_shade.to(args.device)

        self.bfm_model = bfm_model

    def forward(self, inputs):
        out_shape, out_texture, out_exp, _, _ = self.model_face(inputs)
        _, _, _, out_camera_params, out_shade = self.model_cam_shade(inputs)
        faces, masks = self.bfm_model.reconstruct_face(out_shape, out_texture, out_exp, out_camera_params, out_shade)
        return faces, masks