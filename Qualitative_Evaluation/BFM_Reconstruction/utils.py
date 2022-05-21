import os
import shutil
import h5py
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesVertex,
    BlendParams,
    PointLights
)
from pytorch3d.renderer.utils import convert_to_tensors_and_broadcast
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from sh_lights import SphericalHarmonicsLights
import numpy as np
from pytorch3d.io import save_obj

class dotdict(dict):
    __getattr__ = dict.get

def get_image_grid(pic):
    grid = torchvision.utils.make_grid(pic, nrow=8, padding=2)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return ndarr

def denormalize(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def save_image(pic, path=None, denorm=False):
    if denorm:
        pic = denormalize(pic)
    ndarr = get_image_grid(pic)    
    
    if path == None:
        plt.imshow(ndarr)
        plt.show()
    else:
        im = Image.fromarray(ndarr)
        im.save(path)
        
def save_landmarks(vertices, path):
    vertices = vertices.cpu().numpy()
    detections3d_7 = np.array([
        vertices[2736], # right eye vertex
        vertices[6219],
        vertices[9892],
        vertices[13360],
        vertices[8168],
        vertices[5779],
        vertices[10598],
    ]).astype(np.float32)
    np.save(path, detections3d_7)

def save_mesh(path, mesh, val_path):
    # batch of mesh and val_path
    # also copy landmarks
    for v, f, p in zip(mesh.verts_list(), mesh.faces_list(), val_path):
        out_path = f'{path}/{os.path.dirname(p)}'
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        mesh_path = f'{path}/{p}'.replace('.jpg', '.obj')
        landmarks_path = f'{path}/{p}'.replace('.jpg', '.npy')
        save_landmarks(v, landmarks_path)
        save_obj(mesh_path, v, f)
    print('saved: ', path)

class BFM2017MeshRenderer:
    raster_settings = RasterizationSettings(
        image_size=224,
        blur_radius=0.0,  # no blur
        bin_size=0,
    )

    def __init__(self, args):
        print("Loading BFM 2017 into GPU... (this can take a while)")
        self.args = args
        self.file = h5py.File(self.args.bfm_dir, 'r')
        # This can take a few seconds
        self.faces = torch.Tensor(np.array(self.file["shape"]["representer"]["cells"]).T).unsqueeze(0).to(self.args.device).float()
        self.shape_mu = torch.Tensor(self.file["shape"]["model"]["mean"]).reshape(
            1, self.args.n_vertices*3, 1).to(self.args.device).float()
        self.shape_pca_basis = torch.Tensor(self.file["shape"]["model"]["pcaBasis"][:, :self.args.coeff_count]).reshape(
            1, self.args.n_vertices*3, self.args.coeff_count).to(self.args.device).float()
        self.shape_pca_std = torch.Tensor(self.file["shape"]["model"]["pcaVariance"][:self.args.coeff_count]).reshape(
            1, self.args.coeff_count, 1).to(self.args.device).float().sqrt()
        self.color_mu = torch.Tensor(self.file["color"]["model"]["mean"]).reshape(
            1, self.args.n_vertices*3, 1).to(self.args.device).float()
        self.color_pca_basis = torch.Tensor(self.file["color"]["model"]["pcaBasis"][:, :self.args.coeff_count]).reshape(
            1, self.args.n_vertices*3, self.args.coeff_count).to(self.args.device).float()
        self.color_pca_std = torch.Tensor(self.file["color"]["model"]["pcaVariance"][:self.args.coeff_count]).reshape(
            1, self.args.coeff_count, 1).to(self.args.device).float().sqrt()
        self.expression_mu = torch.Tensor(self.file["expression"]["model"]["mean"]).reshape(
            1, self.args.n_vertices*3, 1).to(self.args.device).float()
        self.expression_pca_basis = torch.Tensor(self.file["expression"]["model"]["pcaBasis"][:, :self.args.exp_count]).reshape(
            1, self.args.n_vertices*3, self.args.exp_count).to(self.args.device).float()
        self.expression_pca_std = torch.Tensor(self.file["expression"]["model"]["pcaVariance"][:self.args.exp_count]).reshape(
            1, self.args.exp_count, 1).to(self.args.device).float().sqrt()

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=None,
                raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(
                device=self.args.device,
                cameras=None,
                lights=None,
                blend_params=BlendParams(background_color=(0,0,0))
            )
        )
        print("Done")

    def __delete__(self, instance):
        self.file.close()

    def reconstruct_face(self, shape_param, color_param, exp_param, camera_param, shade_param):
        shape_param = shape_param.unsqueeze(2)
        color_param = color_param.unsqueeze(2)
        exp_param = exp_param.unsqueeze(2)

        shape_param = shape_param * self.shape_pca_std
        color_param = color_param * self.color_pca_std
        exp_param = exp_param * self.expression_pca_std

        if self.args.exp_off:
            vertices = self.shape_mu + \
                self.shape_pca_basis @ shape_param
        else:
            vertices = self.shape_mu + \
                self.shape_pca_basis @ shape_param + \
                self.expression_pca_basis @ exp_param

        colors = self.color_mu + \
            self.color_pca_basis @ color_param 
        
        vertices = vertices.reshape(-1, self.args.n_vertices, 3).float() # batch, no of vertices, 3 dim
        colors = colors.reshape(-1, self.args.n_vertices, 3).float() # batch, no of vertices, 3 dim
        vertices, faces, colors = convert_to_tensors_and_broadcast(vertices, self.faces, colors, device=self.args.device)
        meshes = Meshes(vertices, faces, TexturesVertex(colors))
        if self.args.constant_shade:
            location = ((0,0,1000),)
            lights = PointLights(ambient_color=((0.3, 0.3, 0.3),), diffuse_color=((0.7, 0.7, 0.7),), specular_color=((0, 0, 0),), device=self.args.device, location=location)
        else:
            lights = SphericalHarmonicsLights(device=self.args.device, sh_params=shade_param.reshape(-1, 9, 3))
        if self.args.constant_pose:
            elev = 0
            azim = 0
        else:
            elev = camera_param[:, 0]
            azim = camera_param[:, 1] * -1
        R, T = look_at_view_transform(self.args.cam_distance, elev, azim, degrees=False)
        if not self.args.constant_pp:
            batch_size = camera_param.shape[0]
            principal_point = camera_param[:, 2:4] * 112
            camera_distance = torch.ones((batch_size, 1)).to(self.args.device) * self.args.cam_distance # default z translation
            T = torch.cat((principal_point, camera_distance), dim=1)
            T[:,0] = T[:,0] * -1
        if self.args.constant_fov:
            fov = self.args.fov
        else:
            fov = camera_param[:, 4]
        cameras = FoVPerspectiveCameras(device=self.args.device, R=R, T=T, znear=10.0, zfar=1000000.0, fov=fov, degrees=self.args.constant_fov)
        imgs = self.renderer(meshes, cameras=cameras, lights=lights)
        # masks = torch.where(imgs == 0, 0., 1.)[..., :3].permute((0, 3, 1, 2))
        return imgs[..., :3].permute((0, 3, 1, 2)), meshes


