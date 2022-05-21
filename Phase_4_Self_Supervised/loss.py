import torch
import torchvision.models
import resnet50_ft_dims_2048 as resnet50_model
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class VGG19(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(
            pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


# Perceptual loss that uses a pretrained VGG network
class ImageNetVGG19Loss(torch.nn.Module):
    def __init__(self):
        super(ImageNetVGG19Loss, self).__init__()
        self.vgg = VGG19()
        # if gpu_ids:
        #     self.vgg.cuda()
        self.criterion = torch.nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, fake, real):
        x_vgg, y_vgg = self.vgg(fake), self.vgg(real)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * \
                self.criterion(x_vgg[i], y_vgg[i].detach())
            # loss = self.criterion(x_vgg[i], y_vgg[i].detach())
            # print(list(x_vgg[i].shape), loss.data, (self.weights[i] * loss).data, self.weights[i])
        return loss

class ResNet50(nn.Module):
    def __init__(self, model_path):
        super(ResNet50, self).__init__()
        self.resnet = resnet50_model.resnet50_ft(weights_path=model_path)

    def forward(self, x):
        _, x = self.resnet(x)
        x = x.squeeze(2)
        x = x.squeeze(2)
        return x

class VGGFaceResNet50Loss(torch.nn.Module):
    def __init__(self, model_path):
        super(VGGFaceResNet50Loss, self).__init__()
        self.resnet = ResNet50(model_path)
        self.criterion = torch.nn.L1Loss()

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, fake, real):
        x_resnet, y_resnet = self.resnet(fake), self.resnet(real)
        loss = self.criterion(x_resnet, y_resnet.detach())
        return loss

class PerceptualLoss(torch.nn.Module):
    def __init__(self, model_path):
        super(PerceptualLoss, self).__init__()
        self.resnet = ResNet50(model_path)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, fake, real):
        # fake = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(fake)
        id_featureA = F.normalize(self.resnet(fake), dim=-1, p=2)
        id_featureB = F.normalize(self.resnet(real), dim=-1, p=2)  
        cosine_d = torch.sum(id_featureA * id_featureB, dim=-1)
        # assert torch.sum((cosine_d > 1).float()) == 0
        return torch.sum(1 - cosine_d) / cosine_d.shape[0]  

### image level loss
def photo_loss(imageA, imageB, mask, eps=1e-6):
    """
    l2 norm (with sqrt, to ensure backward stabililty, use eps, otherwise Nan may occur)
    Parameters:
        imageA       --torch.tensor (B, 3, H, W), range (0, 1), RGB order 
        imageB       --same as imageA
    """
    loss = torch.sqrt(eps + torch.sum((imageA - imageB) ** 2, dim=1, keepdims=True)) * mask
    loss = torch.sum(loss) / torch.max(torch.sum(mask), torch.tensor(1.0).to(mask.device))
    return loss