import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as kr
import torch
import torch.nn as nn
import torch.nn.functional as Fu
import torchvision.models as models

class Double_Convolution(nn.Module):
    def __init__(self, ic, oc):
        super(Double_Convolution, self).__init__()
        self.double_convolution = nn.Sequential(
            nn.Conv2d(ic, oc, 3, 1, 1),
            nn.BatchNorm2d(oc),
            nn.ReLU(inplace=True),
            nn.Conv2d(oc, oc, 3, 1, 1),
            nn.BatchNorm2d(oc),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_convolution(x)
        return x

class Down_Block(nn.Module):
    def __init__(self, ic, oc):
        super(Down_Block, self).__init__()
        self.downsample = nn.MaxPool2d(2)
        self.double_conv = Double_Convolution(ic, oc)

    def pad(self, x):
        _, _, h, w = x.shape
        pad_h = (h % 2 != 0)
        pad_w = (w % 2 != 0)
        if pad_h or pad_w:
            padding = (0, int(pad_w), 0, int(pad_h))  # (left, right, top, bottom)
            x = F.pad(x, padding, mode='constant', value=0)
        return x

    def forward(self, x):
        x = self.pad(x)
        x = self.downsample(x)
        x = self.double_conv(x)
        return x

class Up_Block(nn.Module):
    def __init__(self, ic, oc):
        super(Up_Block, self).__init__()
        self.conv = nn.Conv2d(ic, oc, 2, 1)
        self.double_conv = Double_Convolution(ic, oc)

    def forward(self, x1, x2):
        _, _, h_out, w_out = x2.shape
        x1 = F.interpolate(x1, size=(h_out, w_out), mode='bilinear', align_corners=False)
        padding = (1, 0, 1, 0)  # (left, right, top, bottom)
        x1 = F.pad(x1, padding, mode='constant', value=0)
        x1 = self.conv(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.double_conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, ic, oc):
        super(UNet, self).__init__()
        self.inconv = Double_Convolution(ic, 64)
        self.downconv1 = Down_Block(64, 128)
        self.downconv2 = Down_Block(128, 256)
        self.downconv3 = Down_Block(256, 512)
        self.downconv4 = Down_Block(512, 1024)
        self.upconv4 = Up_Block(1024, 512)
        self.upconv3 = Up_Block(512, 256)
        self.upconv2 = Up_Block(256, 128)
        self.upconv1 = Up_Block(128, 64)
        self.outconv = nn.Conv2d(64, oc, 1, 1)
        self.dice_loss = kr.losses.dice_loss

    def forward(self, x, mask=None):
        x1 = self.inconv(x)
        x2 = self.downconv1(x1)
        x3 = self.downconv2(x2)
        x4 = self.downconv3(x3)
        x5 = self.downconv4(x4)
        x6 = self.upconv4(x5, x4)
        x7 = self.upconv3(x6, x3)
        x8 = self.upconv2(x7, x2)
        x9 = self.upconv1(x8, x1)
        x10 = self.outconv(x9)

        if mask is not None:
            mask = mask.squeeze(dim = 1)
            mask = mask.long()
            loss = self.dice_loss(x10, mask) #kr.losses.focal_loss(x10, mask, alpha = 0.1, reduction= 'mean')
            return x10, loss
        return x10

def get_layer_output(model, input_tensor):

    layer_names = ['inconv', 'upconv1', 'downconv4']

    # Dictionary to store the outputs
    layer_outputs = {}

    # Hook function to capture the output
    def hook(module, input, output):
        layer_outputs[module.name] = output
    
    # Register the hooks
    hooks = []
    for name, module in model.named_modules():
        if name in layer_names:
            module.name = name  # Add a name attribute to the module
            hooks.append(module.register_forward_hook(hook))
    
    # Run the forward pass
    with torch.no_grad():
        _ = model(input_tensor)
    
    # Remove the hooks
    for hook in hooks:
        hook.remove()
    
    return layer_outputs

class GLAPALH(nn.Module):
    def __init__(self, glob = 0.01, loc = 0.02, parts= 0.01):
        super(GLAPALH, self).__init__()
    
        self.global_branch_shallow, self.global_branch_deep = self._init_global_branch()
        self.local_branch = self._init_local_branch()
        self.fc = nn.Linear(512 + 128, 3)
        self.glob = glob  
        self.loc = loc
        self.parts = parts

        self.unet = UNet(ic=1, oc=3)
        
        # incase you have pretrained segmentation model weights, uncomment the following lines
        # pretrained_dict = torch.load('unet.pth')
        # self.unet.load_state_dict(pretrained_dict)
        # for param in self.unet.parameters():
        #     param.requires_grad = False  

        self.mask_conv = nn.Conv2d(64, 2, kernel_size=1) 

    def forward(self, x, mask, label):

        #Unet outputs
        layer_outputs = get_layer_output(self.unet, x)
        F_C = layer_outputs['inconv']
        F_S = layer_outputs['upconv1']

        # Extract masks
        maskc = (mask == 2).float()
        maskbt = (mask == 1).float()

        # Global Branch
        F_G1 = self.global_branch_shallow(F_C)
        F_G = self.global_branch_deep(F_G1)
        F_G = F_G.view(F_G.size(0), -1)
        #F_G = Fu.avg_pool2d(F_G, kernel_size=F_G.size()[2:]).view(F_G.size(0), -1)


        # Local Branch
        F_L1 = self.local_branch(F_C)

        #Creating Weighted Masks Wc and Wbt
        W_L_logits = self.mask_conv(F_S)
        W_L = torch.softmax(W_L_logits, dim=1)
        W_C_L, W_BT_L = W_L[:, 0, :, :], W_L[:, 1, :, :]
        W_C_L = W_C_L.unsqueeze(1)
        W_BT_L = W_BT_L.unsqueeze(1)

        # # Resize masks
        W_C_L = Fu.interpolate(W_C_L, size=F_L1.shape[2:], mode='nearest')
        W_BT_L = Fu.interpolate(W_BT_L, size=F_L1.shape[2:], mode='nearest')

        ## Batch Normalization
        F_L_C = F_L1 * W_C_L
        F_L_BT = F_L1 * W_BT_L

        F_L_C = Fu.avg_pool2d(F_L_C, kernel_size=F_L_C.size()[2:]).view(F_L_C.size(0), -1)
        F_L_BT = Fu.avg_pool2d(F_L_BT, kernel_size=F_L_BT.size()[2:]).view(F_L_BT.size(0), -1)

        # Concatenate the features after batch normalization
        F_L = torch.cat((F_L_C, F_L_BT), dim=1)

        # # Fusion Head
        F = torch.cat((F_G, F_L), dim=1)
        out = self.fc(F)
        
        # BCE Loss
        bce_loss = Fu.cross_entropy(out, label)

        # Regularizers
        global_regularizer = self.global_regularizer(F_G1)
        local_regularizer = self.local_regularizer(F_L1)
        parts_regularizer = self.parts_regularizer(W_C_L, W_BT_L, maskc, maskbt)
        
        # Total Loss
        final_loss = bce_loss + self.glob * global_regularizer + self.loc * local_regularizer + self.parts * parts_regularizer

        return out, final_loss

    def global_regularizer(self, F1_G):
        # Calculate the total number of elements per image in the batch
        N = F1_G.numel()  # Number of elements per batch item
        I = F1_G
        horiz_diff = (I[:, :, :, :-1] - I[:, :, :, 1:]) ** 2
        vert_diff = (I[:, :, :-1, :] - I[:, :, 1:, :]) ** 2
        reg = (horiz_diff[:, :, :-1, :] + vert_diff[:, :, :, :-1]).sum()
        return reg.mean() / N

    def local_regularizer(self, F1_L):

        N_prime = F1_L.numel()  / F1_L.size(0)
        mean_I_prime = torch.mean(F1_L, dim=(2, 3)) 
        squared_diffs = (F1_L - mean_I_prime[:, :, None, None]) ** 2
        sum_squared_diffs = torch.sum(squared_diffs, dim=(1, 2, 3))
        reg_per_image = - (sum_squared_diffs / N_prime)
        avg_reg_per_batch = torch.mean(reg_per_image)
    
        return avg_reg_per_batch 
    
    def parts_regularizer(self, W_C_L, W_BT_L, M_C, M_BT):
        def dice_loss(W, M):
            epsilon = 1e-5
            intersection = 2.0 * (W * M).sum()
            union = W.sum() + M.sum() + epsilon
            loss = 1 - intersection / union
            return loss

        M_C = Fu.interpolate(M_C, size=W_C_L.shape[2:], mode='nearest')
        M_BT = Fu.interpolate(M_BT, size=W_BT_L.shape[2:], mode='nearest')

        L_Dice_C = dice_loss(W_C_L, M_C)
        L_Dice_BT = dice_loss(W_BT_L, M_BT)
        L_Parts = L_Dice_C + L_Dice_BT

        return L_Parts

    def _init_global_branch(self):
        resnet18a = models.resnet18(pretrained = True)
        shallow_layers = list(resnet18a.children())[2:5]
        deep_layers = list(resnet18a.children())[5:-1]
        for param in resnet18a.parameters():
            param.requires_grad = True
        return nn.Sequential(*shallow_layers), nn.Sequential(*deep_layers)
    
    def _init_local_branch(self):
        resnet18b = models.resnet18(pretrained = True)
        shallow_layers = list(resnet18b.children())[2:5]
        for param in resnet18b.parameters():
            param.requires_grad = True
        return nn.Sequential(*shallow_layers)
    