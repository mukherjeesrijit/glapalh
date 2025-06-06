import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import cv2
from model import get_layer_output, GLAPALH
from dataloader import ScanDataset2D
import torch
import torch.nn.functional as Fu
from tqdm import tqdm

def generate_cam_G(model, input_tensor, target_class, maps_G):
    original_size = input_tensor.shape[1:]
    fc_weights = model.fc.weight[target_class][:512]
    
    cam = torch.zeros(maps_G.shape[2:], dtype=torch.float32).to(maps_G.device)
    for i, w in enumerate(fc_weights):
        #print(i)
        cam += w * maps_G[0, i, :, :]
    
    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= cam.max()

    cam = cam.unsqueeze(0).unsqueeze(0)
    cam = Fu.interpolate(cam, size=original_size, mode='bilinear', align_corners=False)
    return cam.squeeze(0).squeeze(0).cpu().numpy()

def generate_cam_C(model, input_tensor, target_class, maps_C):
    original_size = input_tensor.shape[1:]

    fc_weights = model.fc.weight[target_class][512:576]
    
    cam = torch.zeros(maps_C.shape[2:], dtype=torch.float32).to(maps_C.device)
    for i, w in enumerate(fc_weights):
        cam += w * maps_C[0, i, :, :]
    
    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= cam.max()

    cam = cam.unsqueeze(0).unsqueeze(0)
    cam = Fu.interpolate(cam, size=original_size, mode='bilinear', align_corners=False)
    return cam.squeeze(0).squeeze(0).cpu().numpy()

def generate_cam_B(model, input_tensor, target_class, maps_B):
    original_size = input_tensor.shape[1:]
    fc_weights = model.fc.weight[target_class][576:640]
    
    cam = torch.zeros(maps_B.shape[2:], dtype=torch.float32).to(maps_B.device)
    for i, w in enumerate(fc_weights):
        cam += w * maps_B[0, i, :, :]
    
    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= cam.max()

    cam = cam.unsqueeze(0).unsqueeze(0)
    cam = Fu.interpolate(cam, size=original_size, mode='bilinear', align_corners=False)
    return cam.squeeze(0).squeeze(0).cpu().numpy()

def normalize_cam(cam):
    """Normalize the CAM to be between -1 and 1 with non-negative values."""
    cam -= cam.min()
    cam /= cam.max()
    return cam

def generate_cams(model, images, predicted_label, F_L_C1, F_L_BT1, maps_G):
    cam_C = generate_cam_C(model, images, predicted_label, F_L_C1)
    cam_B = generate_cam_B(model, images, predicted_label, F_L_BT1)
    cam_G = generate_cam_G(model, images, predicted_label, maps_G)
    return cam_C, cam_B, cam_G

def display_cams(image, cam_L, cam_G, cam_combined, mask, label):
    fig, axs = plt.subplots(1, 4, figsize=(20, 12))

    mask = (mask > 0).astype(float)

    # Display original image
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title(f"Original Image_{label}")
    axs[0].axis('off')

    # Display CAM_L with binary mask
    axs[1].imshow(image, cmap='gray')
    cam_L_img = axs[1].imshow(normalize_cam(cam_L*mask), cmap='jet', alpha=0.5)  # Apply binary mask
    axs[1].set_title("CAM_L (Local) with Mask")
    axs[1].axis('off')

    # Display CAM_G with binary mask
    axs[2].imshow(image, cmap='gray')
    cam_G_img = axs[2].imshow(normalize_cam(cam_G*mask), cmap='jet', alpha=0.5)  # Apply binary mask
    axs[2].set_title("CAM_G (Global) with Mask")
    axs[2].axis('off')

    # Display Combined CAM with binary mask
    axs[3].imshow(image, cmap='gray')
    cam_combined_img = axs[3].imshow(normalize_cam(cam_combined*mask), cmap='jet', alpha=0.5)  # Apply binary mask
    axs[3].set_title("Combined CAM with Mask")
    axs[3].axis('off')

    # Add colorbars for the CAMs
    for i, cam_img in enumerate([cam_L_img, cam_G_img, cam_combined_img]):
        cbar = plt.colorbar(cam_img, ax=axs[i+1], orientation='vertical', fraction=0.03, pad=0.03)
        cbar.set_label('Activation Intensity')

    plt.tight_layout()
    plt.show()

def viz_cam(input_patient_id, slice_num, model_path, cloud_file, splits_file, batch_size=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # List to store the features and labels for t-SNE
    features = []
    labels_list = []
    patient_ids_list = []
    slice_ids_list = []
                    
    # Initialize the model
    model = GLAPALH(glob=0.01, loc=0.02, parts=0.01).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load validation dataset
    val_dataset = ScanDataset2D(cloud_file, splits_file, is_training=False, augment = False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Hook for extracting intermediate outputs
    hook_outputs = []
    def hook_fn(module, input, output):
        hook_outputs.append(output)

    deep_layers = list(model.global_branch_deep.children())
    hook_layer = deep_layers[-2][1].conv2  # Second last layer
    hook_layer.register_forward_hook(hook_fn)
            
    # Forward pass
    with torch.no_grad():
        for images, masks, patient_ids, slice_idxs, labels in tqdm(val_loader, desc="Evaluating"):
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            outputs, loss = model(images, masks, labels)
            
            hook_outputs = []

            for i in range(len(patient_ids)):
                image = images[i].cpu().numpy().squeeze(0)
                mask = masks[i].cpu().numpy().squeeze(0)
                patient_id = patient_ids[i]
                label = labels[i]
                prob = torch.softmax(outputs[i], dim=0)
                predicted_label = torch.argmax(prob).item()
                slice_id = slice_idxs[i].item()

                if patient_id == input_patient_id and slice_id >= slice_num:
                    print(patient_id)
                    print(slice_num)

                    # Extract layer outputs
                    layer_outputs = get_layer_output(model.unet, images)
                    F_C = layer_outputs['inconv']
                    F_S = layer_outputs['upconv1']

                    # Global Branch
                    F_G1 = model.global_branch_shallow(F_C)
                    F_G = model.global_branch_deep(F_G1)    
                    F_G = F_G.view(F_G.size(0), -1)
                    
                    maps_G = hook_outputs[0]  # Extracted feature map

                    # Local Branch
                    F_L1 = model.local_branch(F_C)
                    W_L_logits = model.mask_conv(F_S)
                    W_L = torch.softmax(W_L_logits, dim=1)
                    W_C_L, W_BT_L = W_L[:, 0, :, :], W_L[:, 1, :, :]
                    W_C_L = W_C_L.unsqueeze(1)
                    W_BT_L = W_BT_L.unsqueeze(1)

                    W_C_L = Fu.interpolate(W_C_L, size=F_L1.shape[2:], mode='nearest')
                    W_BT_L = Fu.interpolate(W_BT_L, size=F_L1.shape[2:], mode='nearest')

                    F_L_C1 = F_L1 * W_C_L
                    F_L_BT1 = F_L1 * W_BT_L

                    F_L_C = Fu.avg_pool2d(F_L_C1, kernel_size=F_L_C1.size()[2:]).view(F_L_C1.size(0), -1)
                    F_L_BT = Fu.avg_pool2d(F_L_BT1, kernel_size=F_L_BT1.size()[2:]).view(F_L_BT1.size(0), -1)

                    F_L = torch.cat((F_L_C, F_L_BT), dim=1)
                    F = torch.cat((F_G, F_L), dim=1)
                    out = model.fc(F)

                    # Example shapes
                    num_global_features = F_G.size(1)  # Number of channels in F_G
                    num_local_features = F_L_C.size(1)  # Number of channels in F_L_C and F_L_BT

                    feature_for_tsne = F.cpu().numpy().flatten() 
                    # Append features and corresponding labels/slice info
                    features.append(feature_for_tsne)
                    labels_list.append(labels[i].item())
                    patient_ids_list.append(patient_ids[i])
                    slice_ids_list.append(slice_idxs[i].item())

                    cam_C, cam_B, cam_G = generate_cams(model, images[i], predicted_label, F_L_C1, F_L_BT1, maps_G)
                    cam_L = cam_B + cam_C
                    cam = cam_C + cam_B + cam_G

                    display_cams(image, cam_L, cam_G, cam, mask, label)

# output_model_path = rf"glapalh/model.pth"
# splits_file = rf"glapalh/splits.csv"
# cloud_file = rf"glapalh/cloud.csv" 
# patient_id = "patientid3"
# slice_num = 5 
# viz_cam(patient_id, slice_num, output_model_path, cloud_file, splits_file, batch_size=1)
