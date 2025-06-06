import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader import ScanDataset2D
from model import GLAPALH
import torch.optim as optim
from collections import defaultdict
import copy

def solve_model(cloud_file, splits_file, output_model_path, num_epochs=10, batch_size=32, patience=2):
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    train_dataset = ScanDataset2D(cloud_file, splits_file, is_training=True, augment = True)
    val_dataset = ScanDataset2D(cloud_file, splits_file, is_training=False, augment = False)
    
    print(f"Training patients: {train_dataset}")
    print(f"Validation patients: {val_dataset}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    model = GLAPALH(glob=0.01, loc=0.02, parts=0.01).to(device)

    # DiceLoss is suitable for segmentation tasks
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)

    best_val_loss = float('inf')  # Initialize best validation loss
    epochs_without_improvement = 0  # Counter for early stopping
    best_model_weights = None

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total_train_samples = 0
        for images, masks, patient_ids, slice_idxs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device) 
            masks = masks.to(device)
            labels = labels.to(device) # Ensure labels are on the correct device
            optimizer.zero_grad()
            outputs, loss = model(images, masks, labels) 
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            total_train_samples += images.size(0)

        # Validation loop
        model.eval()
        val_loss = 0.0
        total_val_samples = 0
        with torch.no_grad():
            for images, masks, patient_ids, slice_idxs, labels in val_loader:
                images = images.to(device)  # Use .to(device) instead of .cuda()
                masks = masks.to(device)  # Use .to(device) instead of .cuda()
                labels = labels.to(device)  # Ensure labels are on the correct device
                outputs, loss = model(images, masks, labels)
                val_loss += loss.item() * images.size(0)
                total_val_samples += images.size(0)

        # Average the loss for the epoch
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Length: {total_val_samples:.4f}")

        train_loss /= total_train_samples
        val_loss /= total_val_samples
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss > 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())  # Safely copy weights
            epochs_without_improvement = 0

            # Save immediately when new best model is found
            torch.save(best_model_weights, output_model_path)
            print(f"🔒 New best model saved at epoch {epoch+1} with Val Loss: {best_val_loss:.4f}")

        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("⏹️ Early stopping triggered!")
                break

    if best_model_weights is not None:
        print(f"Best model saved to {output_model_path}")
        print(f"Best Val Loss: {best_val_loss:.4f}")

def evaluate_model(model_path, cloud_file, splits_file, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    # Load validation dataset
    val_dataset = ScanDataset2D(cloud_file, splits_file, is_training=False, augment = False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load trained model
    model = GLAPALH(glob=0.01, loc=0.02, parts=0.01).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Initialize metrics
    running_loss = 0.0
    total_samples = 0
    patientwise_data = defaultdict(lambda: {"probs": [], "labels": [], "slice_ids": []})

    with torch.no_grad():
        for images, masks, patient_ids, slice_idxs, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            outputs, loss = model(images, masks, labels)
            probs = F.softmax(outputs, dim=1).cpu()

            running_loss += loss.item() * images.size(0) #* batch_size 
            total_samples += images.size(0) # batch_size

            for i in range(images.size(0)):
                pid = patient_ids[i]
                prob = probs[i]
                label = labels[i].cpu().item()
                slice_id = slice_idxs[i]

                patientwise_data[pid]["probs"].append(prob)
                patientwise_data[pid]["labels"].append(label)
                patientwise_data[pid]["slice_ids"].append(slice_id)

    print(f"Val Loss: {running_loss:.4f}")
    print(f"Length: {total_samples:.4f}")

    avg_loss = running_loss / total_samples
    correct_predictions = 0
    total_patients = len(patientwise_data)

    for pid, data in patientwise_data.items():
        slice_ids = data["slice_ids"]
        probs = data["probs"]
        true_label = data["labels"][0]

        # Sort by slice index and select middle 50%
        sorted_slices = sorted(zip(slice_ids, probs), key=lambda x: x[0])
        num_slices = len(sorted_slices)
        start = num_slices // 4
        end = start + (num_slices // 2)
        middle_probs = [p for _, p in sorted_slices[start:end]]

        # Aggregate
        mean_prob = torch.stack(middle_probs).mean(dim=0)
        predicted_label = mean_prob.argmax().item()

        if predicted_label == true_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_patients if total_patients > 0 else 0.0

    print(f"\nEvaluation Summary:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Patient-wise Accuracy: {accuracy * 100:.2f}%")

    return avg_loss, accuracy

# output_model_path = rf"glapalh/model.pth"
# splits_file = rf"glapalh/splits.csv"
# cloud_file = rf"glapalh/cloud.csv" 
# solve_model(cloud_file, splits_file, output_model_path, num_epochs=10, batch_size=32, patience=5)
# print(evaluate_model(output_model_path, cloud_file, splits_file, batch_size=32))