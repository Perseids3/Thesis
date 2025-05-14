import os
import time
import logging
import json
import argparse
import re
import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from datetime import datetime

import random




# Set up logging
def setup_logging(output_dir):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_dir = os.path.join(output_dir, f"{timestamp}_res")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log_file_{timestamp}.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ])
    return log_dir

class MRIDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        img_data = nib.load(img_path).get_fdata()
        img_tensor = np.resize(img_data, (160, 256, 256))
        img_tensor = torch.tensor(img_tensor, dtype=torch.float32).unsqueeze(0)
        label = self.labels[idx]
        return img_tensor, label


class Simple3DCNN(nn.Module):
    def __init__(self):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.flattened_size = 64 * 20 * 32 * 32
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(-1, self.flattened_size)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def prepare_data(data_path, modality):
    df = pd.read_csv(data_path, sep='\t')
    df_hc = df[df['group'] == 'hc']
    df_fcd = df[df['group'] == 'fcd']
    id_hc = df_hc['participant_id'].to_list()
    id_fcd = df_fcd['participant_id'].to_list()

    image_label_pair = []
    for sub_id in id_fcd + id_hc:
        anat_dir = os.path.join(sub_id, 'anat')
        anat_dir = os.path.join('data', anat_dir)
        label = 1 if sub_id in id_fcd else 0
        for file in os.listdir(anat_dir):
            if '.gz' in file and 'roi' not in file and modality in file:
                image_label_pair.append((os.path.join(anat_dir, file), label))

    file_paths = [pair[0] for pair in image_label_pair]
    labels = [pair[1] for pair in image_label_pair]
    return file_paths, labels



def train_and_validate(config, file_paths, labels, log_dir):
    best_model = None  
    best_val_loss = float("inf") 
    best_model_path = log_dir
    
    fold_train_accuracies = []
    fold_val_accuracies = []
    fold_train_losses = []
    fold_val_losses = []
    fold_train_recalls = []
    fold_val_recalls = []
    
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        file_paths, labels, test_size=config["test_size"], stratify=labels, random_state=seed # Double
    )

    skf = StratifiedKFold(n_splits=config["k_folds"], shuffle=True, random_state=seed)
    folds = []

    for train_idx, val_idx in skf.split(train_paths, train_labels):
        
        train_files = [train_paths[i] for i in train_idx]
        train_labels_split = [train_labels[i] for i in train_idx]
        val_files = [train_paths[i] for i in val_idx]
        val_labels_split = [train_labels[i] for i in val_idx]
        train_dataset = MRIDataset(train_files, train_labels_split)
        val_dataset = MRIDataset(val_files, val_labels_split)
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
        folds.append((train_loader, val_loader, train_files, val_files))


    for fold_idx, (train_loader, val_loader, train_files, val_files) in enumerate(folds):
        logger.info(f"Starting Fold {fold_idx + 1}")
        train_ids = sorted([re.search(r'sub-(\d+)', path).group(1) for path in train_files if re.search(r'sub-(\d+)', path)])
        val_ids   = sorted([re.search(r'sub-(\d+)', path).group(1) for path in val_files if re.search(r'sub-(\d+)', path)])


        # logger.info(f"Training IDs for Fold {fold_idx + 1}: {train_ids}")
        # logger.info(f"Validation IDs for Fold {fold_idx + 1}: {val_ids}")


        model = Simple3DCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        
        fold_start_time = time.time()
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        train_recalls, val_recalls = [], []

        for epoch in range(config["epochs"]):

            if epoch % 50 == 0 or epoch == 0:
        
                logger.info(f"Starting Epoch {epoch + 1} of Fold {fold_idx + 1}")

            model.train()
            epoch_train_loss = 0.0
            train_predictions, train_labels_actual = [], []
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                train_predictions.extend(predicted.cpu().numpy())
                train_labels_actual.extend(labels.cpu().numpy())
            train_losses.append(epoch_train_loss / len(train_loader))
            train_accuracy = (np.array(train_predictions) == np.array(train_labels_actual)).mean()
            train_recall = torchmetrics.functional.recall(
                torch.tensor(train_predictions), torch.tensor(train_labels_actual), average="macro", task="binary"
            ).item()
            train_accuracies.append(train_accuracy)
            train_recalls.append(train_recall)

            model.eval()
            epoch_val_loss = 0.0
            val_predictions, val_labels_actual = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    epoch_val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    val_predictions.extend(predicted.cpu().numpy())
                    val_labels_actual.extend(labels.cpu().numpy())

            val_losses.append(epoch_val_loss / len(val_loader))
            val_accuracy = (np.array(val_predictions) == np.array(val_labels_actual)).mean()
            val_recall = torchmetrics.functional.recall(
                torch.tensor(val_predictions), torch.tensor(val_labels_actual), average="macro", task="binary"
            ).item()
            val_accuracies.append(val_accuracy)
            val_recalls.append(val_recall)
            
        fold_end_time = time.time()  
        
        fold_model_path = os.path.join(log_dir, f"fold_{fold_idx + 1}_model.pth")
        torch.save(model.state_dict(), fold_model_path)
        logger.info(f"Saved model for Fold {fold_idx + 1} at {fold_model_path}")
        
        if val_losses[-1] < best_val_loss:  # Compare final validation loss for the fold
            best_val_loss = val_losses[-1]
            best_model_path = fold_model_path
            logger.info(f"Updated Best Model: Fold {fold_idx + 1}, Validation Loss: {best_val_loss:.4f}")


        # Generate and save plots for this fold (Loss, Accuracy, and Recall)
        epochs = range(1, len(train_losses) + 1)
        # Loss
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.title(f"Loss vs Epoch (Fold {fold_idx + 1})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(log_dir, f"fold_{fold_idx + 1}_loss.png"))
        plt.close()
   

        # Accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_accuracies, label="Train Accuracy")
        plt.plot(epochs, val_accuracies, label="Validation Accuracy")
        plt.title(f"Accuracy vs Epoch (Fold {fold_idx + 1})")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(log_dir, f"fold_{fold_idx + 1}_accuracy.png"))
        plt.close()
 

        # Recall
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_recalls, label="Train Recall")
        plt.plot(epochs, val_recalls, label="Validation Recall")
        plt.title(f"Recall vs Epoch (Fold {fold_idx + 1})")
        plt.xlabel("Epoch")
        plt.ylabel("Recall")
        plt.legend()
        plt.savefig(os.path.join(log_dir, f"fold_{fold_idx + 1}_recall.png"))
        plt.close()

        logger.info(f"Fold {fold_idx + 1}, Epoch {epoch + 1}, Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, Validation Accuracy: {val_accuracies[-1]:.4f}, Train Recall: {train_recalls[-1]:.4f}, Validation Recall: {val_recalls[-1]:.4f}, Time: {fold_end_time - fold_start_time:.2f} seconds")
        
        fold_train_accuracies.append(train_accuracies[-1])
        fold_val_accuracies.append(val_accuracies[-1])
        fold_train_losses.append(train_losses[-1])
        fold_val_losses.append(val_losses[-1])
        fold_train_recalls.append(train_recalls[-1])
        fold_val_recalls.append(val_recalls[-1])
    logger.info(f"Best Model selected with Validation Loss: {best_val_loss:.4f}, saved at {best_model_path}")
    
    # Plot METRIC vs. Fold 
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(fold_train_accuracies) + 1), fold_train_accuracies, label="Train Accuracy")
    plt.plot(range(1, len(fold_val_accuracies) + 1), fold_val_accuracies, label="Test Accuracy")
    for i, (train_acc, val_acc) in enumerate(zip(fold_train_accuracies, fold_val_accuracies)):
        plt.text(i + 1, train_acc, f"{train_acc:.2f}", ha="center", va="bottom", fontsize=8)
        plt.text(i + 1, val_acc, f"{val_acc:.2f}", ha="center", va="bottom", fontsize=8)
    plt.title("Accuracy vs Fold Number")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(log_dir, "accuracy_vs_fold.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(fold_train_losses) + 1), fold_train_losses, label="Train Loss")
    plt.plot(range(1, len(fold_val_losses) + 1), fold_val_losses, label="Test Loss")
    for i, (train_loss, val_loss) in enumerate(zip(fold_train_losses, fold_val_losses)):
        plt.text(i + 1, train_loss, f"{train_loss:.2f}", ha="center", va="bottom", fontsize=8)
        plt.text(i + 1, val_loss, f"{val_loss:.2f}", ha="center", va="bottom", fontsize=8)
    plt.title("Loss vs Fold Number")
    plt.xlabel("Fold")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(log_dir, "loss_vs_fold.png"))
    plt.close()

   
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(fold_train_recalls) + 1), fold_train_recalls, label="Train Recall")
    plt.plot(range(1, len(fold_val_recalls) + 1), fold_val_recalls, label="Test Recall")
    for i, (train_recall, val_recall) in enumerate(zip(fold_train_recalls, fold_val_recalls)):
        plt.text(i + 1, train_recall, f"{train_recall:.2f}", ha="center", va="bottom", fontsize=8)
        plt.text(i + 1, val_recall, f"{val_recall:.2f}", ha="center", va="bottom", fontsize=8)

    plt.title("Recall vs Fold Number")
    plt.xlabel("Fold")
    plt.ylabel("Recall")
    plt.legend()
    plt.savefig(os.path.join(log_dir, "recall_vs_fold.png"))
    plt.close()
    
    # Testing
    logger.info("Starting Test Set Evaluation")
    best_model = Simple3DCNN().to(device)
    best_model.load_state_dict(torch.load(best_model_path))
    
    model = best_model
    model.eval()
    
    test_predictions, test_labels_actual = [], []
    test_loss = 0.0
    
    test_case_details = []
    with torch.no_grad():
        for images, labels in DataLoader(MRIDataset(test_paths, test_labels), batch_size=4, shuffle=False, num_workers=config['num_workers']):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_predictions.extend(predicted.cpu().numpy())
            test_labels_actual.extend(labels.cpu().numpy())
            for i, prediction in enumerate(predicted.cpu().numpy()):
                test_case_details.append({
                    "id": test_paths[i],  # Using the file path as the ID
                    "actual_label": labels.cpu().numpy()[i],
                    "predicted_label": prediction
                })
                
    true_positive = sum((np.array(test_predictions) == 1) & (np.array(test_labels_actual) == 1))
    false_positive = sum((np.array(test_predictions) == 1) & (np.array(test_labels_actual) == 0))
    true_negative = sum((np.array(test_predictions) == 0) & (np.array(test_labels_actual) == 0))
    false_negative = sum((np.array(test_predictions) == 0) & (np.array(test_labels_actual) == 1))
    
    
    test_accuracy = (np.array(test_predictions) == np.array(test_labels_actual)).mean()
    test_recall = torchmetrics.functional.recall(
        torch.tensor(test_predictions), torch.tensor(test_labels_actual), average="macro", task="binary"
    ).item()

    logger.info(f"Final Test Results - Loss: {test_loss / len(test_labels):.4f}, Accuracy: {test_accuracy:.4f}, Recall: {test_recall:.4f}")
    logger.info(f"True Positives: {true_positive}, False Positives: {false_positive}, "
            f"True Negatives: {true_negative}, False Negatives: {false_negative}")
    logger.info("Test Case Details:")
    logger.info(f"{'ID':<40} {'Actual Label':<15} {'Predicted Label':<15}")
    for case in test_case_details:
        logger.info(f"{case['id']:<40} {case['actual_label']:<15} {case['predicted_label']:<15}")
    
if __name__ == "__main__":
    seed = 41  # or any constant integer

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Name of the model to use (e.g., 3DCNN)")
    
    args = parser.parse_args()
    with open("config.json", "r") as f:
        config = json.load(f)
    if args.model not in config:
        raise ValueError(f"No such model: {args.model}. Check your configuration file.")

    model_config = config[args.model]
    if model_config["modality"] not in ['FLAIR', 'T1w']:
        raise ValueError(f"No such modality. Must be one of FLAIR or T1w.")

    log_dir = setup_logging(model_config["output_dir"])
    logger = logging.getLogger()
    logger.info(f'Epoch: {model_config["epochs"]}, Batch_size: {model_config["batch_size"]}, Modality: {model_config["modality"]}, learning_rate: {model_config["learning_rate"]}, num_workers: {model_config["num_workers"]}, test_size: {model_config["test_size"]}')
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    file_paths, labels = prepare_data("data/participants.tsv", model_config["modality"])
    train_and_validate(model_config, file_paths, labels, log_dir)
