from torch.utils.data import Dataset, Subset, ConcatDataset
import torch
from torchvision import datasets, transforms, models
import numpy as np
from sklearn.metrics import (f1_score, recall_score, roc_auc_score, 
                             precision_score, confusion_matrix, 
                             precision_recall_curve, average_precision_score)

class RelabelDataset(Dataset):
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image, torch.tensor(self.label, dtype=torch.long)

def make_dataset(folder, n_1, n_2, transform):
    dataset = datasets.ImageFolder(root=folder, transform=transform)
    real, fake = get_indices(dataset)
    real_subset = RelabelDataset(Subset(dataset, real[:n_1]), label=0) 
    fake_subset = RelabelDataset(Subset(dataset, fake[:n_2]), label=1)
    return ConcatDataset([real_subset, fake_subset])

def get_indices(dataset):
    real_indices = [i for i, label in enumerate(dataset.targets) if label == dataset.class_to_idx['real']]
    fake_indices = [i for i, label in enumerate(dataset.targets) if label == dataset.class_to_idx['fake']]
    return real_indices, fake_indices


def train(model, data_loader, valid_loader, criterion, optimizer, device, scheduler=None, num_epochs=5):
    best_recall = 0
    loss_values = []
    best_threshold_per_epoch = []

    for epoch in range(num_epochs):
        all_labels = []
        all_probs = []
        model.train()
        epoch_loss = 0

        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.long().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.long().to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                all_labels.extend(labels.cpu().numpy().flatten())
                all_probs.extend(probs[:, 1].cpu().numpy().flatten())

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Find optimal threshold via precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = f1_scores.argmax()
        best_threshold = thresholds[best_idx]
        best_threshold_per_epoch.append(best_threshold)
        
        # Predict with optimal threshold
        all_predicted = (all_probs > best_threshold).astype(int)

        loss_values.append(epoch_loss / len(data_loader))

        if scheduler is not None:
            scheduler.step(np.mean(all_predicted == all_labels))

        # Calculate metrics with optimal threshold
        f1 = f1_score(all_labels, all_predicted, zero_division=0)
        recall = recall_score(all_labels, all_predicted, zero_division=0)
        precision = precision_score(all_labels, all_predicted, zero_division=0)
        auc_roc = roc_auc_score(all_labels, all_probs)
        pr_auc = average_precision_score(all_labels, all_probs)
        cm = confusion_matrix(all_labels, all_predicted)
        cm = torch.tensor(cm)

        if recall > best_recall:
            best_recall = recall
            torch.save(model.state_dict(), 'best_model.pth')

        print(f"\nEpoch : {epoch}")
        print(f"Optimal Threshold : {best_threshold:.4f}")
        print(f"Train Loss        : {epoch_loss/len(data_loader):.4f}")
        print(f"F1 Score          : {f1:.4f}")
        print(f"Recall            : {recall:.4f}  <- how many anomalies caught")
        print(f"Precision         : {precision:.4f}")
        print(f"AUC-ROC           : {auc_roc:.4f}")
        print(f"PR-AUC            : {pr_auc:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"                   Predicted Real  Predicted Fake")
        print(f"Actual Real        {cm[0][0].item():<15} {cm[0][1].item()}")
        print(f"Actual Fake        {cm[1][0].item():<15} {cm[1][1].item()}")

    return loss_values, best_threshold_per_epoch


def test(model, test_loader, device, threshold=None):
    all_labels = []
    all_predicted = []
    all_probs = []
    model.eval()

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.long().to(device)

        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy().flatten())
        all_probs.extend(probs[:, 1].cpu().numpy().flatten())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # If no threshold provided, compute optimal from test data
    if threshold is None:
        precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        threshold = thresholds[f1_scores.argmax()]
        
        print(f"(Optimal threshold computed from test data: {threshold:.4f})")
    
    all_predicted = (all_probs > threshold).astype(int)

    f1 = f1_score(all_labels, all_predicted, zero_division=0)
    recall = recall_score(all_labels, all_predicted, zero_division=0)
    precision = precision_score(all_labels, all_predicted, zero_division=0)
    auc_roc = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_predicted)
    cm = torch.tensor(cm)

    print(f"\n{'='*50}")
    print(f"TEST SET RESULTS (Threshold: {threshold:.4f})")
    print(f"{'='*50}")
    print(f"F1 Score          : {f1:.4f}")
    print(f"Recall            : {recall:.4f}  <- how many anomalies caught")
    print(f"Precision         : {precision:.4f}")
    print(f"AUC-ROC           : {auc_roc:.4f}")
    print(f"PR-AUC            : {pr_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                   Predicted Real  Predicted Fake")
    print(f"Actual Real        {cm[0][0].item():<15} {cm[0][1].item()}")
    print(f"Actual Fake        {cm[1][0].item():<15} {cm[1][1].item()}")
    print(f"{'='*50}")
    return threshold, f1_scores, thresholds