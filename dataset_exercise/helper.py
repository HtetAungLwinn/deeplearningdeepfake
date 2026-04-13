from torch.utils.data import Dataset, Subset, ConcatDataset
import torch
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score, confusion_matrix

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
    for epoch in range(num_epochs):
        all_labels = []
        all_predicted = []
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
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.long().to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(probs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy().flatten())
                all_predicted.extend(predicted.cpu().numpy().flatten())
                all_probs.extend(probs[:,1].cpu().numpy().flatten())

        val_accuracy = 100 * correct / total    
        if scheduler != None:
            scheduler.step(val_accuracy)
        f1 = f1_score(all_labels, all_predicted)
        recall = recall_score(all_labels, all_predicted)
        precision = precision_score(all_labels, all_predicted)
        auc = roc_auc_score(all_labels, all_probs)
        cm = confusion_matrix(all_labels, all_predicted)
        cm = torch.tensor(cm)  # convert confusion matrix to tensor too
        print(f"\nEpoch : {epoch} ")
        print(f"Train Loss     : {epoch_loss/len(data_loader):.4f}")
        print(f"Validation Accuracy  : {val_accuracy:.2f}%")
        print(f"F1 Score       : {f1:.4f}")
        print(f"Recall         : {recall:.4f}  ← how many anomalies caught")
        print(f"Precision      : {precision:.4f}")
        print(f"AUC-ROC        : {auc:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted Real  Predicted Fake")
        print(f"Actual Real      {cm[0][0].item():<15} {cm[0][1].item()}")
        print(f"Actual Fake      {cm[1][0].item():<15} {cm[1][1].item()}")


def test(model, test_loader, device):
    total = 0
    correct = 0
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
            predicted = torch.argmax(probs, dim=1) 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        all_labels.extend(labels.cpu().numpy().flatten())
        all_predicted.extend(predicted.cpu().numpy().flatten())
        all_probs.extend(probs[:,1].cpu().numpy().flatten())
            
    test_accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_predicted)
    recall = recall_score(all_labels, all_predicted)
    precision = precision_score(all_labels, all_predicted)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_predicted)
    cm = torch.tensor(cm)  # convert confusion matrix to tensor too

    print(f"Test Accuracy  : {test_accuracy:.2f}%")
    print(f"F1 Score       : {f1:.4f}")
    print(f"Recall         : {recall:.4f}  ← how many anomalies caught")
    print(f"Precision      : {precision:.4f}")
    print(f"AUC-ROC        : {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted Real  Predicted Fake")
    print(f"Actual Real      {cm[0][0].item():<15} {cm[0][1].item()}")
    print(f"Actual Fake      {cm[1][0].item():<15} {cm[1][1].item()}")