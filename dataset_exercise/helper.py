from torch.utils.data import Dataset, Subset
import torch
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score, confusion_matrix

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor()
])

class RelabelDataset(Dataset):
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image, self.label

def make_subset(folder, label, n):
    dataset = datasets.ImageFolder(root=folder, transform=transform)
    indices = torch.randperm(len(dataset))[:n].tolist()
    return RelabelDataset(Subset(dataset, indices), label)



def train(model, data_loader, valid_loader, criterion, optimizer, device, num_epochs=5, lr=0.001):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

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
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                predicted = (outputs > 0.5).float()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_accuracy:.2f}%, Loss: {epoch_loss / len(data_loader):.4f}")
            
def test(model, test_loader, device):
    total = 0
    correct = 0
    all_labels = []
    all_predicted = []
    all_probs = []
    model.eval()
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        with torch.no_grad():
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            predicted = (outputs > 0.5).float() 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        all_labels.extend(labels.cpu().numpy().flatten())
        all_predicted.extend(predicted.cpu().numpy().flatten())
        all_probs.extend(outputs.cpu().numpy().flatten())
            
    test_accuracy = 100 * correct / total
    f1 = f1_score(all_labels.tolist(), all_predicted.tolist())
    recall = recall_score(all_labels.tolist(), all_predicted.tolist())
    precision = precision_score(all_labels.tolist(), all_predicted.tolist())
    auc = roc_auc_score(all_labels.tolist(), all_probs.tolist())
    cm = confusion_matrix(all_labels.tolist(), all_predicted.tolist())
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