import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

LETTER_IMAGES_FOLDER = "extracted_characters"
MODEL_FILENAME = "captcha_recognition_model.pth"
MODEL_LABELS_FILENAME = "model_labels.dat"

# 3. Dataset and DataLoader
class CaptchaDataset(Dataset):
    def __init__(self, samples, label_binarizer, transform=None):
        self.samples = samples
        self.lb = label_binarizer
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = cv2.imread(image_path)
        if image is None:
            image = np.zeros((40, 40, 1), dtype=np.float32)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (40, 40))
            image = np.expand_dims(image, axis=2)
        if self.transform:
            image = self.transform(image)
        label_vec = self.lb.transform([label])[0].astype(np.float32)
        return image, label_vec
    
# 4. Model Definition
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = None
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        self.act2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.skip:
            identity = self.skip(identity)
        out += identity
        out = self.act2(out)
        return out

class EnhancedModel(nn.Module):
    def __init__(self, num_classes=36):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 7, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool2d(2)
        self.res1 = ResidualBlock(64, 64)
        self.drop1 = nn.Dropout(0.2)
        self.pool2 = nn.MaxPool2d(2)
        self.res2 = ResidualBlock(64, 128)
        self.drop2 = nn.Dropout(0.2)
        self.pool3 = nn.MaxPool2d(2)
        self.res3 = ResidualBlock(128, 256)
        self.drop3 = nn.Dropout(0.3)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.act2 = nn.LeakyReLU(0.1)
        self.drop4 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.act3 = nn.LeakyReLU(0.1)
        self.drop5 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.res1(x)
        x = self.drop1(x)
        x = self.pool2(x)
        x = self.res2(x)
        x = self.drop2(x)
        x = self.pool3(x)
        x = self.res3(x)
        x = self.drop3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.drop4(x)
        x = self.fc2(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.drop5(x)
        x = self.fc3(x)
        return x
    
if __name__ == "__main__":
    # 1. Gather all samples and split
    all_samples = []
    for label in os.listdir(LETTER_IMAGES_FOLDER):
        label_folder = os.path.join(LETTER_IMAGES_FOLDER, label)
        if os.path.isdir(label_folder):
            for image_file in os.listdir(label_folder):
                image_path = os.path.join(label_folder, image_file)
                all_samples.append((image_path, label))

    train_samples, test_samples = train_test_split(all_samples, test_size=0.25, random_state=42)

    # 2. Label Binarizer
    all_labels = [chr(i) for i in range(97, 123)] + [str(i) for i in range(10)]
    lb = LabelBinarizer().fit(all_labels)

    # Save label binarizer
    with open(MODEL_LABELS_FILENAME, "wb") as f:
        pickle.dump(lb, f)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.15, 0.15), shear=15),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    train_dataset = CaptchaDataset(train_samples, lb, transform=train_transform)
    test_dataset = CaptchaDataset(test_samples, lb, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # 5. Training Setup
    num_classes = 36
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    model = EnhancedModel(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    # 6. Training Loop with Early Stopping and Checkpoint
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    num_epochs = 150

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(torch.sigmoid(outputs), dim=1)
            targets = torch.argmax(labels, dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += images.size(0)
        train_acc = train_correct / train_total
        train_loss /= train_total

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(torch.sigmoid(outputs), dim=1)
                targets = torch.argmax(labels, dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += images.size(0)
        val_acc = val_correct / val_total
        val_loss /= val_total

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        # Early stopping and checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_FILENAME)
            print(f"Model saved at epoch {epoch+1} with val_acc={val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
        # Reduce LR on plateau
        if patience_counter > 0 and patience_counter % 5 == 0:
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = max(old_lr * 0.5, 1e-6)
                param_group['lr'] = new_lr
                print(f"Learning rate reduced to {new_lr}")
