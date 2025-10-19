import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
import os

# ---------------------
# 1. 데이터셋 & 전처리
# ---------------------
data_dir = "데이터주소"
batch_size = 32

transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "validation": transforms.Compose([   # 🔑 validation 추가
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([         # 🔑 test 추가
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# 📌 폴더 경로 각각 지정
#기존이미지(전체중 90%로 학습)
train_dataset_origin = datasets.ImageFolder("원본이미지주소", transform["train"])
origin_num_samples = int(len(train_dataset_origin) * 0.9)
indices = np.random.choice(len(train_dataset_origin), origin_num_samples, replace=False)
origin_small_dataset = Subset(train_dataset_origin, indices)
#변형이미지(전체충 10%로 학습)
train_dataset_noised = datasets.ImageFolder("변형이미지주소", transform["train"])
noised_num_samples = int(len(train_dataset_noised) * 0.1)
indices = np.random.choice(len(train_dataset_noised), noised_num_samples, replace=False)
noised_small_dataset = Subset(train_dataset_noised, indices)





combined_train_dataset = ConcatDataset([origin_small_dataset, noised_small_dataset])
val_dataset   = datasets.ImageFolder(os.path.join(data_dir, "validation"), transform["validation"])
test_dataset  = datasets.ImageFolder(os.path.join(data_dir, "test"), transform["test"])

train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

num_classes = len(train_dataset_origin.classes)  # 라벨 개수 자동 추출

# ---------------------
# 2. 모델 정의
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

# ---------------------
# 3. Loss & Optimizer
# ---------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# ---------------------
# 4. 학습 루프
# ---------------------
best_acc = 0.0
num_epochs = 10
save_path = "저장경로"

for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch+1}/{num_epochs}]")

    # ---- Training_Origin ----
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total
    # ---- Validation ----
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    val_loss /= total
    val_acc = correct / total

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # ---- Save Best Model ----
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"✅ 모델 저장됨: {save_path}")
#해야할것 (결과시각화)----------------------------------------
print(f"\n🎯 학습 완료! 최고 정확도: {best_acc:.4f}")
