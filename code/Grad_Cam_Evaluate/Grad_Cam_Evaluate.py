import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ---------------------
# 1. 데이터셋 불러오기
# ---------------------
test_dir = "테스트 데이터 폴더"  # 테스트 데이터 폴더
save_wrong_dir = "wrong_heatmaps"
save_correct_dir = "correct_heatmap"
os.makedirs(save_wrong_dir, exist_ok=True)
os.makedirs(save_correct_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

classes = test_dataset.classes

# ---------------------
# 2. 모델 불러오기
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
model.load_state_dict(torch.load("D:/origin_img_trained_10epochs.pth", map_location=device))
model = model.to(device)
model.eval()

# ---------------------
# 3. Grad-CAM 클래스
# ---------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, class_idx):
        grads = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP
        cam = (grads * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# EfficientNet-B0 마지막 conv layer
target_layer = model.features[-1][0]
grad_cam = GradCAM(model, target_layer)

# ---------------------
# 4. 테스트셋 평가 & 잘못된 샘플 저장
# ---------------------
number_class = len(classes) #라벨 갯수 저장
correct_list = [0 for _ in range(number_class)]  #맞은 것 개수세기 위한 리스트
wrong_list = [0 for _ in range(number_class)] #틀린 것 개수세기 위한 리스트
max_correct_save = 5 #클래스별 최대 5개까지만 저장

for idx, (image, label) in enumerate(test_loader):
    image, label = image.to(device), label.to(device)
    output = model(image)
    pred = output.argmax(1)

    if pred != label:  # 틀린 경우만 저장
        wrong_list[label] +=1 #잘못된 곳 카운트
        # Grad-CAM 생성
        model.zero_grad()
        class_idx = pred.item()
        score = output[0, class_idx]
        score.backward(retain_graph=True)
        cam = grad_cam.generate(class_idx)

        # 원본 이미지 복원
        img_np = image.squeeze().cpu().permute(1, 2, 0).numpy()
        img_np = (img_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        img_np = np.clip(img_np, 0, 1)

        # 히트맵 적용
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (heatmap.astype(np.float32) / 255 + img_np) / 2
        overlay = np.clip(overlay, 0, 1)

        # 저장
        filename = f"{idx}_true-{classes[label.item()]}_pred-{classes[pred.item()]}.png"
        save_path = os.path.join(save_wrong_dir, filename)
        Image.fromarray((overlay * 255).astype(np.uint8)).save(save_path)
    else: #맞은경우
        correct_list[label] += 1
        if correct_list[label] <= max_correct_save:
            # Grad-CAM 생성
            model.zero_grad()
            class_idx = pred.item()
            score = output[0, class_idx]
            score.backward(retain_graph=True)
            cam = grad_cam.generate(class_idx)

            # 원본 이미지 복원
            img_np = image.squeeze().cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            img_np = np.clip(img_np, 0, 1)

            # 히트맵 적용
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = (heatmap.astype(np.float32) / 255 + img_np) / 2
            overlay = np.clip(overlay, 0, 1)

            # 저장
            filename = f"{idx}_true-{classes[label.item()]}_pred-{classes[pred.item()]}.png"
            save_path = os.path.join(save_correct_dir, filename)
            Image.fromarray((overlay * 255).astype(np.uint8)).save(save_path)

# ---------------------
# 5. 저장 확인
# ---------------------
print(f"✅ 잘못 예측된 샘플 히트맵 저장 완료 → {save_wrong_dir}")
print(f"✅ 정확하게 예측된 샘플 히트맵 저장 완료 → {save_correct_dir}")
print(f"\n맞은 개수: {correct_list}")
print(f"틀린 개수: {wrong_list}")

# ---------------------
# 6. 시각화
# ---------------------
x = np.arange(number_class)
plt.figure(figsize=(8, 6))
plt.bar(x, correct_list, label='Correct', color='green')
plt.bar(x, wrong_list, bottom=correct_list, label='Wrong', color='red')

plt.xticks(x, classes)
plt.ylabel("Number of Samples")
plt.title("Number of Correct and Wrong Samples by Class")
plt.legend()
plt.tight_layout()
plt.show()