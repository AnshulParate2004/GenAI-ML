import os
import torch
from fer_pytorch import fer

# 📂 Where to save your model
save_folder = r"D:\emotionmodel_ferplus"
os.makedirs(save_folder, exist_ok=True)

# ⬇️ Download pretrained FER+ ResNet-18 model
model = fer.get_pretrained_model("resnet18")

# 💾 Save model weights
model_path = os.path.join(save_folder, "resnet18_ferplus.pth")
torch.save(model.state_dict(), model_path)

print(f"✅ Pretrained FER+ ResNet-18 model downloaded & saved at:\n{model_path}")
