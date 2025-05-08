import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from sklearn.metrics import jaccard_score
# from model_2 import MobileNetV3_Segmenter

# Your F1 score function
def compute_multiclass_fscore(mask_gt, mask_pred, beta=1):
    f_scores = []
    for class_id in np.unique(mask_gt):
        tp = np.sum((mask_gt == class_id) & (mask_pred == class_id))
        fp = np.sum((mask_gt != class_id) & (mask_pred == class_id))
        fn = np.sum((mask_gt == class_id) & (mask_pred != class_id))

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f_score = (
            (1 + beta**2)
            * (precision * recall)
            / ((beta**2 * precision) + recall + 1e-7)
        )
        f_scores.append(f_score)
    return np.mean(f_scores)


# 1. Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileNetV3_Segmenter(num_classes=19).to(device)
checkpoint = torch.load("/content/v3_10k_ckpt.pth", map_location=device)
model.load_state_dict(checkpoint)  # Load weights
model.eval()

# 2. Preprocess function for test images
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Adjust to match training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Adjust if different
])

# 3. Load test images and predict masks
test_dir = "/content/drive/MyDrive/celeb/test/images"  # Path to test images
mask_dir = "/content/drive/MyDrive/celeb/test/masks"   # Path to ground truth masks
output_dir = "/content/drive/MyDrive/celeb/test/output"
os.makedirs(output_dir, exist_ok=True)

test_images = os.listdir(test_dir)
f1=[]
IOU=[]
for img_name in test_images:
    # Load and preprocess image
    img_path = os.path.join(test_dir, img_name)
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        pred = model(input_tensor)  # Shape: [1, num_classes, H, W]
        # pred = torch.softmax(pred, dim=1)  # Convert logits to probabilities
        pred_mask = pred.argmax(dim=1).squeeze().cpu().numpy()  # Shape: [H, W]

    # 4. Load ground truth mask
    mask_name = img_name.replace(".jpg", ".png")  # Adjust extension if needed
    mask_path = os.path.join(mask_dir, mask_name)
    gt_mask = np.array(Image.open(mask_path))  # Assuming class indices (0, 1, 2, ...)

    # # Resize ground truth to match prediction size, if necessary
    # if gt_mask.shape != pred_mask.shape:
    #     gt_mask = np.array(Image.fromarray(gt_mask).resize(pred_mask.shape[::-1], Image.NEAREST))

    # 5. Compute metrics
    # IoU (multiclass)
    iou = jaccard_score(gt_mask.flatten(), pred_mask.flatten(), average="macro")
    # F1 score (beta=1 for standard F1)
    f1_score = compute_multiclass_fscore(gt_mask, pred_mask, beta=1)

    print(f"Image: {img_name}, IoU: {iou:.4f}, F1 Score: {f1_score:.4f}")
    f1.append(f1_score)
    IOU.append(iou)
    # Optional: Save predicted mask
    pred_image = Image.fromarray(pred_mask.astype(np.uint8))
    pred_image.save(os.path.join(output_dir, f"{mask_name}"), format="PNG")

print("Mean F1 Score:", np.mean(f1))
print("Mean IOU score:", np.mean(IOU))
print("Unique GT classes:", np.unique(gt_mask))
print("Unique Pred classes:", np.unique(pred_mask))
# Optional: Visualize a sample
import matplotlib.pyplot as plt
def visualize_sample(image, pred_mask, gt_mask):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(image)
    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask, cmap="jet")  # 'jet' for multiclass visualization
    plt.subplot(1, 3, 3)
    plt.title("Ground Truth Mask")
    plt.imshow(gt_mask, cmap="jet")
    plt.savefig("/content/v3_orig_output")

# Call for one sample
visualize_sample(image, pred_mask, gt_mask)