import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

# 1. Set paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    PROJECT_DIR, "models", "efficientnet_b0_finetuned_best.pth"
)

TEST_PATH = os.path.expanduser("~/Downloads/Dataset/Test")

# 2. Use CPU
device = torch.device("cpu")
print("Device:", device)

# 3. Load the same model architecture
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features, 2
)

checkpoint = torch.load(
    MODEL_PATH, map_location=device, weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# 4. Use the same image preprocessing as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 5. Load test images
test_dataset = datasets.ImageFolder(
    TEST_PATH,
    transform=transform
)

print("\nClass mapping:", test_dataset.class_to_idx)
print("Total images:", len(test_dataset))

# 6. Check that the labels match the trained model
expected_mapping = {"Fake": 0, "Real": 1}

if test_dataset.class_to_idx != expected_mapping:
    raise ValueError(
        f"Unexpected class mapping: {test_dataset.class_to_idx}"
    )

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

# 7. Evaluate the model
# Rows = actual class; columns = predicted class
confusion = [[0, 0], [0, 0]]

correct = 0
total = 0

print("\nEvaluating model...")

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        for actual, predicted in zip(labels, predictions):
            confusion[actual.item()][predicted.item()] += 1

# 8. Calculate metrics
accuracy = correct / total * 100

# Fake = class 0, Real = class 1
fake_tp = confusion[0][0]
fake_fn = confusion[0][1]
fake_fp = confusion[1][0]
fake_tn = confusion[1][1]

fake_precision = fake_tp / (fake_tp + fake_fp) if fake_tp + fake_fp else 0
fake_recall = fake_tp / (fake_tp + fake_fn) if fake_tp + fake_fn else 0

real_tp = confusion[1][1]
real_fn = confusion[1][0]
real_fp = confusion[0][1]

real_precision = real_tp / (real_tp + real_fp) if real_tp + real_fp else 0
real_recall = real_tp / (real_tp + real_fn) if real_tp + real_fn else 0

fake_f1 = (
    2 * fake_precision * fake_recall / (fake_precision + fake_recall)
    if fake_precision + fake_recall else 0
)

real_f1 = (
    2 * real_precision * real_recall / (real_precision + real_recall)
    if real_precision + real_recall else 0
)

# 9. Print results
print("\n========== EVALUATION RESULTS ==========")
print(f"Total images: {total}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {accuracy:.2f}%")

print("\nConfusion Matrix")
print("                 Predicted Fake  Predicted Real")
print(f"Actual Fake      {confusion[0][0]:13}  {confusion[0][1]:13}")
print(f"Actual Real      {confusion[1][0]:13}  {confusion[1][1]:13}")

print("\nFake Class")
print(f"Precision: {fake_precision * 100:.2f}%")
print(f"Recall:    {fake_recall * 100:.2f}%")
print(f"F1-score:  {fake_f1 * 100:.2f}%")

print("\nReal Class")
print(f"Precision: {real_precision * 100:.2f}%")
print(f"Recall:    {real_recall * 100:.2f}%")
print(f"F1-score:  {real_f1 * 100:.2f}%")

print("\nMacro F1-score:", f"{(fake_f1 + real_f1) / 2 * 100:.2f}%")
print("\nEvaluation completed.")