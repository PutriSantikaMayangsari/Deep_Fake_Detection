# modules/visualization.py

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

def plot_and_save_metrics(train_loss, val_loss, train_acc, val_acc, save_path="training_metrics.png"):
    epochs_range = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss, 'o-', label='Training Loss')
    plt.plot(epochs_range, val_loss, 'o-', label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_acc, 'o-', label='Training Accuracy')
    plt.plot(epochs_range, val_acc, 'o-', label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)

    plt.suptitle('Model Training Metrics')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.show()
    print(f"Metrics plot saved to {save_path}")

def evaluate_on_test_set(model, test_loader, device, save_path="test_evaluation_results.png"):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            # Cek jika input adalah sekuens (video) atau gambar tunggal
            if len(inputs.shape) == 5: # [B, T, C, H, W] for video
                sequences, labels = inputs.to(device), labels.to(device)
                outputs = model(sequences)
            else: # [B, C, H, W] for image
                images, labels = inputs.to(device), labels.to(device)
                outputs = model(images)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs.data, 1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n--- Test Set Evaluation ---")
    print(classification_report(all_labels, all_preds, target_names=['REAL', 'FAKE']))

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['REAL', 'FAKE'])
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
    plt.title('Confusion Matrix')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Test evaluation plots saved to {save_path}")