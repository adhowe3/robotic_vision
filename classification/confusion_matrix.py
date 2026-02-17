import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

## import the model files
from fish_classify import FishClassifier

def get_predictions(model, device, loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            output = model(data)
            preds = output.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.numpy())

    return all_targets, all_preds

def plot_and_save_cm(y_true, y_pred, class_names, model_name):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('Actual Species')
    plt.xlabel('Predicted Species')
    
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name}.png')
    print(f"Saved {model_name} confusion matrix to disk.")
    plt.show()