import pandas as pd
from tqdm import tqdm
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.svm import SVC
from sklearn.metrics import classification_report


ROOT_DIR = "oyster_shell/"
BG_PATH = os.path.join(ROOT_DIR, "background.tif")
TRAIN_DIR = os.path.join(ROOT_DIR, "train")

def preprocess_oyster(image_path, bg_path):
    # Load the oyster image and the background
    if os.path.exists(image_path) and os.path:
        img = cv2.imread(image_path)
    else:
        print(image_path, "Path does not exist!")
        exit()
    bg = cv2.imread(bg_path)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    diff = cv2.absdiff(img_blur, bg)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_diff, 5, 255, cv2.THRESH_BINARY)
    
    return img, mask

def get_oyster_metrics(mask):
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    # Get the largest contour (assuming it's the oyster)
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    compactness = (perimeter**2) / (4 * np.pi * area + 1e-5)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0

    # Elongation
    rect = cv2.minAreaRect(cnt)
    width, height = rect[1]
    major = max(width, height)
    minor = min(width, height)
    elongation = major / (minor + 1e-5)

    # Extent (Area / Bounding Box Area)
    _, _, b_w, b_h = cv2.boundingRect(cnt)
    extent = float(area) / (b_w * b_h) if (b_w * b_h) > 0 else 0
    
    # Equivalent Diameter
    equiv_diameter = np.sqrt(4 * area / np.pi)

    # roughness
    hull_perimeter = cv2.arcLength(hull, True)
    roughness = perimeter / (hull_perimeter + 1e-5)

    # Combine everything into the dictionary
    results = {
        "compactness": compactness,
        "elongation": elongation,
        "solidity": solidity,
        "roughness": roughness,
        "extent": extent,
        "contour": cnt # Keep for visualization
    }
    
    return results


def visualize_oyster_analysis(img, mask, metrics):
    if metrics is None:
        print("No oyster detected in this image.")
        return

    # Create a copy so we don't modify the original
    vis_img = img.copy()
    cnt = metrics['contour']

    # 1. Draw the actual contour (Green line, thickness 3)
    cv2.drawContours(vis_img, [cnt], -1, (0, 255, 0), 3)

    # 2. Draw the Rotated Rect (Blue line) for Elongation
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(vis_img, [box], 0, (255, 0, 0), 2)

    # 3. ADDED: Draw the Convex Hull (Red line) for Solidity
    hull = cv2.convexHull(cnt)
    cv2.drawContours(vis_img, [hull], -1, (0, 0, 255), 2) # Red line

    # Display the results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Binary Mask (After Morphology)")
    plt.imshow(mask, cmap='gray')
    
    plt.subplot(1, 2, 2)
    # Added Solidity to the title for easy checking
    plt.title(f"Contour Analysis\nComp: {metrics['compactness']:.2f} | Elong: {metrics['elongation']:.2f}")
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    
    plt.tight_layout()
    plt.savefig("plt.png")


def extract_all_features(data_dir, bg_path):
    features_list = []
    # Class names are the folder names: ['Banana', 'Irregular', 'Good', 'Bad']
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for label in classes:
        class_path = os.path.join(data_dir, label)
        print(f"Processing class: {label}")
        
        for img_name in tqdm(os.listdir(class_path)):
            img_path = os.path.join(class_path, img_name)
            if not img_name.lower().endswith(".tif"):
                continue

            # Use your successful preprocessing from Step 3
            img, mask = preprocess_oyster(img_path, bg_path) 
            metrics = get_oyster_metrics(mask)
            
            if metrics:
                # Add the label to our dictionary
                metrics['label'] = label
                metrics['filename'] = img_name
                # Remove the contour array so it can be saved in a table
                # del metrics['contour'] 
                features_list.append(metrics)
                
    return pd.DataFrame(features_list)

# before model eval, lets see how features seperate
def plot_oyster_features(df):
    plt.figure(figsize=(10, 6))
    # Plot Elongation vs Compactness
    sns.scatterplot(data=df, x='elongation', y='roughness', hue='label', style='label', s=100)
    
    plt.title("Oyster Shape Analysis: Elongation vs Roughness")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("pd_feature_map.png")

# define the model, simple nn
class OysterNet(nn.Module):
    def __init__(self, input_size=5, num_classes=4):
        super(OysterNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

        
# Define the Dataset Class
class OysterShapeDataset(Dataset):
    def __init__(self, dataframe):
        # Maps folder names to integers 0-3
        self.label_map = {name: i for i, name in enumerate(sorted(dataframe['label'].unique()))}
        self.labels = [self.label_map[label] for label in dataframe['label']]
        self.features = dataframe[feature_cols].values
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
    
def train_and_evaluate(model, train_loader, test_loader, optimizer, epochs=100):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optimizer
    
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Every 10 epochs, check accuracy on test set
        if (epoch + 1) % 10 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Test Accuracy: {accuracy:.2f}%")


def plot_confusion_matrix(model, test_loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    # Generate the matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Oyster Classification Confusion Matrix')
    plt.savefig("oyster_confusion.png")

def plot_xgb_confusion_matrix(model, X_test, y_test, class_names):
    # 1. Get predictions from XGBoost
    # y_pred will be an array of integers (0, 1, 2, 3)
    y_pred = model.predict(X_test)
    
    # 2. Generate the matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # 3. Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Oyster Classification: XGBoost Confusion Matrix', fontsize=14)
    plt.savefig("xgb_confusion.png")


if __name__ == "__main__":
    example_img = os.path.join(TRAIN_DIR, "Good/125.tif")
    original, mask = preprocess_oyster(example_img, BG_PATH)
    cv2.imwrite("mask.png", mask)
    cv2.imwrite("og.png", original)

    metrics = get_oyster_metrics(mask)
    visualize_oyster_analysis(original, mask, metrics)

    ## create the model
    model = OysterNet(input_size=5, num_classes=4)


    print("Extracting Train Features...")
    df_train = extract_all_features(os.path.join(ROOT_DIR, "train"), BG_PATH)
    plot_oyster_features(df_train)

    print("Extracting Test Features...")
    df_test = extract_all_features(os.path.join(ROOT_DIR, "test"), BG_PATH)

    # Scaling the data
    # We "fit" on train and "transform" on test to prevent data leakage
    scaler = StandardScaler()
    feature_cols = ['compactness', 'elongation', 'solidity', 'roughness', 'extent']
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_test[feature_cols] = scaler.transform(df_test[feature_cols])

    # Create Loaders
    train_ds = OysterShapeDataset(df_train)
    test_ds = OysterShapeDataset(df_test)

    # weigted sampler
    # target_list = torch.tensor(train_ds.labels)
    # counts = df_train['label'].value_counts().sort_index().values
    # class_weights = 1. / torch.tensor(counts, dtype=torch.float)
    # sample_weights = class_weights[target_list]
    # # Create the sampler
    # sampler = WeightedRandomSampler(
    #     weights=sample_weights,
    #     num_samples=len(sample_weights),
    #     replacement=True
    # )

    # We don't need to shuffle test
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Peek at the actual tensors being fed to the model
    example_x, example_y = next(iter(train_loader))
    print("Example Tensor Input:\n", example_x[0])

    # 3. Pass these weights to the loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    print("Train Mapping:", train_ds.label_map)
    print("Test Mapping:", test_ds.label_map)

    # train_and_evaluate(model=model, train_loader=train_loader, optimizer=optimizer, test_loader=test_loader, epochs=500)
    # class_names should be sorted alphabetically to match our Label Encoder
    # e.g., ['bad', 'banana', 'good', 'irregular']
    # classes = sorted(df_train['label'].unique())
    # plot_confusion_matrix(model, test_loader, classes)

    #### XGBOOST #####
    X_train = df_train[feature_cols]
    y_train = pd.Categorical(df_train['label']).codes

    X_test = df_test[feature_cols]
    y_test = pd.Categorical(df_test['label']).codes
    classes = sorted(df_train['label'].unique())

    # 2. Initialize the Model
    # 'multi:softprob' is for multi-class classification
    # we use 'n_estimators' to give it enough trees to learn
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    model_xgb = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        objective='multi:softprob',
        num_class=4,
        random_state=42
    )

    # 3. Fit the Model
    model_xgb.fit(X_train, y_train, sample_weight=sample_weights)

    # 4. Predict and Evaluate
    y_pred = model_xgb.predict(X_test)

    print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=sorted(df_train['label'].unique())))

    plot_xgb_confusion_matrix(model_xgb, X_test, y_test, classes)

    xgb.plot_importance(model_xgb)
    plt.title("What makes an oyster? Feature Importance")
    plt.savefig("xgb_importance.png")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score

    # 1. Initialize Random Forest
    # 'balanced_subsample' calculates weights at every tree level, which is great for overlap
    # model_rf = RandomForestClassifier(
    #     n_estimators=200, 
    #     max_depth=6, 
    #     class_weight='balanced_subsample',
    #     random_state=42
    # )

    # # 2. Fit the model
    # model_rf.fit(X_train, y_train)

    # # 3. Predict
    # y_pred_rf = model_rf.predict(X_test)

    # # 4. Print Results
    # print(f"Random Forest Total Accuracy: {accuracy_score(y_test, y_pred_rf):.2%}")
    # print("\nDetailed Classification Report:")
    # print(classification_report(y_test, y_pred_rf, target_names=classes))


