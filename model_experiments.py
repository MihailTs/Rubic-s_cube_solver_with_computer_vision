from sklearn.model_selection import GridSearchCV
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

class ColorCNN(nn.Module):

    def __init__(self,
                 num_classes,
                 pooling='max',
                 kernel_size=3,
                 activation_function='relu'):
        super().__init__()

        self.kernel_size = kernel_size
        if kernel_size == 3:
           self.padding = 1
        elif kernel_size == 5:
           self.padding = 2
        else:
           raise ValueError('kernel_size should either be 3 or 5')

        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size, padding=self.padding)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, padding=self.padding)

        if pooling == 'max_pooling':
            self.pool = nn.MaxPool2d(2, 2)
        elif pooling == 'average_pooling':
            self.pool = nn.AvgPool2d(2, 2)
        else:
            raise ValueError("Illegal pooling method")

        if activation_function != 'relu' and activation_function != 'elu' and activation_function != 'sigmoid' and activation_function != 'tanh':
            raise ValueError("Illegal activation function")
        self.activation_function = activation_function

        self.fc1 = nn.Linear(32 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        if self.activation_function == 'relu':
          x = self.pool(F.relu(self.conv1(x)))
          x = self.pool(F.relu(self.conv2(x)))
        elif self.activation_function == 'elu':
          x = self.pool(F.elu(self.conv1(x)))
          x = self.pool(F.elu(self.conv2(x)))
        elif self.activation_function == 'sigmoid':
          x = self.pool(F.sigmoid(self.conv1(x)))
          x = self.pool(F.sigmoid(self.conv2(x)))
        elif self.activation_function == 'tanh':
          x = self.pool(F.tanh(self.conv1(x)))
          x = self.pool(F.tanh(self.conv2(x)))

        x = x.view(x.size(0), -1)
        if self.activation_function == 'relu':
          x = F.relu(self.fc1(x))
        elif self.activation_function == 'elu':
          x = F.elu(self.fc1(x))
        elif self.activation_function == 'sigmoid':
          x = F.sigmoid(self.fc1(x))
        elif self.activation_function == 'tanh':
          x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


def load_images_to_array(dataloader, device=None):
    """Convert images from dataloader to numpy array and labels"""
    images_list = []
    labels_list = []
    
    for images, labels in dataloader:
        if device:
            images = images.to(device)
        batch_size = images.size(0)
        images_flat = images.view(batch_size, -1)
        images_list.append(images_flat.cpu().numpy())
        labels_list.append(labels.numpy())
    
    X = np.vstack(images_list)
    y = np.concatenate(labels_list)
    return X, y

def train_sklearn_models(X_train, y_train, X_val, y_val):
    """Train KNN, SVM, and Logistic Regression models with GridSearchCV"""
    results = {}
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # KNN GridSearch
    print("\nTraining KNN with GridSearchCV...")
    knn_params = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    knn = KNeighborsClassifier()
    knn_grid = GridSearchCV(knn, knn_params, cv=5, scoring='f1_weighted', n_jobs=-1)
    knn_grid.fit(X_train_scaled, y_train)
    best_knn = knn_grid.best_estimator_
    y_pred_knn = best_knn.predict(X_val_scaled)
    knn_accuracy = np.mean(y_pred_knn == y_val)
    knn_f1 = f1_score(y_val, y_pred_knn, average='weighted')
    results['KNN'] = (knn_accuracy, knn_f1)
    print(f"KNN Best Parameters: {knn_grid.best_params_}")
    print(f"KNN Validation Accuracy: {knn_accuracy:.4f}, F1 Score: {knn_f1:.4f}")
    
    # SVM GridSearch
    print("\nTraining SVM with GridSearchCV...")
    svm_params = {
        'kernel': ['rbf', 'poly'],
        'C': [0.1, 0.3, 0.5, 1, 3, 5, 10],
        'gamma': ['scale', 'auto']
    }
    svm = SVC()
    svm_grid = GridSearchCV(svm, svm_params, cv=5, scoring='f1_weighted', n_jobs=-1)
    svm_grid.fit(X_train_scaled, y_train)
    best_svm = svm_grid.best_estimator_
    y_pred_svm = best_svm.predict(X_val_scaled)
    svm_accuracy = np.mean(y_pred_svm == y_val)
    svm_f1 = f1_score(y_val, y_pred_svm, average='weighted')
    results['SVM'] = (svm_accuracy, svm_f1)
    print(f"SVM Best Parameters: {svm_grid.best_params_}")
    print(f"SVM Validation Accuracy: {svm_accuracy:.4f}, F1 Score: {svm_f1:.4f}")
    
    # Logistic Regression GridSearch
    print("\nTraining Logistic Regression with GridSearchCV...")
    lr_params = {
        'C': [0.1, 0.3, 0.5, 1, 5, 10, 100],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [1000, 2000]
    }
    lr = LogisticRegression(random_state=42)
    lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring='f1_weighted', n_jobs=-1)
    lr_grid.fit(X_train_scaled, y_train)
    best_lr = lr_grid.best_estimator_
    y_pred_lr = best_lr.predict(X_val_scaled)
    lr_accuracy = np.mean(y_pred_lr == y_val)
    lr_f1 = f1_score(y_val, y_pred_lr, average='weighted')
    results['Logistic Regression'] = (lr_accuracy, lr_f1)
    print(f"Logistic Regression Best Parameters: {lr_grid.best_params_}")
    print(f"Logistic Regression Validation Accuracy: {lr_accuracy:.4f}, F1 Score: {lr_f1:.4f}")
    
    return results, scaler, best_knn, best_svm, best_lr

def evaluate_sklearn_on_test(knn, svm, lr, X_test, y_test, scaler):
    """Evaluate sklearn models on test set"""
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # KNN Test
    y_pred = knn.predict(X_test_scaled)
    knn_test_accuracy = np.mean(y_pred == y_test)
    knn_test_f1 = f1_score(y_test, y_pred, average='weighted')
    results['KNN'] = (knn_test_accuracy, knn_test_f1)
    print(f"KNN Test Accuracy: {knn_test_accuracy:.4f}, F1 Score: {knn_test_f1:.4f}")
    
    # SVM Test
    y_pred = svm.predict(X_test_scaled)
    svm_test_accuracy = np.mean(y_pred == y_test)
    svm_test_f1 = f1_score(y_test, y_pred, average='weighted')
    results['SVM'] = (svm_test_accuracy, svm_test_f1)
    print(f"SVM Test Accuracy: {svm_test_accuracy:.4f}, F1 Score: {svm_test_f1:.4f}")
    
    # Logistic Regression Test
    y_pred = lr.predict(X_test_scaled)
    lr_test_accuracy = np.mean(y_pred == y_test)
    lr_test_f1 = f1_score(y_test, y_pred, average='weighted')
    results['Logistic Regression'] = (lr_test_accuracy, lr_test_f1)
    print(f"Logistic Regression Test Accuracy: {lr_test_accuracy:.4f}, F1 Score: {lr_test_f1:.4f}")
    
    return results


def main():

    train_transforms = transforms.Compose(
        [transforms.Resize((20, 20)),
         transforms.ToTensor()])

    dataset_train = ImageFolder(
        'data/colors_train',
        transform=train_transforms,
    )

    dataset_test = ImageFolder(
        'data/colors_test',
        transform=train_transforms,
    )

    dataset_validate = ImageFolder(
        'data/colors_validation',
        transform=train_transforms,
    )

    dataloader_train = data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=32,
    )

    dataloader_test = data.DataLoader(
        dataset_test,
        shuffle=True,
        batch_size=32,
    )

    dataloader_validate = data.DataLoader(
        dataset_validate,
        shuffle=True,
        batch_size=32,
    )

    poolings = ['max_pooling', 'average_pooling']
    kernel_sizes = [3, 5]
    activation_functions = ['relu', 'sigmoid', 'elu', 'tanh']

    num_classes = len(dataset_train.classes)

    experiments = []
    for pooling in poolings:
        for kernel_size in kernel_sizes:
            for activation_function in activation_functions:
                experiment = {}
                model = ColorCNN(num_classes,
                                 pooling=pooling,
                                 kernel_size=kernel_size,
                                 activation_function=activation_function)

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                # training loop
                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)

                num_epochs = 16
                for epoch in range(num_epochs):
                    model.train()
                    running_loss = 0.0

                    for images, labels in dataloader_train:
                        images, labels = images.to(device), labels.to(device)

                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()

                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader_train):.4f}"
                    )

                # validation loop
                model.eval()
                correct = 0
                total = 0
                all_predicted = []
                all_labels = []

                with torch.no_grad():
                    for images, labels in dataloader_validate:
                        images, labels = images.to(device), labels.to(device)

                        outputs = model(images)
                        _, predicted = torch.max(outputs, 1)

                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        
                        all_predicted.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                val_accuracy = 100 * correct / total
                val_f1 = f1_score(all_labels, all_predicted, average='weighted')
                print(f"Validation Accuracy: {val_accuracy:.2f}%")
                print(f"Validation F1 Score: {val_f1:.4f}")
                
                experiment['model'] = 'CNN'
                experiment['architecture'] = 'layer 1: 16 filters; pooling; layer 2: 32 filters; pooling; layer 3: linear 64; layer 4: linear 6'
                experiment['pooling'] = pooling
                experiment['kernel_size'] = kernel_size
                experiment['activation_function'] = activation_function
                experiment['accuracy'] = correct / total
                experiment['f1_score'] = val_f1
                experiments.append(experiment)


    kernel_sizes = [3]
    activation_functions = ['relu', 'sigmoid']

    for pooling in poolings:
        for kernel_size in kernel_sizes:
            for activation_function in activation_functions:
                experiment = {}
                model = ColorCNN(num_classes,
                                 pooling=pooling,
                                 kernel_size=kernel_size,
                                 activation_function=activation_function)

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                # training loop
                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)

                num_epochs = 20
                for epoch in range(num_epochs):
                    model.train()
                    running_loss = 0.0

                    for images, labels in dataloader_train:
                        images, labels = images.to(device), labels.to(device)

                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()

                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader_train):.4f}"
                    )

                # validation loop
                model.eval()
                correct = 0
                total = 0
                all_predicted = []
                all_labels = []

                with torch.no_grad():
                    for images, labels in dataloader_validate:
                        images, labels = images.to(device), labels.to(device)

                        outputs = model(images)
                        _, predicted = torch.max(outputs, 1)

                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        
                        all_predicted.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                val_accuracy = 100 * correct / total
                val_f1 = f1_score(all_labels, all_predicted, average='weighted')
                print(f"Validation Accuracy: {val_accuracy:.2f}%")
                print(f"Validation F1 Score: {val_f1:.4f}")
                
                experiment['model'] = 'CNN'
                experiment['architecture'] = 'layer 1: 16 filters; pooling; layer 2: 32 filters; pooling; layer 3: linear 64; layer 4: linear 6'
                experiment['pooling'] = pooling
                experiment['kernel_size'] = kernel_size
                experiment['activation_function'] = activation_function
                experiment['accuracy'] = correct / total
                experiment['f1_score'] = val_f1
                experiments.append(experiment)


    df = pd.DataFrame(experiments)
    with pd.ExcelWriter('model_report_color_prediction.xlsx', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, index=False)

    experiment = {}
    model = ColorCNN(num_classes,
                      pooling='max_pooling',
                      kernel_size=3,
                      activation_function='relu')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training loop
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in dataloader_train:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader_train):.4f}"
        )

    # # validation loop
    model.eval()
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader_validate:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_accuracy = 100 * correct / total
    val_f1 = f1_score(all_labels, all_predicted, average='weighted')
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Validation F1 Score: {val_f1:.4f}")
    
    experiment['model'] = 'CNN'
    experiment['architecture'] = 'layer 1: 16 filters; pooling; layer 2: 32 filters; pooling; layer 3: linear 64; layer 4: linear 6'
    experiment['pooling'] = 'max_pooling'
    experiment['kernel_size'] = 3
    experiment['activation_function'] = 'relu'
    experiment['accuracy'] = correct / total
    experiment['f1_score'] = val_f1

    # testing loop for the best CNN model
    model = ColorCNN(num_classes,
                      pooling='max_pooling',
                      kernel_size=3,
                      activation_function='relu')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training loop
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 16
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in dataloader_train:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader_train):.4f}"
        )

    correct = 0
    total = 0
    all_predicted = []
    all_labels = []
    experiment = {}
    with torch.no_grad():
        for images, labels in dataloader_test:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_accuracy = 100 * correct / total
    val_f1 = f1_score(all_labels, all_predicted, average='weighted')
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Validation F1 Score: {val_f1:.4f}")
    
    experiment['model'] = 'CNN'
    experiment['architecture'] = 'layer 1: 16 filters; pooling; layer 2: 32 filters; pooling; layer 3: linear 64; layer 4: linear 6'
    experiment['pooling'] = 'max_pooling'
    experiment['kernel_size'] = 3
    experiment['activation_function'] = 'relu'
    experiment['accuracy'] = correct / total
    experiment['f1_score'] = val_f1
    print(experiment)



    # ============ SKLEARN MODELS ============
    print("\n" + "="*50)
    print("TRAINING SKLEARN MODELS")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data for sklearn models
    print("\nLoading training data...")
    X_train, y_train = load_images_to_array(dataloader_train, device)
    print(f"Training data shape: {X_train.shape}")
    
    print("Loading validation data...")
    X_val, y_val = load_images_to_array(dataloader_validate, device)
    print(f"Validation data shape: {X_val.shape}")
    
    print("Loading test data...")
    X_test, y_test = load_images_to_array(dataloader_test, device)
    print(f"Test data shape: {X_test.shape}")
    
    # Train sklearn models
    sklearn_val_results, scaler, knn, svm, lr = train_sklearn_models(X_train, y_train, X_val, y_val)
    
    print("\n" + "="*50)
    print("TESTING SKLEARN MODELS")
    print("="*50)
    
    sklearn_test_results = evaluate_sklearn_on_test(knn, svm, lr, X_test, y_test, scaler)
    print(sklearn_test_results)


if __name__ == "__main__":
    main()