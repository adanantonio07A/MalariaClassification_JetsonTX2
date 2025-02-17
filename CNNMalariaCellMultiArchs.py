import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import time
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Variables globales
img_dir = "Malaria/cell_images"
img_size = 64
_archs = ["Arch1", "Arch2", "Arch3", "Arch4", "Arch5", "Arch6"]
#arch 1: 32, 32
#arch 2: 32, 32, 32
#arch 3: 48, 48
#arch 4: 48, 48, 48
#arch 5: 64, 64
#arch 6: 64, 64, 64
_epochs = 10

# Función para cargar imágenes
def load_img_data(path):
    image_files = glob.glob(os.path.join(path, "parasitized/*.png")) + \
                  glob.glob(os.path.join(path, "uninfected/*.png"))
    X, y = [], []
    for image_file in image_files:
        label = 0 if "uninfected" in image_file else 1
        img_arr = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img_arr, (img_size, img_size))
        X.append(img_resized)
        y.append(label)
    return X, y

# Definición de arquitecturas de modelos
def create_model(arch_type):
    model = Sequential()
    if arch_type in ["Arch1", "Arch2"]:
        model.add(Conv2D(32, (3, 3), input_shape=(img_size, img_size, 1), activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        if arch_type == "Arch2":
            model.add(Conv2D(32, (3, 3), activation="relu"))
            model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(32, activation="relu"))
    elif arch_type in ["Arch3", "Arch4"]:
        model.add(Conv2D(48, (3, 3), input_shape=(img_size, img_size, 1), activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(48, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        if arch_type == "Arch4":
            model.add(Conv2D(48, (3, 3), activation="relu"))
            model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(48, activation="relu"))
    elif arch_type in ["Arch5", "Arch6"]:
        model.add(Conv2D(64, (3, 3), input_shape=(img_size, img_size, 1), activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        if arch_type == "Arch6":
            model.add(Conv2D(64, (3, 3), activation="relu"))
            model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
    #model.add(Flatten())
    #model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# Función para plotear el promedio de precisión por arquitectura
def plot_architecture_average(fold_avg_acc_by_arch, fold_avg_val_acc_by_arch):
    epochs_range = range(1, _epochs + 1)
    plt.figure(figsize=(12, 6))
    
    for arch_idx, arch_name in enumerate(_archs):
        plt.plot(epochs_range, fold_avg_acc_by_arch[arch_idx], label=f"{arch_name} - Train Avg Accuracy")
        plt.plot(epochs_range, fold_avg_val_acc_by_arch[arch_idx], linestyle="--", label=f"{arch_name} - Val Avg Accuracy")

    plt.title("Average Accuracy per Architecture")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("average_performance_architectures.png")
    plt.show()

# Cargar los datos
X, y = load_img_data(img_dir)
X = np.array(X).reshape(-1, img_size, img_size, 1) / 255
y = np.array(y)

# Validación cruzada y entrenamiento de modelos
kf = StratifiedKFold(n_splits=5)

# Almacenar precisión de entrenamiento y validación promedio para cada arquitectura en cada época
fold_avg_acc_by_arch = [np.zeros(_epochs) for _ in _archs]
fold_avg_val_acc_by_arch = [np.zeros(_epochs) for _ in _archs]

for train_index, test_index in kf.split(X, y):
    for arch_idx, arch in enumerate(_archs):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = create_model(arch)
        history = model.fit(X_train, y_train, batch_size=32, epochs=_epochs, validation_split=0.2, verbose=0)
        
        # Sumar las métricas de precisión y validación para luego promediar
        fold_avg_acc_by_arch[arch_idx] += np.array(history.history['accuracy'])
        fold_avg_val_acc_by_arch[arch_idx] += np.array(history.history['val_accuracy'])

# Promediar las métricas de precisión para cada arquitectura
fold_avg_acc_by_arch = [acc / kf.get_n_splits() for acc in fold_avg_acc_by_arch]
fold_avg_val_acc_by_arch = [val_acc / kf.get_n_splits() for val_acc in fold_avg_val_acc_by_arch]

# Graficar la precisión promedio de cada arquitectura
plot_architecture_average(fold_avg_acc_by_arch, fold_avg_val_acc_by_arch)
