import os
import random

from keras.preprocessing.image import load_img, img_to_array, array_to_img
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.optimizers import RMSprop
from keras.applications import ResNet50
from keras.layers import Dropout, BatchNormalization, GlobalAveragePooling2D
from PIL import Image, ImageEnhance
import cv2

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint





def load_training_data():
    global X_train, y_train, seed
    dataset_path = 'C:/Users/lucin/OneDrive/Desktop/strojaky/project/archive/fruits-360_dataset/fruits-360/Training2/'
    directories = [ 'bush', 'tree', 'vines']
    X_train = []
    y_train = []
    for directory in directories:
        label = 'tree' if 'tree' in directory else 'not_tree'
        directory_path = os.path.join(dataset_path, directory)

        files = os.listdir(directory_path)

        X_train.extend([os.path.join(directory_path, file) for file in files])
        y_train.extend([label] * len(files))
    seed = 42
    combined = list(zip(X_train, y_train))
    random.seed(seed)
    random.shuffle(combined)
    X_train[:], y_train[:] = zip(*combined)
    return X_train, y_train


X_train, y_train = load_training_data()


def preprocess_images_base(file_paths, target_size=(224, 224)):
    images = []
    i = 0
    for file_path in file_paths:
        img = Image.open(file_path)
        img = img.resize((224, 224), Image.BILINEAR)

        #normalized pixel values
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
        if (i % 500 == 0):
            print("processing image ", i)
        i += 1
    return np.array(images)

def preprocess_images_grayscale(file_paths, target_size=(224, 224)):
    images = []
    i = 0
    for file_path in file_paths:
        img = cv2.imread(file_path)
        img = cv2.resize(img, target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #normalized pixel values
        img_array = img_to_array(img) / 255.0
        if (img_array.shape[2] == 1):
            img_array = np.repeat(img_array, 3, axis=2)
        images.append(img_array)
        if (i % 500 == 0):
            print("processing image ", i)
        i += 1
    return np.array(images)

def preprocess_images_equalize_histogram(file_paths, target_size=(224, 224)):
    images = []
    i = 0
    for file_path in file_paths:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, target_size)
        img = cv2.equalizeHist(img)

        #normalized pixel values
        img_array = img_to_array(img) / 255.0
        if (img_array.shape[2] == 1):
            img_array = np.repeat(img_array, 3, axis=2)
        images.append(img_array)
        if (i % 500 == 0):
            print("processing image ", i)
        i += 1
    return np.array(images)


def preprocess_images_clahe(file_paths, target_size=(224, 224)):
    images = []
    i = 0
    for file_path in file_paths:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, target_size)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        #normalized pixel values
        img_array = img_to_array(img) / 255.0
        if (img_array.shape[2] == 1):
            img_array = np.repeat(img_array, 3, axis=2)
        images.append(img_array)
        if (i % 500 == 0):
            print("processing image ", i)
        i += 1
    return np.array(images)


def preprocess_images_gaussian_blur(file_paths, target_size=(224, 224)):
    images = []
    i = 0
    for file_path in file_paths:
        img = cv2.imread(file_path)
        img = cv2.resize(img, target_size)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        #normalized pixel values
        img_array = img_to_array(img) / 255.0
        if (img_array.shape[2] == 1):
            img_array = np.repeat(img_array, 3, axis=2)
        images.append(img_array)
        if (i % 500 == 0):
            print("processing image ", i)
        i += 1
    return np.array(images)

def preprocess_images_edges(file_paths, target_size=(224, 224)):
    images = []
    i = 0
    for file_path in file_paths:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, target_size)
        img = cv2.Canny(img, 50, 150)

        #normalized pixel values
        img_array = img_to_array(img) / 255.0
        if (img_array.shape[2] == 1):
            img_array = np.repeat(img_array, 3, axis=2)
        images.append(img_array)
        if (i % 500 == 0):
            print("processing image ", i)
        i += 1
    return np.array(images)


def display_random_images(images, labels, num_images=10):
    random_indices = np.random.choice(len(images), num_images, replace=False)

    fig, axes = plt.subplots(1, num_images, figsize=(50, 50))

    for i, idx in enumerate(random_indices):
        axes[i].imshow(images[idx])
        axes[i].set_title(labels[idx])
        axes[i].axis('off')

    plt.show()

def create_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def cross_validation(X_train, y_train):
    n_splits = 4
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for train_index, val_index in skf.split(X_train, y_train):
        print("split")
        X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
        y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]

        model = create_model()

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

        model.fit(
            datagen.flow(X_train_cv, y_train_cv, batch_size=32),
            steps_per_epoch=len(X_train_cv) // 32,
            epochs=5,
            validation_data=(X_val_cv, y_val_cv),
            callbacks=[checkpoint]
        )

        model.load_weights('best_model.h5')

        y_pred = model.predict(X_val_cv)
        y_pred_binary = (y_pred > 0.5).astype(int)

        accuracy_scores.append(accuracy_score(y_val_cv, y_pred_binary))
        precision_scores.append(precision_score(y_val_cv, y_pred_binary))
        recall_scores.append(recall_score(y_val_cv, y_pred_binary))
        f1_scores.append(f1_score(y_val_cv, y_pred_binary))
    print(f"Average Accuracy: {np.mean(accuracy_scores)}")
    print(f"Average Precision: {np.mean(precision_scores)}")
    print(f"Average Recall: {np.mean(recall_scores)}")
    print(f"Average F1 Score: {np.mean(f1_scores)}")


def test_preprocessing_technique(X_train_processed):
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_processed, y_train, test_size=0.8, random_state=seed, stratify=y_train
    )

    X_train_split = np.array(X_train_split)
    X_val_split = np.array(X_val_split)
    y_train_split = np.array(y_train_split)
    y_val_split = np.array(y_val_split)
    y_train_split = (y_train_split == 'tree').astype(int)
    y_val_split = (y_val_split == 'tree').astype(int)

    print(X_train_split[0], "\n", X_val_split[0], "\n", y_train_split[0], "\n", y_val_split[0])
    print(X_train_split.shape)
    print(X_val_split.shape)
    print(y_train_split.shape)
    print(y_val_split.shape)

    display_random_images(X_train_split, y_train_split)

    cross_validation(X_train_split, y_train_split)


print(X_train[:10])
print(y_train[:10])


print("----------------------------------------------\nbase preprocesing")
X_train_processed = preprocess_images_base(X_train)
test_preprocessing_technique(X_train_processed)

print("----------------------------------------------\ngrayscale preprocesing")
X_train_processed = preprocess_images_grayscale(X_train)
test_preprocessing_technique(X_train_processed)

print("----------------------------------------------\ngaussian blur preprocesing")
X_train_processed = preprocess_images_gaussian_blur(X_train)
test_preprocessing_technique(X_train_processed)

print("----------------------------------------------\nequalize histogram preprocesing")
X_train_processed = preprocess_images_equalize_histogram(X_train)
test_preprocessing_technique(X_train_processed)

print("----------------------------------------------\nclahe preprocesing - grayscale")
X_train_processed = preprocess_images_clahe(X_train)
test_preprocessing_technique(X_train_processed)

print("----------------------------------------------\nedges preprocesing")
X_train_processed = preprocess_images_edges(X_train)
test_preprocessing_technique(X_train_processed)












