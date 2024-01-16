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
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix



from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint



def load_training_data(dir_path):
    global X_train, y_train, seed
    dataset_path = dir_path
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


def train_model(X_train, y_train, X_val, y_val, model_file_name):

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

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
    model_file_name + '.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
    )

    model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        steps_per_epoch=len(X_train) // 32,
        epochs=12,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )

    y_pred = model.predict(X_val)
    y_pred_binary = (y_pred > 0.5).astype(int)

    accuracy_scores = accuracy_score(y_val, y_pred_binary)
    precision_scores = precision_score(y_val, y_pred_binary)
    recall_scores = recall_score(y_val, y_pred_binary)
    f1_scores = f1_score(y_val, y_pred_binary)

    print(f"Accuracy: {accuracy_scores}")
    print(f"Precision: {precision_scores}")
    print(f"Recall: {recall_scores}")
    print(f"F1 Score: {f1_scores}")

    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")

    return model

def test_preprocessing_technique(X_train_processed, model_file):
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_processed, y_train, test_size=0.78, random_state=seed, stratify=y_train
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

    return train_model(X_train_split, y_train_split, X_val_split, y_val_split, model_file)


def testing_results(model, X_test, X_test_prep, y_test, mislabeled_file):
    y_pred = model.predict(X_test_prep)
    y_pred_binary = (y_pred > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred_binary)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    conf_matrix = confusion_matrix(y_test, y_pred_binary)
    print("Confusion Matrix:")
    print(conf_matrix)

    y_pred_binary = np.ravel(y_pred_binary).tolist()
    print(y_test)
    print(y_pred_binary)
    misclassified_indices = np.where(y_test != y_pred_binary)[0]

    output_file = mislabeled_file
    with open(output_file, 'w') as f:
        for index in misclassified_indices:
            f.write(f"Image: {X_test[index]}, True Label: {y_test[index]}, Predicted Label: {y_pred_binary[index]}\n")

    print(f"Misclassified images saved to {mislabeled_file}")

X_train, y_train = load_training_data('C:/Users/lucin/OneDrive/Desktop/strojaky/project/archive/fruits-360_dataset/fruits-360/Training2/')
X_test, y_test = load_training_data('C:/Users/lucin/OneDrive/Desktop/strojaky/project/archive/fruits-360_dataset/fruits-360/Test/')

X_test = np.array(X_test)
y_test = np.array(y_test)
y_test = (y_test == 'tree').astype(int)


print(X_train[:10])
print(y_train[:10])


print("----------------------------------------------\ngaussian blur preprocesing")
print("gaussian blur training")
X_train_processed = preprocess_images_gaussian_blur(X_train)
print("gaussian blur testing")
X_test_processed = preprocess_images_gaussian_blur(X_test)
model_gaussian_blur = test_preprocessing_technique(X_train_processed, 'gaussian_blur_model')
testing_results(model_gaussian_blur, X_test, X_test_processed, y_test, 'gaussian_blur_mislabeled.txt')

print("----------------------------------------------\nequalize histogram preprocesing")
print("equalize histogram training")
X_train_processed = preprocess_images_equalize_histogram(X_train)
print("equalize histogram testing")
X_test_processed = preprocess_images_equalize_histogram(X_test)
model_equalize_histogram = test_preprocessing_technique(X_train_processed, 'equalize_histogram_model')
testing_results(model_equalize_histogram, X_test, X_test_processed, y_test, 'equalize_histogram_mislabeled.txt')


X_test_multiple, y_test_multiple = load_training_data('C:/Users/lucin/OneDrive/Desktop/strojaky/project/archive/fruits-360_dataset/fruits-360/test-multiple_fruits/')

X_test_multiple = np.array(X_test_multiple)
y_test_multiple = np.array(y_test_multiple)
y_test_multiple = (y_test_multiple == 'tree').astype(int)

model_gaussian_blur = create_model()
model_gaussian_blur.load_weights('gaussian_blur_model.h5')
X_test_multiple_processed = preprocess_images_gaussian_blur(X_test_multiple)
testing_results(model_gaussian_blur, X_test_multiple, X_test_multiple_processed, y_test_multiple, 'gaussian_blur_mislabeled_multiple.txt')


model_equalize_histogram = create_model()
model_equalize_histogram.load_weights('equalize_histogram_model.h5')
X_test_multiple_processed = preprocess_images_equalize_histogram(X_test_multiple)
testing_results(model_equalize_histogram, X_test_multiple, X_test_multiple_processed, y_test_multiple, 'equalize_histogram_mislabeled_multiple.txt')









