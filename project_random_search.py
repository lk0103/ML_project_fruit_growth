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


def preprocess_images(file_paths, target_size=(224, 224)):
    images = []
    i = 0
    for file_path in file_paths:
        img = Image.open(file_path)
        img = img.resize(target_size, Image.BILINEAR)

        #normalized pixel values
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
        if (i % 500 == 0):
            print("processing image ", i)
        i += 1
    return np.array(images)

print(X_train[:10])
print(y_train[:10])

X_train_processed = preprocess_images(X_train)


X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_processed, y_train, test_size=0.7, random_state=seed, stratify=y_train
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



def display_random_images(images, labels, num_images=10):
    random_indices = np.random.choice(len(images), num_images, replace=False)

    fig, axes = plt.subplots(1, num_images, figsize=(50, 50))

    for i, idx in enumerate(random_indices):
        axes[i].imshow(images[idx])
        axes[i].set_title(labels[idx])
        axes[i].axis('off')

    plt.show()

display_random_images(X_train_split, y_train_split)


def random_search_one_layer():
    global model

    print("\n\n--------------------------------------------")
    print("using VGG16 model (excluding the top layer)")
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for i in range(20):
        print("\n --------------------new model")
        model = random_model(base_model)
        if i < 4:
            continue
        model.fit(X_train_split, y_train_split, epochs=10, batch_size=32,
              validation_data=(X_val_split, y_val_split))

        val_loss, val_acc = model.evaluate(X_val_split, y_val_split)
        print(f"Validation Accuracy: {val_acc * 100:.2f}%")



def random_model(base_model):
    flatten_or_GAP = random.randint(0, 3)
    drop_out = random.randint(0, 2)
    drop_out_value_1 = random.choice([0.5, 0.6, 0.7, 0.8])
    drop_out_value_2 = random.choice([0.2, 0.3, 0.4])
    batch_norm = True if random.randint(0, 3) < 1 else False
    optimizer = random.randint(0, 2)
    units = random.choice([64, 128, 256])
    two_layers = False if random.randint(0, 8) < 8 else True
    learning_rate = 0.0001 if random.randint(0, 4) < 4 else 0.001


    model = Sequential()
    model.add(base_model)

    if (flatten_or_GAP <= 2):
        print("model.add(Flatten())")
        model.add(Flatten())
    else:
        print("model.add(GlobalAveragePooling2D())")
        model.add(GlobalAveragePooling2D())

    if batch_norm:
        print("model.add(BatchNormalization())")
        model.add(BatchNormalization())
    if (drop_out > 0):
        print("model.add(Dropout(" + str(drop_out_value_1) + "))")
        model.add(Dropout(drop_out_value_1))
    print("model.add(Dense(" + str(units) + ", activation='relu'))")
    model.add(Dense(units, activation='relu'))

    if (drop_out > 1):
        print("model.add(Dropout(" + str(drop_out_value_2) + "))")
        model.add(Dropout(drop_out_value_2))
    if two_layers:
        print("model.add(Dense(" + str(units / 2) + ", activation='relu'))")
        model.add(Dense(units / 2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    if optimizer < 2:
        print("optimazer Adam(learning_rate=" + str(learning_rate) + ")")
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    else:
        print("optimazer RMSprop(learning_rate=" + str(learning_rate) + ")")
        model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    return model


random_search_one_layer()






