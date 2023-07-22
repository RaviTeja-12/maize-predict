import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

# Data Preprocessing

encoder = OneHotEncoder()
encoder.fit([[0], [1], [2], [3]])

# Loading the data

data = []
result = []

def load_images_from_folder(folder, label):
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(folder, filename))
            img = img.resize((50, 50))
            img = np.array(img)
            if img.shape == (50, 50, 3):
                data.append(img)
                result.append(label)

load_images_from_folder(r'E:\data\Blight', 0)
load_images_from_folder(r'E:\data\Common_Rust', 1)
load_images_from_folder(r'E:\data\Gray_Leaf_Spot', 2)
load_images_from_folder(r'E:\data\Healthy', 3)

result = np.array(result)
result = encoder.transform(result.reshape(-1, 1)).toarray()

data = np.array(data)

# Splitting data into training and testing data

x_train, x_test, y_train, y_test = train_test_split(data, result, test_size=0.30, shuffle=True, random_state=1)

# Creating the model

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='softmax'))

# Compiling the model

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), verbose=1)

model.save('MaizeML.h5')