import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import load_model

# Data Preprocessing

encoder = OneHotEncoder()
encoder.fit([[0], [1], [2], [3]])

# Loading the data

data = []
paths = []
result = []

for r, d, f in os.walk(r"E:\data\Blight"):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    img = img.resize((50, 50))  # Resize to match AlexNet input shape
    img = np.array(img)
    if img.shape == (50, 50, 3):
        data.append(img)
        result.append(encoder.transform([[0]]).toarray())

paths = []
for r, d, f in os.walk(r"E:\data\Common_Rust"):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    img = img.resize((50, 50))
    img = np.array(img)
    if img.shape == (50, 50, 3):
        data.append(img)
        result.append(encoder.transform([[1]]).toarray())

paths = []
for r, d, f in os.walk(r"E:\data\Gray_Leaf_Spot"):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    img = img.resize((50, 50))
    img = np.array(img)
    if img.shape == (50, 50, 3):
        data.append(img)
        result.append(encoder.transform([[2]]).toarray())

paths = []
for r, d, f in os.walk(r"E:\data\Healthy"):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    img = img.resize((50, 50))
    img = np.array(img)
    if img.shape == (50, 50, 3):
        data.append(img)
        result.append(encoder.transform([[3]]).toarray())

result = np.array(result)
result = result.reshape(-1, 4)

data = np.array(data)

# Splitting data into training and testing data

x_train, x_test, y_train, y_test = train_test_split(data, result, test_size=0.25, shuffle=True, random_state=1)

# Check if the trained model file exists
if os.path.exists('maizeAlexnet.h5'):
    # Load the trained model
    model = load_model('maizeAlexnet.h5')
else:
    # Creating the AlexNet model
    model = Sequential()
    model.add(Conv2D(96, (3, 3), strides=(1, 1), activation='relu', input_shape=(50, 50, 3)))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(Conv2D(384, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Conv2D(384, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    # Compiling the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Training the model
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), verbose=1)
    # Save the trained model
    model.save('maizeAlexnet.h5')

# Function to determine class names based on labels

def names(number):
    if number == 0:
        return "It's a leaf with disease BLIGHT"
    elif number == 1:
        return "It's a leaf with disease Common Rust"
    elif number == 2:
        return "It's a leaf with disease Gray_Leaf_Spot"
    elif number == 3:
        return "It's a Healthy leaf"
