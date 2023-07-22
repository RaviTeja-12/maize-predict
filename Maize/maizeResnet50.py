import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model, load_model

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
    img = img.resize((50, 50))  # Resize to match ResNet50 input shape
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
if os.path.exists('maizeResnet50.h5'):
    # Load the trained model
    model = load_model('maizeResnet50.h5')
else:
    # Creating the ResNet50 model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(50, 50, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(4, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compiling the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Training the model
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), verbose=1)
    # Save the trained model
    model.save('maizeResnet50.h5')

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
