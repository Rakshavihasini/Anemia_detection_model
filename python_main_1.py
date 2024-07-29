import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = image.load_img(os.path.join(folder, filename), target_size=(224, 224))
        img = image.img_to_array(img)
        img = img/255.0 
        images.append(img)
    return np.array(images)

anemic_images = load_images('/content/anemia')
non_anemic_images = load_images('/content/non_anemia')

labels_anemic = np.ones(len(anemic_images))
labels_non_anemic = np.zeros(len(non_anemic_images))

images = np.concatenate([anemic_images, non_anemic_images], axis=0)
labels = np.concatenate([labels_anemic, labels_non_anemic], axis=0)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5)) 
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))