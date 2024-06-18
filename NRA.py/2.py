import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import os
import cv2

# Cargar imágenes generadas
images = []
labels = []

for i, filename in enumerate(os.listdir('generated_images')):
    img = cv2.imread(os.path.join('generated_images', filename), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        images.append(img)
        labels.append(i % 10)  # Etiquetas ficticias, puedes ajustarlas según tu necesidad

images = np.array(images).astype('float32') / 255.0  # Normalizar imágenes
images = images.reshape((images.shape[0], -1))  # Aplanar las imágenes para que sean unidimensionales
labels = np.array(labels)

# Crear el modelo de red neuronal
model = Sequential()
model.add(Dense(60, input_dim=25, activation='relu'))  # Capa Oculta 1, activación ReLU
model.add(Dense(40, activation='relu'))                # Capa Oculta 2, activación ReLU
model.add(Dense(32, activation='relu'))                # Capa Oculta 3, activación ReLU
model.add(Dense(10, activation='softmax'))             # Capa de Salida, activación Softmax para clasificación multiclase

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(images, labels, epochs=100, batch_size=1, verbose=1)

# Guardar el modelo en dos partes separadas: JSON y pesos en HDF5
# Serializar el modelo a JSON
model_json = model.to_json()
with open("model1111.json", "w") as json_file:
    json_file.write(model_json)

# Serializar los pesos a HDF5
model.save_weights("model1111.weights.h5")
print("Modelo Guardado!")
