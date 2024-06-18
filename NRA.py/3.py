import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
import cv2
import os

# Cargar el modelo desde el archivo JSON
with open('model1111.json', 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)

# Cargar los pesos en el modelo
model.load_weights('model1111.weights.h5')

# Compilar el modelo
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Cargar imágenes para verificación
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

# Evaluar el modelo
loss, accuracy = model.evaluate(images, labels)
print(f'Pérdida: {loss}, Precisión: {accuracy}')
