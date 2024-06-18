import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense, Flatten

class SequentialClassifier:
    def __init__(self, model_json_path=None, weights_path=None, num_classes=None, input_shape=(5, 5)):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

        if model_json_path and weights_path:
            self.load_model(model_json_path, weights_path)

    def load_model(self, model_json_path, weights_path):
        # Cargar modelo desde archivo JSON
        with open(model_json_path, 'r') as json_file:
            loaded_model_json = json_file.read()
            self.model = model_from_json(loaded_model_json)

        # Cargar pesos al modelo
        self.model.load_weights(weights_path)

        # Compilar el modelo después de cargar los pesos
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        # Obtener el número de clases del modelo cargado si no se especificó previamente
        if not self.num_classes:
            self.num_classes = self.model.layers[-1].output_shape[-1]

    def preprocess_image(self, img):
        img_resized = cv2.resize(img, self.input_shape)
        image_array = np.asarray(img_resized).astype(np.float32) / 255.0
        flattened_image = image_array.flatten()  # Aplanar la imagen a un vector unidimensional
        return np.expand_dims(flattened_image, axis=0)

    def get_prediction(self, img, draw=True, pos=(50, 50), scale=2, color=(0, 255, 0)):
        input_data = self.preprocess_image(img)

        prediction = self.model.predict(input_data)
        index_val = np.argmax(prediction)

        if draw:
            cv2.putText(img, str(index_val), pos, cv2.FONT_HERSHEY_COMPLEX, scale, color, 2)

        return list(prediction[0]), index_val

# Ejemplo de uso
model_json_path = 'model1111.json'
model_weights_path = 'model1111.weights.h5'
num_classes = 10  # Asegúrate de especificar el número correcto de clases del modelo cargado

classifier = SequentialClassifier(model_json_path, model_weights_path, num_classes=num_classes, input_shape=(5, 5))

# Simular la predicción con una imagen de prueba
img_test = np.random.randint(0, 256, (5, 5), dtype=np.uint8)
prediction, index = classifier.get_prediction(img_test, draw=False)
print(prediction)
