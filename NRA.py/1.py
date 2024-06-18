import numpy as np
import os
import cv2

# Crear carpeta para almacenar las imágenes generadas
if not os.path.exists('generated_images'):
    os.makedirs('generated_images')

# Generar 10 imágenes de 5x5 píxeles en escala de grises
for i in range(10):
    image = np.random.randint(0, 256, (5, 5), dtype=np.uint8)
    cv2.imwrite(f'generated_images/image_{i}.png', image)
    print(f'Imagen image_{i}.png generada')
