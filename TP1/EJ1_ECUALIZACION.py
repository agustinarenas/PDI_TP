import cv2
import numpy as np
import matplotlib.pyplot as plt

def ecualizacion_local(img, window_size):
    """
    Realiza ecualización local de histograma sobre una imagen en escala de grises.

    Parámetros:
        img (np.ndarray): Imagen de entrada en escala de grises.
        window_size (tuple): Tamaño de la ventana (ancho, alto) para la ecualización local.

    Retorna:
        np.ndarray: Imagen resultante con ecualización local aplicada.
    """
    # Copiar la imagen de entrada
    img_ecualizada = img.copy()
    height, width = img.shape
    window_w, window_h = window_size
    pad = window_h // 2  # Asume ventana cuadrada para el padding

    # Aplicar padding replicado
    img_padded = cv2.copyMakeBorder(
        img_ecualizada, pad, pad, pad, pad, borderType=cv2.BORDER_REPLICATE
    )

    # Recorrer la imagen en bloques y aplicar ecualización
    for i in range(0, height - window_h, window_h):
        for j in range(0, width - window_w, window_w):
            block = img_padded[i:i + window_h, j:j + window_w]
            ecualizado = cv2.equalizeHist(block)
            img_ecualizada[i:i + window_h, j:j + window_w] = ecualizado

    # Suavizar la imagen para eliminar posibles artefactos
    img_ecualizada = cv2.medianBlur(img_ecualizada, 3)

    return img_ecualizada

# Cargar imagen en escala de grises
img = cv2.imread('TP1/Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)

# Definir tamaño de ventana y aplicar ecualización local
tam_ventana = 18
eq = ecualizacion_local(img, (tam_ventana, tam_ventana))


# Crear figura y ejes
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)

# Mostrar la imagen original
axes[0].imshow(img, cmap='gray')
axes[0].set_title("Imagen original")

# Mostrar la imagen ecualizada
axes[1].imshow(eq, cmap='gray')
axes[1].set_title("Imagen ecualizada")

# Mostrar la figura
plt.tight_layout()
plt.show()

#Analisis
"""
Probamos con diferentes tamaños de ventanas y podemos concluir que mientras mas chica sea la ventana mas claro se van haciendo los cuadrados,
pensamos que mientras mas chicas las ventanas hace los procesos mas detallados por lo tanto tarda mas pero se obtiene mas detallado, sin embargo,
no hace falta que las ventanas sean tan chicas en este caso, ya que con una ventana de 7x7 o 9x9 se logra ver bien los detalles escondidos
"""
