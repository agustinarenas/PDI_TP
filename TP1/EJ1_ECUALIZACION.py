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
  
    # El pad se calcula para cubrir el faltante en altura y ancho necesario para que la ventana recorra sin problemas
    # forzando a que la imagen sea un multiplo entero de la ventana en altura y ancho
    pad_h = (window_h - (height % window_h)) % window_h
    pad_w = (window_w - (width % window_w)) % window_w

    # Aplicar padding replicado
    img_padded = cv2.copyMakeBorder(img_ecualizada, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REPLICATE)

    # Recorrer la imagen en bloques y aplicar ecualización
    for i in range(window_h//2, (height + pad_h) - window_h//2, window_h):
        for j in range(window_w//2, (width + pad_w) - window_w//2, window_w):
            block = img_padded[i:i + window_h, j:j + window_w]
            ecualizado = cv2.equalizeHist(block)
            img_padded[i:i + window_h, j:j + window_w] = ecualizado
    img_ecualizada = img_padded[0:height, 0:width]
    return img_ecualizada

# Cargar imagen en escala de grises
img = cv2.imread('PDI_TP/TP1/Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)
#img = cv2.imread('PDI_TP/TP1/Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)

# Definir tamaño de ventana y aplicar ecualización local
ventana_w = 18
ventana_h = 18
eq_ss = ecualizacion_local(img, (ventana_w, ventana_h))

# Suavizar la imagen para eliminar posibles artefactos
eq = cv2.medianBlur(eq_ss, 3)

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
