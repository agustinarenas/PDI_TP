import cv2
import numpy as np
import matplotlib.pyplot as plt

def ecualizacion_local_pixel_a_pixel(img, window_size):
    """
    Aplica ecualización local pixel a pixel usando una ventana deslizante.

    Parámetros:
        img (np.ndarray): Imagen en escala de grises.
        window_size (tuple): Tamaño de la ventana (ancho, alto).

    Retorna:
        np.ndarray: Imagen ecualizada localmente.
    """
    #Restriccion para que el tamaño de la ventana sea impar
    if window_size[0] % 2 == 0 or window_size[1] % 2 == 0:
        raise ValueError("El tamaño de la ventana debe ser impar en ambas dimensiones (ancho y alto).")
    height, width = img.shape
    window_w, window_h = window_size
    pad_w = window_w // 2
    pad_h = window_h // 2

    # Aplicar padding replicado
    img_padded = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, borderType=cv2.BORDER_REPLICATE)

    # Crear imagen de salida
    img_ecualizada = np.zeros_like(img)

    # Recorrer cada píxel de la imagen original
    for i in range(height):
        for j in range(width):
            # Extraer ventana centrada en (i, j)
            ventana = img_padded[i:i + window_h, j:j + window_w]
            # Ecualizar la ventana_
            ventana_eq = cv2.equalizeHist(ventana)
            # Tomar valor central de la ventana ecualizada
            centro_y = window_h // 2
            centro_x = window_w // 2
            # Asignar el valor central de la ventana ecualizada a la imagen de salida
            img_ecualizada[i, j] = ventana_eq[centro_y, centro_x]

    return img_ecualizada

# Cargar imagen en escala de grises
#img = cv2.imread('Procesamiento de imagenes 1/Tp_Pdi/Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('TP1/Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)

# Definir tamaño de ventana y aplicar ecualización local
ventana_w = 21
ventana_h = 17
eq_ss = ecualizacion_local_pixel_a_pixel(img, (ventana_w, ventana_h))

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