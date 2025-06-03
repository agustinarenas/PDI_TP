import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# PROCESAR IMAGEN - OBTENER FONDO DE APOYO DE LA RESISTENCIA
carpeta = 'PDI_TP/TP2/Resistencias'
nombre_archivo = f'R8_a_out.jpg'
ruta = os.path.join(carpeta, nombre_archivo)

# Cargar imagen
img = cv2.imread(ruta)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

from PIL import Image

# --- Cargar imagen con Pillow ---
imagen_pil = Image.open(ruta).convert("RGB")  # asegurarse que está en RGB

# --- Aplicar dithering adaptativo con 16 colores ---
imagen_dither = imagen_pil.convert(
    mode="P", 
    palette=Image.ADAPTIVE, 
    colors=256, 
    dither=Image.FLOYDSTEINBERG)  # Podés cambiar a Image.NONE para probar sin dithering

# --- Convertir de nuevo a RGB para usar con OpenCV ---
imagen_dither_rgb = imagen_dither.convert("RGB")

# --- Pasar a NumPy (para OpenCV) ---
imagen_cv2 = np.array(imagen_dither_rgb)
imagen_cv2 = cv2.cvtColor(imagen_cv2, cv2.COLOR_RGB2BGR)

# --- Mostrar con OpenCV ---
cv2.imshow("Dithered con 16 colores", imagen_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()




'''
lower_blue = np.array([105, 50, 50])
upper_blue = np.array([135, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
mask_invertida = cv2.bitwise_not(mask)

# Marrón-amarillo claro del cuerpo
lower_rs = np.array([14, 120, 140])
upper_rs = np.array([15, 160, 185])
mask = cv2.inRange(hsv, lower_rs, upper_rs)
mask_invertida = cv2.bitwise_not(mask)


# BORDES
f = mask_invertida #Aop #Fe
se = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
f_mg = cv2.morphologyEx(f, cv2.MORPH_GRADIENT, se)

plt.imshow(cv2.cvtColor(f_mg, cv2.COLOR_BGR2RGB))
plt.title("Imágen de la resistencia amortiguada")
plt.axis("off")
plt.show()


# Definir rango azul
lower_blue = np.array([105, 50, 50])
upper_blue = np.array([135, 255, 255])

# Máscara del azul
mask = cv2.inRange(hsv, lower_blue, upper_blue)
mask_invertida = cv2.bitwise_not(mask)

# APERTURA
A = mask_invertida
B = cv2.getStructuringElement(cv2.MORPH_RECT, (75,75))
Aop = cv2.morphologyEx(A, cv2.MORPH_OPEN, B)

L = 5
F = Aop
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (L,L))
Aop = cv2.erode(A, kernel, iterations=10)

# Plot
plt.imshow(cv2.cvtColor(Aop, cv2.COLOR_BGR2RGB))
plt.title("Imágen - Resistencia")
plt.axis("off")
plt.show()


# Aop binaria 
binaria = Aop.copy()

# Componentes conectadas
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binaria, connectivity=8)

# Convertimos a BGR para dibujar
output = cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)

# Inicializamos extremos del bounding box fusionado
min_x, min_y = float('inf'), float('inf')
max_x, max_y = 0, 0

for i in range(1, num_labels):  # ignoramos fondo
    x, y, w, h, area = stats[i]

    # Actualizamos los extremos
    min_x = min(min_x, x)
    min_y = min(min_y, y)
    max_x = max(max_x, x + w)
    max_y = max(max_y, y + h)

# Dibujamos el bounding box fusionado
cv2.rectangle(output, (min_x, min_y), (max_x, max_y), (0, 255, 255), 2)

# Mostrar resultado
cv2.imshow("Bounding Box fusionado", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------
# Crear imagen vacía con mismo tamaño que hsv
hsv_filtrado = np.zeros_like(hsv)

# Copiar el recorte del bounding box fusionado en su ubicación original
hsv_filtrado[min_y:max_y, min_x:max_x] = hsv[min_y:max_y, min_x:max_x]

# Si querés convertirlo de nuevo a BGR para visualizarlo:
img_resultado = cv2.cvtColor(hsv_filtrado, cv2.COLOR_HSV2BGR)


plt.imshow(cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB))
plt.title("Imágen - Resistencia cortada")
plt.axis("off")
plt.show()


hsv2 = cv2.cvtColor(img_resultado, cv2.COLOR_BGR2HSV)


# Marrón-amarillo claro del cuerpo
lower_rs = np.array([14, 120, 140])
upper_rs = np.array([15, 160, 185])

# Máscara
mask2 = cv2.inRange(hsv2, lower_rs, upper_rs)

plt.imshow(cv2.cvtColor(mask2, cv2.COLOR_BGR2RGB))
plt.title("mask2")
plt.axis("off")
plt.show()

mask2_inv =cv2.bitwise_not(mask2)

plt.imshow(cv2.cvtColor(mask2_inv, cv2.COLOR_BGR2RGB))
plt.title("mask2_inv")
plt.axis("off")
plt.show()

A = Aop
B = cv2.getStructuringElement(cv2.MORPH_RECT, (75,75))
Aclau = cv2.morphologyEx(A, cv2.MORPH_CLOSE, B)

mask2_inv[Aclau == 0] = 0

plt.imshow(cv2.cvtColor(Aclau, cv2.COLOR_BGR2RGB))
plt.title("Aclau")
plt.axis("off")
plt.show()

plt.imshow(cv2.cvtColor(mask2_inv, cv2.COLOR_BGR2RGB))
plt.title("mask2_inv")
plt.axis("off")
plt.show()

# APERTURA
A = mask2_inv
B = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
Aop2 = cv2.morphologyEx(A, cv2.MORPH_OPEN, B)

#Aop2[mask_invertida == 0] = 0
'''
