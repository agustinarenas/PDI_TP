import cv2
import numpy as np
import matplotlib.pyplot as plt

# Definimos función para mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)


#img = cv2.imread('Procesamiento de imagenes 1/Tp_2_Pdi/placa.png')
img = cv2.imread('PDI_TP/TP2/placa.png')
type(img)
img.dtype
img.shape


# Stats
img.min()
img.max()

img_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
r,g,b = img_original.shape
#img_gray = cv2.imread('Procesamiento de imagenes 1/Tp_2_Pdi/placa.png',cv2.IMREAD_GRAYSCALE)
img_gray = cv2.imread('PDI_TP/TP2/placa.png',cv2.IMREAD_GRAYSCALE)

#1-A
# Suavizacion imagen
blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

#Binzarizacion de imagen
img_bin = cv2.adaptiveThreshold(blur, 255, 
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 
                                11, 2.1)

# Aplicacion de apertura
se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
fop = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, se)

#Aplicacion de dilatacion
kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
Fd = cv2.dilate(fop, kernel, iterations=1)

#Aplicacion de clausura
se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
fop_cl = cv2.morphologyEx(Fd, cv2.MORPH_CLOSE, se)

# Componentes conectados
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fop_cl, cv2.CV_32S, connectivity=8)

img_out_objetos = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

# Guardar todos los objetos relativamente grandes (para uso posterior)
objetos = []
for i in range(1, num_labels):  # Ignorar fondo
    x, y, w, h, area = stats[i]
    if area > 2000:
        cv2.rectangle(img_out_objetos, (x, y), (x + w, y + h), (0, 255, 0), 2)
        objetos.append(i)

# Filtrado del chip
img_out = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
chip = []
for i in objetos:
    x, y, w, h, area = stats[i]
    aspect_ratio = w / h if h != 0 else 0
    if area > 50000 and 0.1 < aspect_ratio < 0.6:
        cv2.rectangle(img_out, (x, y), (x + w, y + h), (255, 0, 0), 2)
        chip.append(i)

# Filtrado de los capacitores electrolitos
capacitores = []
for i in objetos:
    x, y, w, h, area = stats[i]
    aspect_ratio = w / h if h != 0 else 0 # Sirve para ver la forma 
    cx, cy = centroids[i]
    intensidad = img_gray[int(cy), int(cx)] # Valor de intensidad del píxel en el centroide
    if 7500 < area < 75000 and 0.7 < aspect_ratio < 1.8 and intensidad > 170 and h < 650:
        cv2.rectangle(img_out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        capacitores.append(i)

# Filtrado de las resistencias
resistencias = []
for i in objetos:
    x, y, w, h, area = stats[i]
    aspect_ratio = w / h if h != 0 else 0
    cx, cy = centroids[i]
    intensidad = img_gray[int(cy), int(cx)]
    # Condiciones ajustadas para resistencias(intensidad,ancho,area,forma,alto)
    if  100 < intensidad and w > 230 and 3000 < area < 10000 and 1.8 < aspect_ratio < 5 or 0.2 < aspect_ratio < 0.48 and 220 < h < 350:
        cv2.rectangle(img_out, (x, y), (x + w, y + h), (0, 0, 255), 2)
        resistencias.append(i)

# 1-B
# Clasificacion de capacitores electrolitos por tamaño de area
# Inicializar contadores
tipo1 = 0
tipo2 = 0
tipo3 = 0
tipo4 = 0
img_classif = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
for i in capacitores:
    
    x, y, w, h, area = stats[i]
    if area > 50000:
        color = (255, 0, 0)  #Tipo 1
        label = "Tipo 1"
        tipo1 += 1
    elif area > 20000:
        color = (0, 255, 0)  #Tipo 2
        label =  "Tipo 2"
        tipo2 += 1
    elif area >15000:
        color = (0, 0, 255)  #Tipo 3
        label = "Tipo 3"
        tipo3 += 1
    else:
        color = (255,255,0) #Tipo 4
        label = "Tipo 4"
        tipo4 += 1
    cv2.rectangle(img_classif, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img_classif, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

#Cantidad de capacitores por tipo (esquina superior izquierda de la imagen)
cv2.putText(img_classif, f"Tipo 1: {tipo1}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 6)
cv2.putText(img_classif, f"Tipo 2: {tipo2}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 6)
cv2.putText(img_classif, f"Tipo 3: {tipo3}", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 6)
cv2.putText(img_classif, f"Tipo 4: {tipo4}", (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 6)
#1-C
#Conteo de cantidad de resistencias en la placa
cantidad_resistencias = len(resistencias)

#Funcion para mostrar paso a paso lo que se hizo
def main():
    #Imagen original
    imshow(img_original,title='Imagen Original',blocking=True)

    #Imagen procesada en escala de grises
    imshow(fop_cl,title='Imagen Suavizada, Binarizada, Aplicacion De Apertura, Dilatacion y Clausura',blocking=True)

    #Imagen con deteccion de componentes conectadas
    imshow(img_out_objetos,title='Imagen con componentes conectadas',blocking=True)

    #1-A
    imshow(img_out,title='Imagen con los tres tipos de componentes principales',blocking=True)

    #1-B
    imshow(img_classif,title='Imagen con cuatro tipos de capacitores electrolitos',blocking=True)

    #1-C
    print(f'La cantidad exacta de resistencias electricas presentes en la placa son: {cantidad_resistencias}')
    

resultados = main()
