import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('TP1/Imagen_con_detalles_escondidos.tif',cv2.IMREAD_GRAYSCALE)

def EcualizacionLocal(img,size):
    mitad_ventana = size//2
    img_fil = cv2.copyMakeBorder(img, mitad_ventana,mitad_ventana
                                 ,mitad_ventana,mitad_ventana,
                                 borderType = cv2.BORDER_REPLICATE)
    rows,cols = img.shape
    
    img_equalizada = np.zeros_like(img)
    for i in range(mitad_ventana, rows - mitad_ventana):
            for j in range(mitad_ventana, cols - mitad_ventana):
                # Extraer la ventana MxN alrededor del pixel (i, j)
                ventana = img_fil[i - mitad_ventana : i + mitad_ventana + 1, j - mitad_ventana : j + mitad_ventana + 1]
        
                histograma =cv2.calcHist([ventana], [0], None, [256], [0, 256])
                histn = histograma.astype(np.double) / ventana.size
                
                # Calcular el CDF (función de distribución acumulativa) del histograma local
                cdf = histn.cumsum()
                cdf_normalizar = cdf * 255 / cdf[-1]  # Normalizar entre 0 y 255

                # Mapear el valor del píxel central usando la CDF
                img_equalizada[i, j] = cdf_normalizar[img[i, j]]

    img_suavizada = cv2.medianBlur(img_equalizada, 3)
    return img_suavizada

def EcualizacionLocal_2(img, size):
    mitad_ventana = size // 2
    img_padded = cv2.copyMakeBorder(img, mitad_ventana, mitad_ventana, 
                                    mitad_ventana, mitad_ventana, 
                                    borderType=cv2.BORDER_REPLICATE)
    
    img_equalizada = np.zeros_like(img)
    rows, cols = img.shape

    for i in range(rows):
        for j in range(cols):
            # Extraer ventana local
            ventana = img_padded[i:i + size, j:j + size]

            # Calcular histograma y ecualización
            histograma = cv2.calcHist([ventana], [0], None, [256], [0, 256])
            histn = histograma.astype(np.float32) / ventana.size
            cdf = histn.cumsum()
            cdf_normalizado = (cdf * 255 / cdf[-1]).astype(np.uint8)  # Normalizar a 0-255

            # Asignar el nuevo valor al píxel central
            img_equalizada[i, j] = cdf_normalizado[img[i, j]]
    img_suavizada = cv2.medianBlur(img_equalizada, 3)   
    
    return img_suavizada

# Fución para hacer el ecualizado de histograma local.
def EcualizacionLocal_3(img, dimV):
    """
    Recibe: Imagen para realizarle el ecualizado de histograma local y una tupla con el tamaño de la ventana que se quiere utilizar.
    Devuelve: La imagen con el ecualizado de histograma local realizado.
    """
    #Copia de la imagen de entrada y guarda el tamaño de la misma.
    imgEcualizada = img.copy()
    imgW, imgH = img.shape

    #Guarda el tamaño de la ventana a utilizar.
    ventanaW, ventanaH = dimV
    mitad_ventana=ventanaH//2
    #Aplica el BORDER_REPLICATE.
    imgEcualizada = cv2.copyMakeBorder(imgEcualizada, mitad_ventana, mitad_ventana,mitad_ventana,mitad_ventana, borderType=cv2.BORDER_REPLICATE)

    #Recorre la imagen por ventanas del tamaño recibido.
    for i in range(0, imgW - ventanaW, ventanaW):
        for j in range(0, imgH - ventanaH, ventanaH):
            #Realiza un ecualizado de histograma en la ventana en que se encuentra.
            imgEcualizada[i:i+ventanaH,j:j+ventanaW] = cv2.equalizeHist(imgEcualizada[i:i+ventanaH,j:j+ventanaW])

    #Aplica medianBlur para borrar los puntos blancos de la imagen obtenida.
    imgEcualizada = cv2.medianBlur(imgEcualizada, 3)

    return imgEcualizada

# Crea la nueva imagen aplicando la función y la muestra. 
eq = EcualizacionLocal_3(img, (19, 19))
tamaño_ventana = 7

imagen_ecualizada = EcualizacionLocal(img,tamaño_ventana)
imagen_ecualizada2 = EcualizacionLocal_2(img,tamaño_ventana)
#imagen_ecualizada3 = EcualizacionLocal_2(img,tamaño_ventana)
#imagen_binaria = cv2.adaptiveThreshold(imagen_ecualizada2, 255, 
                                       #cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       #cv2.THRESH_BINARY, 27, 14)
ax = plt.subplot(221);plt.imshow(img,cmap="gray"),plt.title("Imagen priginal")
plt.subplot(222, sharex=ax, sharey=ax);plt.imshow(imagen_ecualizada,cmap="gray"), plt.title("imagen ecualizada1")
plt.subplot(223, sharex=ax, sharey=ax);plt.imshow(imagen_ecualizada2,cmap="gray"), plt.title("imagen ecualizada2")
plt.subplot(224, sharex=ax, sharey=ax);plt.imshow(eq,cmap="gray"), plt.title("imagen ecualizada3")

plt.show()

#Analisis
"""
En la imagen con detalles escondidos se pueden ver 5 cuadrados, uno en cada punta y uno en el centro.
Con el correspondiente procesamiento de la imagen pudimos ver cuales son los detalles escondidos detras de los cuadrados.
Arriba a la izquierda tenemos un cuadrado mas chico en el centro. Arriba a la derecha tenemos una linea cruzada en el centro.
Abajo a la izquierda tenemos lineas horizontalmente con distintas escalas de 'grises'. Abajo a la derecha hay un circulo en el centro.
Por ultimo en el centro tenemos la letra 'a'.
Probamos con diferentes tamaños de ventanas y podemos concluir que mientras mas chica sea la ventana mas claro se van haciendo los cuadrados,
pensamos que mientras mas chicas las ventanas hace los procesos mas detallados por lo tanto tarda mas pero se obtiene mas detallado, sin embargo,
no hace falta que las ventanas sean tan chicas en este caso, ya que con una ventana de 7x7 o 9x9 se logra ver bien los detalles escondidos
"""
