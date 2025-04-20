import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow(img, new_fig=True, title=None, color_img=False, blocking=True, colorbar=True, ticks=False):
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
        plt.show()

img = cv2.imread('./TP1/multiple_choice_1.png',cv2.IMREAD_GRAYSCALE)

print(type(img))


def acondicionamiento_imagen(img):
    """
    Procesa la imagen para detectar el encabezado y los separadores verticales entre campos.

    Parámetros:
    img (np.ndarray): Imagen en escala de grises.

    Devuelve:
    tuple: Una lista con las posiciones x de los separadores verticales y el recorte del encabezado.
    """
    img_zeros = img==0 # Por simplicidad, genero una matriz booleana con TRUE donde supongo que hay letras (pixel = 0)

    img_row_zeros = img_zeros.any(axis=1)

    xr = img_row_zeros*(img.shape[1]-1)     # Generamos valores en el eje x (el eje que apunta hacia la derecha) --> dos posibles valores: 0 o el ancho de la imagen (cantidad de columnas de la imagen -1).
    yr = np.arange(img.shape[0])            # Generamos valores en el eje y (el eje que apunta hacia abajo) --> Vector que arranca en 0, aumenta de a 1 y llega hasta la cantidad de filas de la imagen [0, 1, ..., N_filas-1]  

    x = np.diff(img_row_zeros)          
    renglones_indxs = np.argwhere(x)    # Esta variable contendrá todos los inicios y finales de los renglones

    ii = np.arange(0,len(renglones_indxs),2)    # 0 2 4 ... X --> X es el último nro par antes de len(renglones_indxs)
    renglones_indxs[ii]+=1

    r_idxs = np.reshape(renglones_indxs, (-1,2))

    #Detectar la línea más alta (la del encabezado)
    alturas = r_idxs[:,1] - r_idxs[:,0]
    indice_max = np.argmax(alturas)
    renglon_encabezado = r_idxs[indice_max]

    img_encabezado = img[renglon_encabezado[0]:renglon_encabezado[1], :]

    umbral_bin = 130
    img_encabezado_zeros = img_encabezado < umbral_bin
    # Contar cantidad de pixeles negros por columna
    suma_col = img_encabezado_zeros.sum(axis=0)

    # Elegir un umbral: por ejemplo, más del 80% de la altura del encabezado
    umbral = 0.8 * img_encabezado_zeros.shape[0]
    columnas_lineas = suma_col > umbral

    # Visualizaru
    xc = np.arange(img_encabezado_zeros.shape[1])
    yc = columnas_lineas * (img_encabezado_zeros.shape[0]-1)
    x_idxs = np.argwhere(columnas_lineas).flatten()       

    # Agrupar columnas que estén cerca (dentro de una tolerancia de píxeles)
    diferencias = np.diff(x_idxs)
    grupos = np.where(diferencias > 10)[0] + 1  # 10 es una tolerancia, podés ajustarla

    columnas_finales = np.split(x_idxs, grupos)

    # Extraer un valor representativo de cada grupo (ej: la mediana)
    x_separadores = [int(np.median(col)) for col in columnas_finales]

    return x_separadores,img_encabezado

def recorte_campos(x_separadores,img_encabezado,img):
    """
    Recorta los campos del encabezado de la imagen en base a los separadores detectados.

    Parámetros:
    x_separadores (list): Coordenadas x de los separadores verticales.
    img_encabezado (np.ndarray): Imagen recortada del encabezado.
    img (np.ndarray): Imagen original.

    Devuelve:
    list: Lista de imágenes correspondientes a los campos útiles del encabezado.
    """
    x_separadores,img_encabezado = acondicionamiento_imagen(img)
    recortes = []
    for i in range(len(x_separadores) - 1):
        x0 = x_separadores[i]
        x1 = x_separadores[i + 1]
        campo = img_encabezado[:, x0:x1]   # Recorte entre separadores
        recortes.append(campo)
    # Por ejemplo: salteo el campo 0 y tomo los campos 1 y 2
    indices_utiles = [1, 3, 5,7]
    # Extraer solo los campos deseados
    recortes_filtrados = [recortes[i] for i in indices_utiles]
    return recortes_filtrados


x_sep, img_encab = acondicionamiento_imagen(img)

img_rec = recorte_campos(x_sep, img_encab,img)
'''
for img in img_rec:
    #imshow(img)
    img = img[:, 5:-5]
    _, recorte_bin = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY_INV)
    imshow(recorte_bin)
'''

# --- Detectar los diferentes componentes -------------------------------------------

for i, img in enumerate(img_rec):
    #img = cv2.imread('objects.tif', cv2.IMREAD_GRAYSCALE)
    img = img[:, 5:-5]
    _, img = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY_INV)
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)

    # CONTROL DE NAME - ID - CODE - DATE

    # Umbral para filtrar puntos/accentos (ajústalo según la imagen)
    min_area = 0
    if i == 0:
        min_area = 10

    # Obtener centroides y áreas, ignorando el fondo (índice 0)
    filtered_x_coords = []
    for idx in range(1, num_labels):  # Comenzamos desde 1 para ignorar  # Ver de actualizar para que filtre la max-area ya que puede no ser el primero
        area = stats[idx, cv2.CC_STAT_AREA]
        if area >= min_area:
            x = centroids[idx][0]
            filtered_x_coords.append(x)

    num_carac = len(filtered_x_coords)
    print("Número de caracteres detectados:", num_carac) 

    # Si no hay letras válidas detectadas
    if not filtered_x_coords:
        num_words = 0
    else:
        # Ordenar coordenadas X
        x_coords_sorted = sorted(filtered_x_coords)

        # Detectar separación entre palabras
        separation_threshold = 9.7  # Ajustable según la imagen
        num_words = 1
        for j in range(1, len(x_coords_sorted)):
            if x_coords_sorted[j] - x_coords_sorted[j - 1] > separation_threshold:
                num_words += 1

    print("Número de palabras detectadas:", num_words)

    #print(num_labels)
    #break


    '''
    imshow(img=labels)
    print(np.unique(labels))

    # Coloreamos los elementos
    labels_norm = np.uint8( (255/(num_labels-1)) * labels)    # Cambiamos el rango:
    print(np.unique(labels_norm))                             # 0 a num_labels -->  0 a 255 (con paso 255/num_labels)
    imshow(labels_norm)

    # im_color = cv2.applyColorMap(labels_norm, cv2.COLORMAP_HOT)
    im_color = cv2.applyColorMap(labels_norm, cv2.COLORMAP_JET)
    imshow(im_color, color_img=True, colorbar=False)                    # Observar los colores ---> Están en BGR...

    im_color = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)    # ... Paso a RGB
    imshow(im_color, color_img=True, colorbar=False)

    for centroid in centroids:
        cv2.circle(im_color, tuple(np.int32(centroid)), 9, color=(255,255,255), thickness=-1)
    for st in stats:
        cv2.rectangle(im_color, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(0,255,0), thickness=2)
    imshow(img=im_color, color_img=True, colorbar=False)

    #break
    '''
