# BIBLIOTECAS
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Defininimos función para mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
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


# 2-A_Definimos funciones para detectar respuestas
def procesar_circulos(img, mostrar_plots=False):
    """
    Procesa una imagen en escala de grises para detectar círculos, ordenarlos por filas,
    y determinar si están marcados según la intensidad promedio del parche central.
    """

    # Detección de círculos usando la transformada de Hough
    circles = cv2.HoughCircles(
        img,
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=10,
        param1=50,
        param2=15,
        minRadius=9,
        maxRadius=14
    )

    estados_circulos = []

    if circles is not None:
        # Conversión y redondeo de coordenadas
        circles = np.uint16(np.around(circles))
        circle_list = [(x, y, r) for x, y, r in circles[0]]

        # Agrupación de círculos por filas según proximidad vertical
        filas = []
        tolerancia = 10
        for c in sorted(circle_list, key=lambda c: c[1]):  # ordenar por Y
            x, y, r = c
            colocado = False
            for fila in filas:
                if abs(int(fila[0][1]) - int(y)) < tolerancia:
                    fila.append(c)
                    colocado = True
                    break
            if not colocado:
                filas.append([c])

        # Ordenar los círculos dentro de cada fila de izquierda a derecha
        for fila in filas:
            fila.sort(key=lambda c: c[0])

        # Aplanar lista de círculos en el orden correcto (fila por fila, de arriba a abajo)
        circle_list_ordenada = [c for fila in filas for c in fila]

        # Conversión de imagen a RGB para dibujar
        fc = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # === BLOQUE DE VISUALIZACIÓN (controlado por 'mostrar_plots') ===
        if mostrar_plots:
            # Dibujo de círculos y etiquetas
            for idx, (x, y, r) in enumerate(circle_list_ordenada):
                diameter = 2 * r
                cv2.circle(fc, (x, y), r, (0, 255, 0), 2)           # círculo
                cv2.circle(fc, (x, y), 2, (0, 0, 255), 2)           # centro
                cv2.putText(fc, f"{diameter}px", (x - 10, y - 10), # diámetro
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # Visualización
            plt.figure(figsize=(12, 12), dpi=100)
            imshow(fc, colorbar=False, title=f"Círculos ordenados por filas ({len(circle_list_ordenada)})", new_fig=False)
            plt.tight_layout()
            plt.show()
        # === FIN BLOQUE DE VISUALIZACIÓN ===
    respuestas_detectadas = []
    # Evaluación de intensidad en el centro de cada círculo
    for i, (x, y, r) in enumerate(circle_list_ordenada):
        # Extrae un parche 14x14 alrededor del centro del círculo; img en escala de grises
        patch = img[max(y-7, 0):y+7, max(x-7, 0):x+7]

        # Convierte el parche a escala de grises
        #patch_gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

        # Calcula la intensidad promedio
        mean_intensity = np.mean(patch)

        # Clasificación: marcado o vacío
        estado = "MARCADO" if mean_intensity < 140 else "VACÍO"
        respuestas_detectadas.append(estado)

        # Imprime el resultado de forma legible
        #print(f"Círculo {i+1}: ({x}, {y}), radio={r}, intensidad promedio={mean_intensity:.2f} → {estado}")

    # Respuestas correctas
    respuestas_correctas = [
        "MARCADO", "VACÍO", "VACÍO", "VACÍO", "VACÍO",  # Pregunta 1
        "MARCADO", "VACÍO", "VACÍO", "VACÍO", "VACÍO",  # Pregunta 2
        "VACÍO", "MARCADO", "VACÍO", "VACÍO", "VACÍO",  # Pregunta 3
        "MARCADO", "VACÍO", "VACÍO", "VACÍO", "VACÍO",  # Pregunta 4
        "VACÍO", "VACÍO", "VACÍO", "MARCADO", "VACÍO",  # Pregunta 5
        "VACÍO", "MARCADO", "VACÍO", "VACÍO", "VACÍO",  # Pregunta 6
        "VACÍO", "MARCADO", "VACÍO", "VACÍO", "VACÍO",  # Pregunta 7
        "MARCADO", "VACÍO", "VACÍO", "VACÍO", "VACÍO",  # Pregunta 8
        "VACÍO", "MARCADO", "VACÍO", "VACÍO", "VACÍO",  # Pregunta 9
        "VACÍO", "VACÍO", "VACÍO", "VACÍO", "MARCADO",  # Pregunta 10
        "MARCADO", "VACÍO", "VACÍO", "VACÍO", "VACÍO",  # Pregunta 11
        "VACÍO", "VACÍO", "VACÍO", "MARCADO", "VACÍO",  # Pregunta 12
        "VACÍO", "VACÍO", "MARCADO", "VACÍO", "VACÍO",  # Pregunta 13
        "VACÍO", "VACÍO", "MARCADO", "VACÍO", "VACÍO",  # Pregunta 14
        "VACÍO", "VACÍO", "VACÍO", "MARCADO", "VACÍO",  # Pregunta 15
        "VACÍO", "VACÍO", "VACÍO", "VACÍO", "MARCADO",  # Pregunta 16
        "MARCADO", "VACÍO", "VACÍO", "VACÍO", "VACÍO",  # Pregunta 17
        "VACÍO", "VACÍO", "MARCADO", "VACÍO", "VACÍO",  # Pregunta 18
        "VACÍO", "VACÍO", "MARCADO", "VACÍO", "VACÍO",  # Pregunta 19
        "VACÍO", "VACÍO", "VACÍO", "MARCADO", "VACÍO",  # Pregunta 20
        "VACÍO", "MARCADO", "VACÍO", "VACÍO", "VACÍO",  # Pregunta 21
        "VACÍO", "VACÍO", "VACÍO", "VACÍO", "MARCADO",  # Pregunta 22
        "MARCADO", "VACÍO", "VACÍO", "VACÍO", "VACÍO",  # Pregunta 23
        "VACÍO", "VACÍO", "VACÍO", "VACÍO", "MARCADO",  # Pregunta 24
        "VACÍO", "VACÍO", "VACÍO", "MARCADO", "VACÍO",  # Pregunta 25
    ]
    # Crear un reporte de comparación entre las respuestas correctas y las detectadas
    total_opciones = len(respuestas_detectadas)
    total_preguntas = 0
    respuestas_dadas = 0
    respuestas_correctas_contador = 0
    respuestas_sin_marcar = 0
    print("\n----- RESULTADOS DE LAS RESPUESTAS -----")
    for i in range(0, len(respuestas_correctas), 5):
        
        pregunta_num = int(i/5 + 1)
        correctas = respuestas_correctas[i:i+5]
        respuestas = respuestas_detectadas[i:i+5]
        total_preguntas += 1

        marcadas = respuestas.count("MARCADO")

        try:
            indice_correcto = correctas.index("MARCADO")
            indice_respuesta = respuestas.index("MARCADO")
            respuestas_dadas += 1

        except ValueError:
            print("Pregunta {}: NO MARCADO".format(pregunta_num))
            respuestas_sin_marcar += 1
            continue

        if indice_correcto == indice_respuesta and marcadas == 1:
            print("Pregunta {}: OK".format(pregunta_num))
            respuestas_correctas_contador += 1

        else:
            print("Pregunta {}: MAL".format(pregunta_num))
    
    # --- Resumen ---
    print("\n----- RESUMEN -----")
    print("Total opciones: {}".format(total_opciones))
    print("Total preguntas: {}".format(total_preguntas))
    print("Respuestas detectadas: {} / {}".format(respuestas_dadas, total_preguntas))
    print("Respuestas correctas: {} / {}".format(respuestas_correctas_contador, total_preguntas))
    print("Respuestas sin marcar: {} / {}".format(respuestas_sin_marcar, total_preguntas))
    #return estado


# 2-B_Definimos funciones para controlar datos

def acondicionamiento_imagen(img):
    """
    Procesa la imagen para detectar el encabezado y los separadores verticales entre campos.

    Parámetros:
    img (np.ndarray): Imagen en escala de grises.

    Devuelve:
    tuple: Una lista con las posiciones x de los separadores verticales y el recorte del encabezado.
    """

    # Crear máscara booleana donde el píxel es negro (0)
    img_zeros = img == 0

    # Filas que contienen al menos un valor negro
    img_row_zeros = img_zeros.any(axis=1)

    # Identificar transiciones (bordes de renglones)
    x = np.diff(img_row_zeros.astype(np.int8))
    renglones_indxs = np.argwhere(x)

    # Asegurar que comiencen en una línea "encendida"
    ii = np.arange(0, len(renglones_indxs), 2)
    renglones_indxs[ii] += 1

    # Emparejar inicios y finales de renglones
    r_idxs = np.reshape(renglones_indxs, (-1, 2))

    # Seleccionar el renglón más alto (mayor altura → encabezado)
    alturas = r_idxs[:, 1] - r_idxs[:, 0]
    indice_max = np.argmax(alturas)
    renglon_encabezado = r_idxs[indice_max]

    # Recorte del encabezado desde la imagen original
    img_encabezado = img[renglon_encabezado[0]:renglon_encabezado[1], :]

    # Binarización del encabezado por umbral simple
    umbral_bin = 130
    img_encabezado_zeros = img_encabezado < umbral_bin

    # Sumar valores negros por columna
    suma_col = img_encabezado_zeros.sum(axis=0)

    # Umbral de cantidad de píxeles negros para considerar "línea vertical"
    umbral = 0.8 * img_encabezado_zeros.shape[0]
    columnas_lineas = suma_col > umbral

    # Índices x de columnas que superan el umbral
    x_idxs = np.argwhere(columnas_lineas).flatten()

    # Agrupar columnas cercanas (dentro de una tolerancia de píxeles)
    diferencias = np.diff(x_idxs)
    grupos = np.where(diferencias > 10)[0] + 1
    columnas_finales = np.split(x_idxs, grupos)

    # Seleccionar un valor representativo por grupo (mediana)
    x_separadores = [int(np.median(col)) for col in columnas_finales]

    return x_separadores, img_encabezado


def recorte_campos(x_separadores, img_encabezado):
    """
    Recorta los campos del encabezado en base a los separadores verticales detectados.

    Parámetros:
    x_separadores (list): Coordenadas x de los separadores verticales.
    img_encabezado (np.ndarray): Imagen recortada del encabezado.

    Devuelve:
    list: Lista de imágenes correspondientes a los campos útiles del encabezado.
    """
    recortes = [
        img_encabezado[:, x_separadores[i]:x_separadores[i + 1]]
        for i in range(len(x_separadores) - 1)
    ]

    indices_utiles = [1, 3, 5, 7]
    recortes_filtrados = [recortes[i] for i in indices_utiles if i < len(recortes)]

    return recortes_filtrados


def contar_palabras_en_imagenes(img_rec, min_area_name=10, separation_threshold=9.7, mostrar_plots=False):
    """
    Procesa una lista de imágenes para detectar el número de caracteres y palabras en cada una.

    Parámetros:
    - img_rec (list): Lista de imágenes recortadas en escala de grises.
    - min_area_name (int): Área mínima para considerar un componente como letra (solo en el primer campo).
    - separation_threshold (float): Distancia mínima entre caracteres para separar palabras.
    - mostrar_plots (bool): Si es True, muestra visualizaciones interactivas.

    Devuelve:
    - list: Tuplas con (número de caracteres, número de palabras) por imagen.
    """
    total_datos = 0 # Variable auxiliar para debug
    datos_correctos = 0
    print("\n----- CONTROL DE DATOS COMPLETADOS -----")

    for i, img in enumerate(img_rec):

        total_datos += 1

        # Recorte lateral para evitar bordes
        img = img[:, 5:-5]

        # Umbral binario inverso para resaltar letras
        _, img_bin = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY_INV)

        # Segmentación por componentes conectados
        connectivity = 8
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin, connectivity, cv2.CV_32S)

        # Determinar umbral mínimo de área
        min_area = min_area_name if i == 0 else 0

        # Filtrar caracteres por área y extraer coordenadas X
        filtered_x_coords = []
        for idx in range(1, num_labels):
            area = stats[idx, cv2.CC_STAT_AREA]
            if area >= min_area:
                x = centroids[idx][0]
                filtered_x_coords.append(x)

        num_carac = len(filtered_x_coords)
        #print(f"Imagen {i+1}: {num_carac} caracteres detectados")

        # Determinamos el número de palabras en base a la separación entre caracteres
        if not filtered_x_coords:
            num_words = 0
        else:
            x_coords_sorted = sorted(filtered_x_coords)
            num_words = 1
            for j in range(1, len(x_coords_sorted)):
                if x_coords_sorted[j] - x_coords_sorted[j - 1] > separation_threshold:
                    num_words += 1
        #print(f"Imagen {i+1}: {num_words} palabras detectadas\n")
        #resultados.append((num_carac, num_words))
        
        if i == 0:
            if num_carac <= 25 and num_words >= 2:
                print("Name: OK")
                datos_correctos += 1
            else:
                print("Name: MAL")

        elif i == 1:
            if num_carac == 8 and num_words == 1:
                print("ID: OK")
                datos_correctos += 1
            else:
                print("ID: MAL")
        
        elif i == 2:
            if num_carac == 1 and num_words == 1:
                print("Code: OK")
                datos_correctos += 1
            else:
                print("Code: MAL")

        elif i == 3:
            if num_carac == 8 and num_words == 1:
                print("Date: OK")
                datos_correctos += 1
            else:
                print("Date: MAL")

        # === BLOQUE DE VISUALIZACIÓN (opcional) ===
        if mostrar_plots:
            # Normalización de etiquetas para visualización
            labels_norm = np.uint8((255 / (num_labels - 1)) * labels)
            im_color = cv2.applyColorMap(labels_norm, cv2.COLORMAP_JET)
            im_color = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)

            # Dibujamos los centroides y cajas de los componentes conectados
            for centroid in centroids:
                cv2.circle(im_color, tuple(np.int32(centroid)), 4, color=(255, 255, 255), thickness=-1)
            for st in stats:
                cv2.rectangle(im_color, (st[0], st[1]), (st[0] + st[2], st[1] + st[3]), color=(0, 255, 0), thickness=1)

            # Mostrar la imagen con los boxes detectados
            plt.figure(figsize=(8, 4))
            plt.imshow(im_color)
            plt.axis('off')
            plt.title(f"Componentes conectados - Imagen {i+1}")
            plt.tight_layout()
            plt.show()
        # === FIN BLOQUE DE VISUALIZACIÓN ===

    # --- Resumen ---
    print("\n----- RESUMEN -----")
    print("Total de datos: {}".format(total_datos))
    print("Datos correctos: {} / {}".format(datos_correctos, total_datos))
    #return "end"

# 2-C_Procesar n imagenes
for exam_id in range(1, 6):
    print(f"\n===== EXAMEN {exam_id} =====")

    # Cargamos la imagen
    img = cv2.imread(f'TP1/multiple_choice_{exam_id}.png', cv2.IMREAD_GRAYSCALE)

    # 2-A
    procesar_circulos(img, mostrar_plots=False)

    # 2-B
    x_sep, img_encab = acondicionamiento_imagen(img)

    img_rec = recorte_campos(x_sep, img_encab)

    estado = contar_palabras_en_imagenes(img_rec, min_area_name=10, separation_threshold=9.7, mostrar_plots=False)

    #break
