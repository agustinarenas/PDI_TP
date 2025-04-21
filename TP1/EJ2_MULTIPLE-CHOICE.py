import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow(img, ax=None, new_fig=True, title=None, color_img=False, blocking=True, colorbar=True, ticks=False):
    if ax is None:
        if new_fig:
            plt.figure()
        ax = plt.gca()
    if color_img:
        im = ax.imshow(img)
    else:
        im = ax.imshow(img, cmap='gray')
    ax.set_title(title)
    if not ticks:
        ax.set_xticks([]), ax.set_yticks([])
    if colorbar:
        plt.colorbar(im, ax=ax)
    if new_fig:
        plt.show(block=blocking)

# 2-A_Definimos funciones para detectar respuestas
def procesar_circulos(img, mostrar_plots=False, mostrar_resultados=False):
    """
    Procesa una imagen en escala de grises para detectar círculos, ordena por filas,
    y determina si están marcados según la intensidad promedio del parche central.
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
            fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
            imshow(fc, ax=ax, colorbar=False, title=f"Círculos ordenados por filas ({len(circle_list_ordenada)})", new_fig=False)
            plt.tight_layout()
            plt.show()
            plt.close()
        # === FIN BLOQUE DE VISUALIZACIÓN ===
    respuestas_detectadas = []
    # Evaluación de intensidad en el centro de cada círculo
    for i, (x, y, r) in enumerate(circle_list_ordenada):
        # Extrae un parche 14x14 alrededor del centro del círculo; img en escala de grises
        patch = img[max(y-7, 0):y+7, max(x-7, 0):x+7]

        # Calcula la intensidad promedio
        mean_intensity = np.mean(patch)

        # Clasificación: marcado o vacío
        estado = "MARCADO" if mean_intensity < 140 else "VACÍO"
        respuestas_detectadas.append(estado)

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
    if mostrar_resultados:
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
            if mostrar_resultados:
                print("Pregunta {}: NO MARCADO".format(pregunta_num))
            respuestas_sin_marcar += 1
            continue

        if indice_correcto == indice_respuesta and marcadas == 1:
            if mostrar_resultados:
                print("Pregunta {}: OK".format(pregunta_num))
            respuestas_correctas_contador += 1

        else:
            if mostrar_resultados:
                print("Pregunta {}: MAL".format(pregunta_num))
    estado_aprobado = "APROBADO" if respuestas_correctas_contador > 20 else "NO APROBADO"
    # --- Resumen ---
    if mostrar_resultados:
        print("\n----- RESUMEN -----")
        print("Total opciones: {}".format(total_opciones))
        print("Total preguntas: {}".format(total_preguntas))
        print("Respuestas detectadas: {} / {}".format(respuestas_dadas, total_preguntas))
        print("Respuestas correctas: {} / {}".format(respuestas_correctas_contador, total_preguntas))
        print("Respuestas sin marcar: {} / {}".format(respuestas_sin_marcar, total_preguntas))
    return estado_aprobado

# 2-B_Definimos funciones para controlar datos
def acondicionamiento_imagen(img):
    """
    Procesa la imagen para detectar el encabezado y los separadores verticales entre campos.

    Parámetros:
    img (np.ndarray): Imagen en escala de grises.

    Devuelve:
    tuple: Una lista con las posiciones x de los separadores verticales y el recorte del encabezado.
    """
    img_zeros = (img==0) # Por simplicidad, genero una matriz booleana con TRUE donde supongo que hay letras (pixel = 0)

    img_row_zeros = img_zeros.any(axis=1)

    x = np.diff(img_row_zeros)          
    renglones_indxs = np.argwhere(x)    # Esta variable contendrá todos los inicios y finales de los renglones

    ii = np.arange(0,len(renglones_indxs),2) 
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

    # Elegir un umbral
    umbral = 0.8 * img_encabezado_zeros.shape[0]
    columnas_lineas = suma_col > umbral

    
    x_idxs = np.argwhere(columnas_lineas).flatten()       

    # Agrupar columnas que estén cerca (dentro de una tolerancia de píxeles)
    diferencias = np.diff(x_idxs)
    grupos = np.where(diferencias > 10)[0] + 1

    columnas_finales = np.split(x_idxs, grupos)

    # Extraer un valor representativo de cada grupo
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
    #Salteo los campos innecesarios y me quedo con los utiles
    indices_utiles = [1, 3, 5,7]

    # Extraer solo los campos deseados
    recortes_filtrados = [recortes[i] for i in indices_utiles]
    return recortes_filtrados

 




# Función para validar encabezado (Name, Date, Class)
def validar_campo(nombre_campo, objetos_validos, espacios):
    """
    Valida un campo del encabezado según su tipo (name, id, code, date).

    Parámetros:
    nombre_campo (str): Nombre del campo a validar.
    objetos_validos (int): Cantidad de caracteres (objetos detectados).
    espacios (int): Cantidad de espacios detectados entre caracteres.

    Devuelve:
    str: "OK" si el campo cumple con las reglas, "MAL" si no.
    """
    #Name: Debe contener al menos dos palabras y no más de 25 caracteres en total.
    if nombre_campo == 'name':
        return "OK" if objetos_validos <= 25 and espacios >= 1 else "MAL"
    
    #ID: Debe contener sólo 8 caracteres en total, formando una única palabra. 
    elif nombre_campo == 'id':
        return "OK" if objetos_validos == 8 and espacios == 0 else "MAL"
    
    #Date: Debe contener sólo 8 caracteres en total, formando una única palabra.
    elif nombre_campo == 'date':
        return "OK" if objetos_validos == 8 and espacios == 0 else "MAL"
    
    #Code: Debe contener un único caracter. 
    elif nombre_campo == 'code':
        return "OK" if objetos_validos == 1 and espacios == 0 else "MAL"
    else:
        return "MAL"

def correcion_encabezado(img, mostrar_plots=False):
    """
    Corrige los campos del encabezado de un examen detectando caracteres y espacios entre ellos.
    
    Parámetros:
    img (np.ndarray): Imagen en escala de grises del examen.
    mostrar_plots (bool): Si es True, se muestran visualizaciones de los componentes conectados.

    Devuelve:
    str: Estado de validación de cada campo del encabezado (name, id, code, date).
    """
    # Procesamiento inicial del encabezado
    x_separadores, img_encabezado = acondicionamiento_imagen(img)
    recortes_filtrados = recorte_campos(x_separadores, img_encabezado, img)
    
    nombres_campos = ['name', 'id', 'code', 'date']
    resultados = []

    # Recorre cada campo recortado (nombre, ID, código, fecha)
    for nombre_campo, recorte in zip(nombres_campos, recortes_filtrados):
        # Recorta márgenes para evitar ruidos en los bordes
        recorte = recorte[:, 5:-5]
        _, recorte_bin = cv2.threshold(recorte, 160, 255, cv2.THRESH_BINARY_INV)

        # Detecta componentes conectados
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(recorte_bin, 8, cv2.CV_32S)

        objetos_validos = 0
        centroides_validos = []

        # Filtra componentes pequeños
        for j, stat in enumerate(stats[1:], start=1):
            x, y, w, h, area = stat
            if area >= 3:
                objetos_validos += 1
                centroides_validos.append(centroids[j][0])  # Coordenada X

        # Calcula distancias entre centroides y detecta "espacios"
        centroides_validos.sort()
        distancias = np.diff(centroides_validos)
        umbral_espacio = np.median(distancias) * 1.4 if len(distancias) > 0 else 0
        espacios = np.sum(distancias > umbral_espacio)

        # Valida el campo según sus componentes y espacios
        estado = validar_campo(nombre_campo, objetos_validos, espacios)
        resultados.append(f'{nombre_campo}: {estado}')

        # Mostrar visualización si está activado
        if mostrar_plots:
            if num_labels > 1:
                labels_norm = np.uint8((255 / (num_labels - 1)) * labels)
            else:
                labels_norm = np.zeros_like(labels, dtype=np.uint8)

            im_color = cv2.applyColorMap(labels_norm, cv2.COLORMAP_JET)
            im_color = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)

            for centroid in centroids:
                cv2.circle(im_color, tuple(np.int32(centroid)), 4, color=(255, 255, 255), thickness=-1)
            for st in stats:
                cv2.rectangle(im_color, (st[0], st[1]), (st[0] + st[2], st[1] + st[3]), color=(0, 255, 0), thickness=1)

            plt.figure(figsize=(8, 4))
            plt.imshow(im_color)
            plt.axis('off')
            plt.title(f"Componentes conectados - {nombre_campo}")
            plt.tight_layout()
            plt.show()
            plt.close()

    return '\n'.join(resultados)

#2-D_Mostrar imagen del nombre con su aprobacion de examen.
def imagen_correcciones(img):
    """
    Muestra el campo de nombre del encabezado con un ícono de aprobado o no aprobado,
    según el resultado de procesar el examen.

    Parámetros:
    img (np.ndarray): Imagen en escala de grises del examen.
    """
   
    x_separadores, img_encabezado = acondicionamiento_imagen(img)
    recortes_filtrados = recorte_campos(x_separadores, img_encabezado, img)
    nombre = recortes_filtrados[0]  # Primer campo: nombre

    respuesta = procesar_circulos(img,mostrar_resultados=False, mostrar_plots=False)  # 'APROBADO' o 'NO APROBADO'
    
    return nombre,respuesta
# 2-C_Procesar n imagenes. Mostrar todos los resultados.
def correcion_examen():
    """
    Procesa una serie de imágenes de examen y muestra:
    - Corrección de respuestas del multiple choice
    - Estado del encabezado (nombre, ID, etc.)
    - Visual de nombre + aprobado/no aprobado
    """
    resultados = []
    for i in range(1, 6):
        exam_id = f'multiple_choice_{i}.png'
        img_i = cv2.imread(f'PDI_TP/TP1/{exam_id}', cv2.IMREAD_GRAYSCALE)
         
        print(f"===== EXAMEN {i} =====")

        # 2-A
        print(f'{procesar_circulos(img_i, mostrar_resultados=True, mostrar_plots=False)}\n')

        #2-B
        print("----- CONTROL DE DATOS COMPLETADOS -----")
        print(f'Encabezado\n{correcion_encabezado(img_i, mostrar_plots=False)}\n')
        
        #2-D
        nombre, respuesta = imagen_correcciones(img_i)
        resultados.append((nombre, respuesta))
        
    fig, axs = plt.subplots(1, len(resultados), figsize=(20, 4))
    for ax, (nombre_img, estado) in zip(axs, resultados):
        ax.imshow(nombre_img, cmap='gray')
        ax.axis('off')
        if estado == 'APROBADO':
            ax.set_title("✔️", fontsize=12, color='green')
        else:
            ax.set_title("X", fontsize=12, color='red')
    plt.tight_layout()
    plt.show()
    plt.close()
   
correcion_examen()

