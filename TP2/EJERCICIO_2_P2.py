import cv2
import numpy as np
import matplotlib.pyplot as plt


# DETECTAR BANDAS DE RESISTENCIA
def detectar_bandas_resistencia(img_path,debug=False):
    """
    Lee la imagen de la resistencia, recorta el cuerpo sobre fondo azul y devuelve una lista con los 3 colores de banda
    (sin la banda de tolerancia), ordenados de más lejos a más cerca de la tolerancia.
    Si falla en algún paso, retorna None.
    """
    #Leer la imagen
    img = cv2.imread(img_path)
    if img is None:
        return None

    #Convertir la imagen BGR leída a HSV (para segmentación de color)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Definir rangos aproximados de HSV para cada color que nos interesa
    rangos_colores = {
        'NEGRO':   ([0, 0, 0], [160, 100, 45]),
        'MARRÓN':  ([0, 100, 50], [12, 255, 110]),
        'ROJO':   ([0, 150, 120], [10, 255, 255]),
        'NARANJA': ([0, 160, 160],[19, 220, 190]),
        'AMARILLO':([20, 150, 150], [35, 255, 255]),
        'VERDE':   ([35, 50, 50], [85, 255, 255]),
        'AZUL':    ([100, 50, 180], [120, 100, 255]),
        'VIOLETA': ([130, 50, 50], [160, 255, 255]),
        'GRIS':    ([0, 0, 100], [180, 50, 200]),
        'BLANCO':  ([0, 0, 150], [20, 80, 255])}

    #Segmentar el fondo azul para aislar el cuerpo de la resistencia
    lower_blue = np.array([105, 120, 50])
    upper_blue = np.array([135, 255, 255])
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)         # pixeles de color azul
    mask_inv = cv2.bitwise_not(mask_blue)                            # invertimos (cuerpo en blanco)
    
    # Dilatamos un poco para unir fragmentos pequeños
    kernel = np.array([[0,1,0],
                       [1,1,1],
                       [0,1,0]], np.uint8)
    Fd = cv2.dilate(mask_inv, kernel, iterations=2)

    # Apertura con un kernel grande para eliminar ruido interior
    kernel_cuerpo = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    mask_abierta = cv2.morphologyEx(Fd, cv2.MORPH_OPEN, kernel_cuerpo)

    # Clausura para cerrar huecos y obtener un contorno limpio
    kernel_cuerpo = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    mask_cerrada = cv2.morphologyEx(mask_abierta, cv2.MORPH_CLOSE, kernel_cuerpo)

    # Encontrar el contorno más grande: asumimos que es el cuerpo de la resistencia
    contours, _ = cv2.findContours(mask_cerrada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Si no hay contornos, devolvemos None
        return None

    # Elegimos el contorno de mayor área
    cnt_cuerpo = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt_cuerpo)

    # Recortamos la región de interés (cuerpo) en HSV
    img_rec_hsv = img_hsv[y:y + h, x:x + w]
    # También generamos su versión en BGR para dibujar rectángulos/puntos
    img_rec_bgr = cv2.cvtColor(img_rec_hsv, cv2.COLOR_HSV2BGR)

    #Detectar bandas de cada color (incluyendo hasta 2 repeticiones del mismo color)
    #    a) Creamos 'bandas_detectadas' con (centro_x, nombre_color)
    #    b) Aplicamos findContours en cada máscara de color
    #    c) Filtramos por área > 700 para evitar ruido
    bandas_detectadas = []
    for color, (low, high) in rangos_colores.items():
        low_hsv = np.array(low)
        high_hsv = np.array(high)
        # Máscara para el rango HSV de este color
        mask_color = cv2.inRange(img_rec_hsv, low_hsv, high_hsv)
        if color == 'NEGRO':
            # DILATACIÓN: Engrosamos las regiones negras para cerrar pequeños cortes o fragmentaciones
            kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
            Fd = cv2.dilate(mask_color, kernel, iterations=1)

            # CLAUSURA: Unimos regiones separadas que deberían ser una sola banda (como si fuera un "puente")
            kernel_cuerpo = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
            mask_cerrada = cv2.morphologyEx(Fd, cv2.MORPH_CLOSE, kernel_cuerpo)

            # APERTURA: Eliminamos pequeñas manchas o uniones falsas generadas por la dilatación/clausura
            kernel_cuerpo = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask_color = cv2.morphologyEx(mask_cerrada, cv2.MORPH_OPEN, kernel_cuerpo)

        if color == 'MARRÓN':
            #Apertura
            kernel_cuerpo = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask_abierta = cv2.morphologyEx(mask_color, cv2.MORPH_OPEN, kernel_cuerpo)
            #Clausura
            kernel_cuerpo = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
            mask_color = cv2.morphologyEx(mask_abierta, cv2.MORPH_CLOSE, kernel_cuerpo)
        if color == 'VIOLETA':
            #Apertura
            kernel_cuerpo = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_OPEN, kernel_cuerpo)
        conts, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in conts:
            area = cv2.contourArea(cnt)
            if area > 700:  # filtrar ruido muy pequeño
                bx, by, bw, bh = cv2.boundingRect(cnt)
                centro_x = bx + bw // 2
                bandas_detectadas.append((centro_x, color))
                 #Dibujo para depuración: rectángulo verde alrededor de la banda
                cv2.rectangle(img_rec_bgr, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                # Punto rojo en el centroide horizontal
                cv2.circle(img_rec_bgr, (centro_x, by + bh // 2), 4, (0, 0, 255), -1)
                

    # Mostrar la imagen con detecciones de bandas/colores para debug
    if debug:
        plt.figure(figsize=(8, 4))
        plt.imshow(cv2.cvtColor(img_rec_bgr, cv2.COLOR_BGR2RGB))
        plt.title(f"Bandas detectadas y coloreadas. Resistencia {i}")
        plt.axis('off')
        plt.show()

    # Filtrar duplicados permitiendo hasta 2 bandas del mismo color
    bandas_filtradas = []
    conteo_colores = {}

    # Ordenar por coordenada X antes de filtrar
    bandas_detectadas = sorted(bandas_detectadas, key=lambda b: b[0])
    for centro_x, color in bandas_detectadas:
        if color not in conteo_colores:
            conteo_colores[color] = 1
            bandas_filtradas.append((centro_x, color))
        elif conteo_colores[color] < 2:
            conteo_colores[color] += 1
            bandas_filtradas.append((centro_x, color))

   
    # Ordenar las bandas por su coordenada X (de izquierda a derecha)
    bandas_ord = sorted(bandas_filtradas, key=lambda b: b[0])


    # Ordenamos según posición de la dorada, se utiliza la 1° y 3° banda para estimar el lado de la dorada
    # Obtener ancho de la imagen
    ancho_img = img_rec_hsv.shape[1]

    # Distancia del primer centro al borde izquierdo
    dist_izq = bandas_ord[0][0]

    # Distancia del tercer centro al borde derecho
    dist_der = ancho_img - bandas_ord[2][0]

    # Decisión de inversión
    if dist_izq <= dist_der:
        lista_bandas = bandas_ord.copy()  # se deja como está
        return [color for _, color in lista_bandas]
    elif dist_izq > dist_der:
        lista_bandas = bandas_ord[::-1].copy()  # se invierte
        return [color for _, color in lista_bandas]
 
    return None


# FUNCION RESISTENCIA
def calcular_resistencia(colores):
    """
    Calcula el valor de una resistencia a partir de una lista de 3 colores.
    Ejemplo: ['NEGRO', 'MARRÓN', 'ROJO'] → 0.1 * 100 = 10 ohms
    """
    if len(colores) != 3:
        raise ValueError("La lista debe contener exactamente tres colores")

    decena = colores_resistencia[colores[0]][0]
    unidad = colores_resistencia[colores[1]][1]
    multiplicador = colores_resistencia[colores[2]][2]

    valor = (decena*10 + unidad) * multiplicador
    return valor


# DETERMINACION DE LA RESISTENCIA
def mostrar_valores():
    print("RESULTADOS DE LAS RESISTENCIAS DETECTADAS")
    print("-" * 40)

    for i, resultado in enumerate(resultados, start=1):
        nombre = f"R{i}"
        
        if resultado is None:
            print(f"{nombre}: No se pudieron detectar las bandas.")
        else:
            valor_resistencia = calcular_resistencia(resultado)
            bandas_str = " - ".join(resultado)
            print(f"{nombre}: {bandas_str} → {valor_resistencia} Ω")


# Procesar 10 imágenes de las resistencias Rx_a_out(x del 1 al 10)
resultados = []
for i in range(1, 11):
    ruta = f'PDI_TP/TP2/Resistencias/R{i}_a_out.jpg'
    colores_detectados = detectar_bandas_resistencia(ruta,debug=True)
    resultados.append(colores_detectados)

# Códigos de colores de resistencias
colores_resistencia = {
    "NEGRO":   [0, 0, 1],
    "MARRÓN":  [1, 1, 10],
    "ROJO":    [2, 2, 100],
    "NARANJA": [3, 3, 1000],
    "AMARILLO": [4, 4, 10000],
    "VERDE":   [5, 5, 100000],
    "AZUL":    [6, 6, 1000000],
    "VIOLETA": [7, 7, 10000000],
    "GRIS":    [8, 8, 100000000],
    "BLANCO":  [9, 9, 1000000000]}

# Mostrar resultados
for idx, lista in enumerate(resultados, start=1):
    print(f"R{idx}: {lista}")
mostrar_valores()
