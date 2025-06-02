import cv2
import numpy as np
import matplotlib.pyplot as plt

def detectar_bandas_resistencia(img_path):
    """
    Lee la imagen de la resistencia, recorta el cuerpo sobre fondo azul
    y devuelve una lista con los 3 colores de banda (sin la dorada),
    ordenados de la más alejada a la más cercana a la banda dorada.
    Si falla, retorna None.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Rangos HSV para cada color (aproximados)
    rangos_colores = {
        'NEGRO':   ([100, 0, 0], [180, 100, 50]),
        'MARRÓN':  ([100, 100, 60], [120, 255, 110]),
        'ROJO':   ([0, 100, 100], [10, 255, 255]),
        'NARANJA': ([0, 160, 150], [110, 220, 190]),
        'AMARILLO':([25, 100, 100], [35, 255, 255]),
        'VERDE':   ([35, 50, 50], [85, 255, 255]),
        'AZUL':    ([100, 50, 180], [120, 100, 255]),
        'VIOLETA': ([130, 50, 50], [160, 255, 255]),
        'GRIS':    ([0, 0, 100], [180, 50, 200]),
        'BLANCO':  ([0, 0, 150], [20, 80, 255]),
        'DORADO':  ([0, 0, 0], [255, 255, 255])
    }
    # 1) Detectar el cuerpo de la resistencia: segmento azul invertido y apertura
    lower_blue = np.array([105, 50, 50])
    upper_blue = np.array([135, 255, 255])
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
    mask_inv = cv2.bitwise_not(mask_blue)
    kernel_cuerpo = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    mask_abierta = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel_cuerpo)

    # 2) Obtener contorno más grande (cuerpo)
    contours, _ = cv2.findContours(mask_abierta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt_cuerpo = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt_cuerpo)
    img_rec_hsv = img_hsv[y:y + h, x:x + w]

    # 3) Detectar bandas de cada color
    bandas_detectadas = []
    for color, (low, high) in rangos_colores.items():
        low_hsv = np.array(low)
        high_hsv = np.array(high)
        mask_color = cv2.inRange(img_rec_hsv, low_hsv, high_hsv)
        conts, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in conts:
            area = cv2.contourArea(cnt)
            if area > 200:  # filtrar ruido
                bx, by, bw, bh = cv2.boundingRect(cnt)
                centro_x = bx + bw // 2
                bandas_detectadas.append((centro_x, color))

    # 4) Agrupar por color evitando duplicados
    bandas_filtradas = {}
    for centro_x, color in bandas_detectadas:
        if color not in bandas_filtradas:
            bandas_filtradas[color] = centro_x
        else:
            bandas_filtradas[color] = (bandas_filtradas[color] + centro_x) // 2

    bandas_finales = [(x_pos, color) for color, x_pos in bandas_filtradas.items()]
    if len(bandas_finales) < 4:
        return None
    
    # 5) Ordenar por coordenada X
    bandas_ord = sorted(bandas_finales, key=lambda b: b[0])

    # 6) Calcular diferencias consecutivas y encontrar la mayor (dorado)
    difs = [(bandas_ord[i + 1][0] - bandas_ord[i][0], i) for i in range(len(bandas_ord) - 1)]
    difs_sorted = sorted(difs, reverse=True)
    _, idx = difs_sorted[0]
    # 8) Ordenar las 3 bandas restantes por distancia a la dorada
    bandas_color = bandas_ord[:idx+1] + bandas_ord[idx+1+1:]  # salteás la dorada
    bandas_color_sorted = sorted(bandas_color, key=lambda b: b[0])
    lista_bandas = []
    banda1 = bandas_color_sorted[0][1]
    banda2 = bandas_color_sorted[1][1]
    banda3 = bandas_color_sorted[2][1]
    lista_bandas.extend([banda1,banda2,banda3])
    return lista_bandas

# Ejemplo: procesar 10 imágenes con la ruta indicada
resultados = []
for i in range(1, 11):
    ruta = f'Procesamiento de imagenes 1/Tp_2_Pdi/Resistencias/R{i}_a_out.jpg'
    colores_detectados = detectar_bandas_resistencia(ruta)
    resultados.append(colores_detectados)

# Mostrar resultados
for idx, lista in enumerate(resultados, start=1):
    print(f"R{idx}: {lista}")

###FALTA: Acomodar el rango de los colores, los que no toma bien son el rojo,negro y amarillo, los otro los lee bastante bien(Hay que leer imagen en hsv y ver mas o menos para acomodarlos)
#Ya que las imagenes tienen distintas luminosidad y borrosidad y demas factores que intervienen en el color
#Ver tema de cuando hay dos bandas de un mismo color
#Probar
def filtrar_bandas_proximas(bandas, min_dist=20):
    """
    Filtra bandas muy cercanas entre sí para evitar múltiples detecciones del mismo color en el mismo lugar.
    Parámetros:
        bandas: lista de tuplas (x, color)
        min_dist: distancia mínima entre bandas para considerarlas distintas
    Retorna:
        lista filtrada
    """
    bandas = sorted(bandas, key=lambda b: b[0])
    filtradas = []

    for banda in bandas:
        if all(abs(banda[0] - b2[0]) > min_dist for b2 in filtradas):
            filtradas.append(banda)

    return filtradas