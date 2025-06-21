import cv2
import numpy as np

# --- Función para determinar una unica linea ---
def encontrar_extremos_mas_separados(grupo):
    puntos = []
    for linea in grupo:
        x1, y1, x2, y2 = linea[0]  # porque cada 'linea' es [[x1, y1, x2, y2]]
        puntos.append((x1, y1))
        puntos.append((x2, y2))

    max_dist = 0
    punto1, punto2 = None, None

    for i in range(len(puntos)):
        for j in range(i + 1, len(puntos)):
            dist = np.linalg.norm(np.array(puntos[i]) - np.array(puntos[j])) # Distancia euclidiana
            if dist > max_dist:
                max_dist = dist
                punto1, punto2 = puntos[i], puntos[j]

    return [punto1[0], punto1[1], punto2[0], punto2[1]]

# --- Función para generar un rectángulo de influencia alrededor de una línea ---
def generate_influence_area(x1, y1, x2, y2, length_extension, thickness):
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    dx /= length
    dy /= length
    perp_dx = -dy
    perp_dy = dx
    lx = dx * (length_extension / 2)
    ly = dy * (length_extension / 2)
    px = perp_dx * (thickness / 2)
    py = perp_dy * (thickness / 2)
    p1 = (cx - lx - px, cy - ly - py)
    p2 = (cx + lx - px, cy + ly - py)
    p3 = (cx + lx + px, cy + ly + py)
    p4 = (cx - lx + px, cy - ly + py)
    rect = np.array([p1, p2, p3, p4], dtype=np.float32)
    return rect

# --- Función para verificar si ambos extremos están dentro de un polígono ---
def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

# --- Función para ordenar puntos por angulo ---
def ordenar_puntos_por_angulo(puntos, dist_max=None):
    puntos = np.array(puntos, dtype=np.float32)
    
    # Calcular centroide
    centroide = np.mean(puntos, axis=0)

    # Ordenar por ángulo respecto al centroide
    def angulo(p):
        return np.arctan2(p[1] - centroide[1], p[0] - centroide[0])
    
    puntos_ordenados = sorted(puntos, key=angulo)
    puntos_ordenados = np.array(puntos_ordenados)

    if dist_max is not None and len(puntos_ordenados) > 1:
        i = 0
        while i < len(puntos_ordenados) - 1:
            p1 = puntos_ordenados[i]
            p2 = puntos_ordenados[i + 1]
            dist = np.linalg.norm(p2 - p1)

            if dist > dist_max:
                # Eliminar el punto alejado (p2) porque p1 respetaba la distancia
                puntos_ordenados = np.delete(puntos_ordenados, i + 1, axis=0)
                # No incrementar i para volver a comparar p1 con el nuevo p2
            else:
                i += 1

        # También verificar el cierre entre el último y el primero
        if len(puntos_ordenados) > 2:
            dist = np.linalg.norm(puntos_ordenados[0] - puntos_ordenados[-1])
            if dist > dist_max:
                puntos_ordenados = puntos_ordenados[:-1]

    return puntos_ordenados

# --- Función para dibujar poligonales con relleno ---
def dibujar_poligonales_con_relleno(grupos, frame, color=(0, 255, 0), alpha=0.4, borde_color=(0, 200, 0), thickness=2):
    overlay = frame.copy()

    for grupo in grupos:
        puntos = [(l[0][0], l[0][1]) for l in grupo] + [(l[0][2], l[0][3]) for l in grupo]
        puntos = list(set(puntos))

        if len(puntos) >= 3:
            puntos_ordenados = ordenar_puntos_por_angulo(puntos, dist_max=150) # dist_max = 150 anda bien con video 1 por lo menos
            pts_cv2 = puntos_ordenados.reshape((-1, 1, 2)).astype(np.int32)

            # Rellenar sobre el overlay
            cv2.fillPoly(overlay, [pts_cv2], color=color)

            # Dibujar borde sobre la imagen original (frame)
            cv2.polylines(frame, [pts_cv2], isClosed=True, color=borde_color, thickness=thickness)

    # Mezclar overlay con la imagen original
    frame_resultado = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return frame_resultado


# --- Leer y grabar un video ------------------------------------------------
cap = cv2.VideoCapture('PDI_TP/TP3/ruta_2.mp4')     # Abro el video de entrada
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))      # Meta-Información del video de entrada
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    
fps = int(cap.get(cv2.CAP_PROP_FPS))                
#print(width, height)

out = cv2.VideoWriter('Video-Output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))    # Abro el video de salida

# --- Control para pruebas (mostrar solo ciertos frames) ---
mostrar_frames = False        # False para ejecutar completo
max_segundos = 5           # Mostrar solo N segundos del video
max_frames = fps * max_segundos
frame_count = 0

creation = True  # Solo para la primera vez
mask = None      # Para reutilizar en los demás frames

debug = True

while (cap.isOpened()):         # Itero, siempre y cuando el video esté abierto
    ret, frame = cap.read()     # Obtengo el frame
    if ret==True:
    # ----------- Control de duración
        frame_count += 1
        if mostrar_frames and frame_count > max_frames:
            break
    # -----------
    else:
        break    # Corto la reproducción si ret=False, es decir, si hubo un error o no quedán mas frames.

    if creation:
        # Definir puntos del polígono
        #points = np.array([[(450, 285), (500, 285), (925, 540), (100, 540)]], dtype=np.int32)
        points = np.array([[(455, 318), (520, 318), (925, 540), (100, 540)]], dtype=np.int32)

        # Crear máscara vacía
        mask_vid = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        # Dibujar el polígono en la máscara
        cv2.fillPoly(mask_vid, points, 255)

        creation = False  # Ya no se vuelve a crear

    # Procesar solo zona dentro del polígono
    zona = cv2.bitwise_and(frame, frame, mask=mask_vid)

    # -------------------------------------------------------
    #cv2.imshow('Work Zona', zona)
    # -------------------------------------------------------

    # Convertir a HSV
    zona_proc = cv2.cvtColor(zona, cv2.COLOR_BGR2HSV)

    # --- Rango para blanco ---
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 60, 255])
    mask_white = cv2.inRange(zona_proc, lower_white, upper_white)

    # --- Rango para amarillo ---
    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([45, 255, 255])
    mask_yellow = cv2.inRange(zona_proc, lower_yellow, upper_yellow)

    # --- Combinar máscaras ---
    mask = cv2.bitwise_or(mask_white, mask_yellow)
    #mask_invertida = cv2.bitwise_not(mask)


    # ----------- Bloque: Corregir
    # Apertura y Clausura
    A = mask
    B = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    #Acl = cv2.morphologyEx(A, cv2.MORPH_OPEN, B)
    Acl = cv2.morphologyEx(A, cv2.MORPH_CLOSE, B)
    # -----------

    contours, _ = cv2.findContours(Acl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # -------------------------------------------------------
    #cv2.imshow('Work Zona', mask)
    # -------------------------------------------------------


    # Crear una imagen en blanco del mismo tamaño que la máscara
    output = np.zeros_like(mask)
    
    #cnt_cuerpo = max(contours, key=cv2.contourArea)
    min_area = 5

    # Filtrar los contornos por área
    filtered_contours_area = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

    # Dibujar los contornos en blanco sobre fondo negro
    cv2.drawContours(output, filtered_contours_area, -1, (255), 2)  # 255 para blanco, 2 es el grosor


    # -------------------------------------------------------
    #cv2.imshow('Work Zona', output)
    # -------------------------------------------------------


    # ------------------------------ DETECTAR LINEAS
    lines = cv2.HoughLinesP(
    image=output,
    rho= 10,
    theta= np.pi / 180,
    threshold= 10, # Votos
    minLineLength= 10,
    maxLineGap= 35)

    if False: #debug:
        # Crear una imagen negra del mismo tamaño
        line_img = np.zeros((height, width, 3), dtype=np.uint8)
        line_img = cv2.bitwise_not(line_img)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # líneas verdes
    
    # -------------------------------------------------------
    #cv2.imshow('Work Zona', frame)
    # -------------------------------------------------------


    # ------------------------------ AGRUPAR LINEAS
    img_height, img_width = height, width
    max_side = max(img_height, img_width)

    # --- Parámetro ajustable ---
    border_thickness = 50

    remaining_lines = lines.tolist()
    grupos = []

    while remaining_lines:
        seed_line = remaining_lines.pop(0)
        x1, y1, x2, y2 = seed_line[0]
        rect = generate_influence_area(x1, y1, x2, y2, max_side, border_thickness)
        grupo_actual = [seed_line]
        restantes = []
        for line in remaining_lines:
            a1, b1, a2, b2 = line[0]
            if point_in_polygon((a1, b1), rect) and point_in_polygon((a2, b2), rect):
                grupo_actual.append(line)
            else:
                restantes.append(line)
        grupos.append(grupo_actual)
        remaining_lines = restantes

    if False: #debug:
        # --- Visualización: cada grupo de un solo color ---
        img_grouped = np.zeros((height, width, 3), dtype=np.uint8)
        img_grouped = cv2.bitwise_not(img_grouped)

        group_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                        (0, 255, 255), (255, 0, 255), (128, 128, 0), (255, 128, 0)]

        for i, grupo in enumerate(grupos):
            color = group_colors[i % len(group_colors)] # En caso de haber mas grupos, repite colores; da la vuelta
            for line in grupo:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), color, 1)
 
    # -------------------------------------------------------
    frame_con_poligonales = dibujar_poligonales_con_relleno(grupos, frame, color=(0,255,0), alpha=0.4)
    cv2.imshow("Poligonales", frame_con_poligonales)
    # -------------------------------------------------------


    out.write(frame)  # grabo frame --> IMPORTANTE: frame debe tener el mismo tamaño que se definio al crear out.
    if cv2.waitKey(25) & 0xFF == ord('q'):                              # Corto la repoducción si se presiona la tecla "q"
        break

cap.release()               # Cierro el video de entrada
out.release()               # Cierro el video de salida
cv2.destroyAllWindows()     # Destruyo todas las ventanas abiertas
