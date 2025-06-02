import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

# Ejecutar con imagen de prueba
ruta = 'PDI_TP/TP2/Resistencias/R10_a.jpg'


img = cv2.imread(ruta)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([105, 50, 50])
upper_blue = np.array([135, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
mask_invertida = cv2.bitwise_not(mask)

#-------

# APERTURA
A = mask_invertida
B = cv2.getStructuringElement(cv2.MORPH_RECT, (75,75))
Aop = cv2.morphologyEx(A, cv2.MORPH_OPEN, B)

plt.imshow(cv2.cvtColor(Aop, cv2.COLOR_BGR2RGB))
plt.title("IMAGEN DE DISMINUIR LA RESISTENCIA")
plt.axis("off")
plt.show()

#-------

# BORDES
f = Aop #Fe
se = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
f_mg = cv2.morphologyEx(f, cv2.MORPH_GRADIENT, se)


# Suponemos que f_mg es la imagen base usada para HoughLinesP
height, width = f_mg.shape[:2]

# Crear una imagen negra del mismo tamaño
line_img = np.zeros((height, width, 3), dtype=np.uint8)

lines = cv2.HoughLinesP(
    image=f_mg,
    rho=10,
    theta=np.pi / 180,
    threshold=90,
    minLineLength=150,     # ¡ajustá esto!
    maxLineGap=50         # ¡ajustá esto también!
)


# Dibujar líneas en la imagen negra
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # líneas verdes

# Mostrar resultado
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
plt.title("Sólo las líneas detectadas")
plt.axis('off')
plt.show()



img_height, img_width = height, width
max_side = max(img_height, img_width)

# --- Parámetro ajustable ---
border_thickness = 60  # "ancho de borde típico"

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

# --- Visualización: cada grupo de un solo color ---
img_grouped = np.zeros((img_height, img_width, 3), dtype=np.uint8)
group_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (0, 255, 255), (255, 0, 255), (128, 128, 0), (255, 128, 0)]

for i, grupo in enumerate(grupos):
    color = group_colors[i % len(group_colors)]
    for line in grupo:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_grouped, (x1, y1), (x2, y2), color, 2)

plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(img_grouped, cv2.COLOR_BGR2RGB))
plt.title(f"Líneas agrupadas (cada grupo un color) - Cantidad de grupos: {len(grupos)}")
plt.axis("off")
plt.show()

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
            dist = np.linalg.norm(np.array(puntos[i]) - np.array(puntos[j]))
            if dist > max_dist:
                max_dist = dist
                punto1, punto2 = puntos[i], puntos[j]

    return [punto1[0], punto1[1], punto2[0], punto2[1]]




# Lista final con una sola línea representativa por grupo
grupos_unificado = []

for grupo in grupos:
    linea_unificada = encontrar_extremos_mas_separados(grupo)
    grupos_unificado.append([linea_unificada])  # se guarda con misma estructura: lista de lista

# Visualización opcional
img_lineas = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

colores = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # un color por grupo

for i, linea in enumerate(grupos_unificado):
    x1, y1, x2, y2 = linea[0]
    color = colores[i % len(colores)]
    cv2.line(img_lineas, (x1, y1), (x2, y2), color, 4)

plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(img_lineas, cv2.COLOR_BGR2RGB))
plt.title("Líneas representativas por grupo")
plt.axis("off")
plt.show()

#------------

# --- Paso 1: calcular longitud de cada línea ---
lineas_con_longitud = []
for linea in grupos_unificado:
    x1, y1, x2, y2 = linea[0]
    longitud = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    lineas_con_longitud.append((longitud, linea))

# Ordenar por longitud descendente
lineas_con_longitud.sort(reverse=True, key=lambda x: x[0])

# --- Paso 2: obtener las dos líneas más largas y sus puntos medios ---
linea1 = lineas_con_longitud[0][1][0]
linea2 = lineas_con_longitud[1][1][0]

# Calcular puntos medios
mx1 = (linea1[0] + linea1[2]) // 2
my1 = (linea1[1] + linea1[3]) // 2
mx2 = (linea2[0] + linea2[2]) // 2
my2 = (linea2[1] + linea2[3]) // 2

# Punto intermedio entre los dos puntos medios
cx = (mx1 + mx2) // 2
cy = (my1 + my2) // 2

# --- Paso 3: filtrar líneas dentro del área de influencia ---
radio = 250  # Ajustable
grupo_filtrado = []

for _, linea in lineas_con_longitud:
    x1, y1, x2, y2 = linea[0]
    
    dist1 = np.sqrt((x1 - cx)**2 + (y1 - cy)**2)
    dist2 = np.sqrt((x2 - cx)**2 + (y2 - cy)**2)
    
    # Si al menos un extremo está fuera del área circular, conservar
    if dist1 > radio or dist2 > radio:
        grupo_filtrado.append((np.sqrt((x2 - x1)**2 + (y2 - y1)**2), linea))

# --- Paso 4: ordenar de nuevo y tomar las 4 más largas ---
grupo_filtrado.sort(reverse=True, key=lambda x: x[0])
grupo_final = [linea for _, linea in grupo_filtrado[:4]]

# --- Visualización ---
img_lineas_f = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
colores = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

for i, linea in enumerate(grupo_final):
    x1, y1, x2, y2 = linea[0]
    color = colores[i % len(colores)]
    cv2.line(img_lineas_f, (x1, y1), (x2, y2), color, 4)

# Dibujar el círculo de influencia también (opcional)
cv2.circle(img_lineas_f, (cx, cy), radio, (255, 255, 255), 2)

plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(img_lineas_f, cv2.COLOR_BGR2RGB))
plt.title("Líneas finales (con filtro de área central)")
plt.axis("off")
plt.show()

#------------

# Control de cuánto estirar en píxeles
estirar = 1000  # Ajustalo a lo que necesites

grupos_estirado = []

for linea in grupo_final:
    x1, y1, x2, y2 = linea[0]
    
    # Vector director de la línea
    dx = x2 - x1
    dy = y2 - y1
    longitud = np.hypot(dx, dy)
    
    if longitud == 0:
        # Evitar división por cero en líneas degeneradas
        grupos_estirado.append([[x1, y1, x2, y2]])
        continue

    # Vector unitario
    ux = dx / longitud
    uy = dy / longitud

    # Estirar en ambos extremos
    nuevo_x1 = int(x1 - ux * estirar)
    nuevo_y1 = int(y1 - uy * estirar)
    nuevo_x2 = int(x2 + ux * estirar)
    nuevo_y2 = int(y2 + uy * estirar)

    grupos_estirado.append([[nuevo_x1, nuevo_y1, nuevo_x2, nuevo_y2]])


# Crear imagen negra en color (3 canales) del mismo tamaño que la original
img_estirada = np.zeros((f_mg.shape[0], f_mg.shape[1], 3), dtype=np.uint8)

# Colores distintos para hasta 4 grupos
colores = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

for i, linea in enumerate(grupos_estirado):
    x1, y1, x2, y2 = linea[0]  # ← Aseguramos que estamos accediendo correctamente
    color = colores[i % len(colores)]  # Aseguramos que se repitan si hay más de 4
    cv2.line(img_estirada, (x1, y1), (x2, y2), color, 3)

# Mostrar con matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(img_estirada, cv2.COLOR_BGR2RGB))  # Convertimos de BGR a RGB
plt.title("Líneas estiradas (una por grupo)")
plt.axis("off")
plt.show()


def interseccion(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if denom == 0:
        return None  # Líneas paralelas
    
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    
    return int(round(px)), int(round(py))

def distancia(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5


# Calcular puntos únicos de intersección entre todas las líneas
puntos_interseccion = []
num_lineas = len(grupos_estirado)
umbral_distancia = 5  # píxeles para considerar puntos iguales

for i in range(num_lineas):
    line_a = grupos_estirado[i][0]       # extraer primer elemento
    for j in range(i+1, num_lineas):
        line_b = grupos_estirado[j][0]
        pt = interseccion(line_a, line_b)
        if pt:
            x, y = pt
            if 0 <= x < f_mg.shape[1] and 0 <= y < f_mg.shape[0]:
                # Verificar que no esté ya en la lista (puntos cercanos)
                if not any(distancia(pt, p_exist) < umbral_distancia for p_exist in puntos_interseccion):
                    puntos_interseccion.append(pt)


# Asegurarse que hay exactamente 4 puntos
if len(puntos_interseccion) != 4:
    raise ValueError("Se requieren exactamente 4 puntos para aplicar homografía.")

# Crear imagen negra con el mismo tamaño que la original
img_puntos = np.zeros((f_mg.shape[0], f_mg.shape[1], 3), dtype=np.uint8)

# Dibujar puntos en la imagen negra
for pt in puntos_interseccion:
    cv2.circle(img_puntos, pt, radius=10, color=(0, 255, 0), thickness=-1)  # verde

# Mostrar con matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(img_puntos, cv2.COLOR_BGR2RGB))
plt.title("Puntos de intersección")
plt.axis("off")
plt.show()


# Ordenar los puntos: superior izquierda, superior derecha, inferior derecha, inferior izquierda
def ordenar_puntos(puntos):
    puntos = np.array(puntos, dtype=np.float32)

    # Ordenar por eje Y (de arriba a abajo)
    puntos_ordenados = puntos[np.argsort(puntos[:,1])]

    # Tomar dos de arriba y dos de abajo
    top = puntos_ordenados[:2]
    bottom = puntos_ordenados[2:]

    # Ordenar cada par por eje X
    top = top[np.argsort(top[:,0])]
    bottom = bottom[np.argsort(bottom[:,0])]

    # Resultado: [sup izq, sup der, inf der, inf izq]
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)



# Ordenar los puntos de intersección
pts_origen = ordenar_puntos(puntos_interseccion)

# Calcular tamaño destino en función de distancias reales
width_top = np.linalg.norm(pts_origen[0] - pts_origen[1])
width_bottom = np.linalg.norm(pts_origen[3] - pts_origen[2])
height_left = np.linalg.norm(pts_origen[0] - pts_origen[3])
height_right = np.linalg.norm(pts_origen[1] - pts_origen[2])

width = int(max(width_top, width_bottom))
height = int(max(height_left, height_right))

# Puntos destino: rectángulo "plano"
pts_destino = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
], dtype=np.float32)

# Calcular la matriz de homografía
H, _ = cv2.findHomography(pts_origen, pts_destino)

# Aplicar warp
img_estirada = cv2.warpPerspective(img, H, (width, height))

# Mostrar resultados
plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.scatter(*zip(*puntos_interseccion), color='red', s=50)
plt.title("Imagen original con puntos")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_estirada, cv2.COLOR_BGR2RGB))
plt.title("Imagen corregida (homografía)")
plt.axis("off")

plt.show()

import os

# Separar base y extensión
base, ext = os.path.splitext(ruta)

# Crear nueva ruta con sufijo "_out"
ruta_salida = base + '_out' + ext

# Guardar imagen procesada
cv2.imwrite(ruta_salida, img_estirada)
