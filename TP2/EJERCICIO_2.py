import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

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

# Ordenar puntos detectados para homografía
def ordenar_puntos(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]       # top-left
    rect[2] = pts[np.argmax(s)]       # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]    # top-right
    rect[3] = pts[np.argmax(diff)]    # bottom-left
    return rect

# Calcular intersección de dos líneas
def interseccion(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1

    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    det = A1 * B2 - A2 * B1
    if det == 0:
        return None
    x = (B2 * C1 - B1 * C2) / det
    y = (A1 * C2 - A2 * C1) / det
    return [x, y]

# Detectar contorno más grande y líneas extendidas
def detectar_lineas_y_puntos(mask, debug=False, original_img=None):
    # Clausura morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if debug:
        imshow(mask, title="Máscara Azul")
        imshow(closed, title="Tras Clausura")

    # Filtrar contorno más grande
    cnts = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    filtered_mask = np.zeros_like(mask)
    cv2.drawContours(filtered_mask, [c], -1, 255, -1)

    if debug:
        imshow(filtered_mask, title="Máscara con Mayor Contorno")

    # Aproximar contorno a líneas rectas
    epsilon = 0.02 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)
    approx = approx.reshape(-1, 2)

    if len(approx) < 4:
        raise ValueError("Contorno no tiene suficientes puntos para líneas")

    # Crear líneas extendidas (hasta borde de imagen)
    h, w = mask.shape[:2]
    extended_lines = []
    for i in range(len(approx)):
        pt1 = approx[i]
        pt2 = approx[(i + 1) % len(approx)]
        vx = pt2[0] - pt1[0]
        vy = pt2[1] - pt1[1]
        if vx == 0 and vy == 0:
            continue
        norm = np.hypot(vx, vy)
        vx, vy = vx / norm, vy / norm
        start = (int(pt1[0] - 1000 * vx), int(pt1[1] - 1000 * vy))
        end = (int(pt1[0] + 1000 * vx), int(pt1[1] + 1000 * vy))
        extended_lines.append([*start, *end])

    # Obtener intersecciones entre líneas consecutivas
    intersecciones = []
    for i in range(len(extended_lines)):
        l1 = extended_lines[i]
        l2 = extended_lines[(i + 1) % len(extended_lines)]
        punto = interseccion(l1, l2)
        if punto:
            intersecciones.append(punto)

    if len(intersecciones) < 4:
        raise ValueError("No se detectaron 4 intersecciones")

    puntos = np.array(intersecciones[:4], dtype="float32")

    if debug and original_img is not None:
        debug_img = original_img.copy()
        for line in extended_lines:
            x1, y1, x2, y2 = line
            cv2.line(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for pt in puntos:
            cv2.circle(debug_img, tuple(np.int32(pt)), 6, (0, 255, 0), -1)
        imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), title="Líneas y Puntos de Intersección", color_img=True)

    return ordenar_puntos(puntos)

# Aplicar homografía para vista superior
def aplicar_homografia(img, debug=False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([105, 50, 50])
    upper_blue = np.array([135, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    pts_src = detectar_lineas_y_puntos(mask, debug=debug, original_img=img)

    (tl, tr, br, bl) = pts_src
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    pts_dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped

# Ejecutar con imagen de prueba
img = cv2.imread('PDI_TP/TP2/Resistencias/R2_c.jpg')
resultado = aplicar_homografia(img, debug=True)

plt.imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
