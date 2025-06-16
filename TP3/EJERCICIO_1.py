import cv2
import numpy as np


# --- Leer y grabar un video ------------------------------------------------
cap = cv2.VideoCapture('PDI_TP/TP3/ruta_1.mp4')     # Abro el video de entrada
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))      # Meta-Información del video de entrada
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    #
fps = int(cap.get(cv2.CAP_PROP_FPS))                #
#print(width, height)
out = cv2.VideoWriter('Video-Output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))    # Abro el video de salida

# --- Control para pruebas (mostrar solo ciertos frames) ---
mostrar_frames = False        # False para ejecutar completo
max_segundos = 1             # Mostrar solo N segundos del video
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
        # Paso 1: Definir puntos del polígono
        points = np.array([[(450, 285), (500, 285), (925, 540), (100, 540)]], dtype=np.int32)

        # Paso 2: Crear máscara vacía
        mask_vid = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        # Paso 3: Dibujar el polígono en la máscara
        cv2.fillPoly(mask_vid, points, 255)

        creation = False  # Ya no se vuelve a crear

    # Paso 4: Procesar solo zona dentro del polígono
    zona = cv2.bitwise_and(frame, frame, mask=mask_vid)

    zona_proc = cv2.cvtColor(zona, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])

    mask = cv2.inRange(zona_proc, lower_white, upper_white) # Esta debera ser en la que se fusione el resultado de dos mask
    #mask_invertida = cv2.bitwise_not(mask)


    # ----------- Bloque: Obtener bordes
    # Apertura y Clausura
    A = mask
    B = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    Aop = cv2.morphologyEx(A, cv2.MORPH_OPEN, B)
    Acl = cv2.morphologyEx(A, cv2.MORPH_CLOSE, B)

    # Bordes
    f = Acl #Aop
    se = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    f_mg = cv2.morphologyEx(f, cv2.MORPH_GRADIENT, se)

    # -----------

    lines = cv2.HoughLinesP(
    image=f_mg,
    rho=10,
    theta=np.pi / 180,
    threshold=100, # Votos
    minLineLength=150,
    maxLineGap=50)

    if debug:
        # Crear una imagen negra del mismo tamaño
        line_img = np.zeros((height, width, 3), dtype=np.uint8)
        line_img = cv2.bitwise_not(line_img)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 1)  # líneas verdes

        #frame = cv2.add(fondo, zona_proc_color)
    #zona_proc_color = cv2.cvtColor(zona_proc, cv2.COLOR_GRAY2RGB)  # Para recomponer


    # Paso 5: Recomponer el frame
    #fondo = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask_vid))
    #frame = cv2.add(fondo, zona_proc_color)


    # --- Procesamiento ---------------------------------------------
    #cv2.rectangle(frame, (100,100), (200,200), (0,0,255), 2)            # Proceso el frame...
    # frame = cv2.resize(frame, dsize=(int(width/3), int(height/3)))      # Si el video es muy grande y al usar cv2.imshow() no entra en la pantalla, se lo puede escalar (solo para visualización!)
    cv2.imshow('Work Zona', f_mg)                                          # ... muestro el resultado

    # ---------------------------------------------------------------
    out.write(frame)  # grabo frame --> IMPORTANTE: frame debe tener el mismo tamaño que se definio al crear out.
    if cv2.waitKey(25) & 0xFF == ord('q'):                              # Corto la repoducción si se presiona la tecla "q"
        break

cap.release()               # Cierro el video de entrada
out.release()               # Cierro el video de salida
cv2.destroyAllWindows()     # Destruyo todas las ventanas abiertas