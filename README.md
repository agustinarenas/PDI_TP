# PDI_TP

Este repositorio contiene el desarrollo del Trabajo Práctico de la materia **Procesamiento Digital de Imágenes**.

## 📁 Estructura del repositorio

```
PDI_TP/
│
├── TP1/
│   ├── Imagen_con_detalles_escondidos.tif
│   ├── ecualizacion_local.py
│   ├── otros_archivos_utiles.py
│   └── ...
│
├── README.md
```

> ⚠️ **Importante**: Abrir el proyecto desde la carpeta raíz `PDI_TP/` para que las rutas relativas a las imágenes funcionen correctamente.

---

## ▶️ Instrucciones para correr el TP

1. **Cloná el repositorio**:

   ```bash
   git clone https://github.com/tu-usuario/PDI_TP.git
   cd PDI_TP
   ```

2. **Instalá las bibliotecas necesarias** (si no las tenés):

   ```bash
   pip install opencv-python numpy matplotlib
   ```

3. **Abrí el proyecto en VSCode desde la carpeta raíz `PDI_TP/`.**

4. Navegá a la carpeta `TP1/` y ejecutá los scripts `.py` según corresponda.

---

## 🧰 Bibliotecas utilizadas

- [`cv2`](https://pypi.org/project/opencv-python/) – OpenCV para procesamiento de imágenes.
- [`numpy`](https://numpy.org/) – Para operaciones numéricas y matriciales.
- [`matplotlib.pyplot`](https://matplotlib.org/stable/api/pyplot_api.html) – Para visualizar resultados.

---

## ℹ️ Notas

- Todos los archivos necesarios (imágenes, scripts, etc.) están dentro de la carpeta `TP1/`.
- Se recomienda usar **Python 3.8 o superior**.
- La ecualización local se aplica pixel a pixel, usando una ventana deslizante, y permite resaltar detalles ocultos en distintas zonas de la imagen.

---

## 🧑‍💻 Autor

- [Tu Nombre o Usuario de GitHub]
