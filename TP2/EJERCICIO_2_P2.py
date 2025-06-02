import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils


# CODIGOS DE COLORES DE RESISTENCIAS

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
    "BLANCO":  [9, 9, 1000000000]
}


# FUNCION RESISTENCIA

def calcular_resistencia(colores):
    """
    Calcula el valor de una resistencia a partir de una lista de 3 colores.
    Ejemplo: ['NEGRO', 'MARRÓN', 'ROJO'] → 0.1 * 100 = 10 ohms
    """
    if len(colores) != 3:
        raise ValueError("La lista debe contener exactamente tres colores")

    entero = colores_resistencia[colores[0]][0]
    decimal = colores_resistencia[colores[1]][1]
    multiplicador = colores_resistencia[colores[2]][2]

    valor = (entero + decimal / 10) * multiplicador
    return valor


# DETERMINACION DE LA RESISTENCIA

import os

# Ruta completa del archivo
ruta = 'PDI_TP/TP2/Resistencias/R10_a_out.jpg'

# Extraer el nombre del archivo sin extensión
base = os.path.splitext(os.path.basename(ruta))[0]

# Colores detectados
colores = ["NEGRO", "MARRÓN", "ROJO"]

# Calcular valor de resistencia
valor_resistencia = calcular_resistencia(colores)

# Mostrar resultados
print(f"RESISTENCIA {base}")
print(f"Valor: {valor_resistencia} Ω")