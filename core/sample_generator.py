# ==============================================================================
# PROYECTO 10: DETECTOR INTELIGENTE DE DEFECTOS SUPERFICIALES EN PIEZAS INDUSTRIALES
# INSTITUCIÓN: UNIVERSIDAD TECNOLÓGICA DE QUERÉTARO (UTEQ)
# MODULO: core/sample_generator.py
# ==============================================================================
# RESUMEN DEL ARCHIVO:
# Generador automatizado de imágenes sintéticas de calibración y prueba
# que simulan la textura metalmecánica en escala de grises de 200x200 píxeles
# (acorde al formato del dataset NEU). Provee especímenes precargados
# para las pruebas interactivas de la interfaz en Gradio.
# ==============================================================================

import os
import numpy as np
from PIL import Image, ImageDraw


def generate_samples(output_dir: str = "samples") -> None:
    """
    Genera 4 especímenes sintéticos de prueba representativos para Gradio:
    1. Grieta: Línea sinuosa irregular con alto gradiente de luminancia.
    2. Picadura: Cúmulo de pequeños puntos oscuros de erosión superficial.
    3. Rayadura: Estrías lineales finas continuas.
    4. Conforme: Superficie uniforme de acero cepillado sin defectos.

    :param output_dir: Directorio de destino para guardar los archivos PNG.
    :type output_dir: str
    """
    # Creación del directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    width, height = 200, 200

    # Generador con semilla fija para reproducibilidad
    rng = np.random.RandomState(42)

    # --------------------------------------------------------------------------
    # Muestra 1: Grieta (Crack)
    # --------------------------------------------------------------------------
    # Base metálica con ruido sutil
    base_metal = rng.normal(160, 10, (height, width)).clip(120, 200).astype(np.uint8)
    img_grieta = Image.fromarray(base_metal, mode="L")
    draw = ImageDraw.Draw(img_grieta)
    
    # Trazado de grieta sinuosa de alto contraste
    points = [(20, 30), (50, 60), (70, 85), (100, 110), (130, 125), (160, 160), (185, 175)]
    for i in range(len(points) - 1):
        draw.line([points[i], points[i+1]], fill=20, width=3)
        # Bifurcación secundaria
        if i == 3:
            draw.line([(100, 110), (130, 95), (150, 90)], fill=30, width=2)
    img_grieta.save(os.path.join(output_dir, "muestra_grieta.png"))

    # --------------------------------------------------------------------------
    # Muestra 2: Picadura (Pitting)
    # --------------------------------------------------------------------------
    base_metal_p = rng.normal(155, 8, (height, width)).clip(120, 190).astype(np.uint8)
    img_picadura = Image.fromarray(base_metal_p, mode="L")
    draw_p = ImageDraw.Draw(img_picadura)
    # Dibujo de micro-porosidades dispersas
    for _ in range(45):
        cx, cy = rng.randint(25, 175), rng.randint(25, 175)
        rad = rng.randint(1, 4)
        shade = rng.randint(15, 60)
        draw_p.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=shade)
    img_picadura.save(os.path.join(output_dir, "muestra_picadura.png"))

    # --------------------------------------------------------------------------
    # Muestra 3: Rayadura (Scratch)
    # --------------------------------------------------------------------------
    base_metal_s = rng.normal(170, 7, (height, width)).clip(140, 200).astype(np.uint8)
    img_rayadura = Image.fromarray(base_metal_s, mode="L")
    draw_s = ImageDraw.Draw(img_rayadura)
    # Trazado de estrías diagonales
    draw_s.line([(30, 40), (170, 160)], fill=40, width=2)
    draw_s.line([(50, 30), (180, 140)], fill=50, width=1)
    img_rayadura.save(os.path.join(output_dir, "muestra_rayadura.png"))

    # --------------------------------------------------------------------------
    # Muestra 4: Superficie Conforme (Clean Surface)
    # --------------------------------------------------------------------------
    # Superficie homogénea y limpia con suave cepillado
    base_metal_c = rng.normal(180, 4, (height, width)).clip(165, 195).astype(np.uint8)
    img_conforme = Image.fromarray(base_metal_c, mode="L")
    img_conforme.save(os.path.join(output_dir, "muestra_conforme.png"))


if __name__ == "__main__":
    generate_samples()
    print("Muestras sintéticas generadas exitosamente en el directorio 'samples/'")

