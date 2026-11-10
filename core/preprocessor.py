# ==============================================================================
# PROYECTO 10: DETECTOR INTELIGENTE DE DEFECTOS SUPERFICIALES EN PIEZAS INDUSTRIALES
# INSTITUCIÓN: UNIVERSIDAD TECNOLÓGICA DE QUERÉTARO (UTEQ)
# MODULO: core/preprocessor.py
# ==============================================================================
# RESUMEN DEL ARCHIVO:
# Este módulo implementa la tubería (pipeline) de preprocesamiento y normalización
# de imágenes digitales de superficies metálicas industriales.
# Prepara la matriz de entrada cumpliendo con las especificaciones del dataset
# NEU Surface Defect Database (200x200 píxeles, escala de grises y ecualización),
# garantizando compatibilidad arquitectónica inmediata con los modelos de
# Deep Learning que se integrarán en la Fase 2 del proyecto.
# ==============================================================================

import numpy as np
from PIL import Image, ImageOps
from typing import Tuple, Dict, Any, Optional


class ImagePreprocessor:
    """
    Clase encargada de la validación, normalización de dimensiones, conversión
    de canales de color y extracción de métricas estadísticas de luminancia
    sobre fotografías industriales de piezas metálicas.
    """

    # Dimensión canónica estandarizada por el dataset NEU (Northeastern University)
    CANONICAL_WIDTH: int = 200
    CANONICAL_HEIGHT: int = 200

    def __init__(self, target_size: Tuple[int, int] = (CANONICAL_WIDTH, CANONICAL_HEIGHT)):
        """
        Inicializa el preprocesador configurando la resolución meta de salida.

        :param target_size: Tupla (ancho, alto) en píxeles a la que se normalizará la imagen.
        :type target_size: Tuple[int, int]
        """
        # Asignación de la resolución objetivo para tensores de entrada
        self.target_size = target_size

    def validate_and_load(self, image_input: Any) -> Image.Image:
        """
        Verifica la validez del espécimen de entrada y lo convierte a formato PIL.Image.
        Soporta instancias de PIL.Image, arreglos NumPy, rutas en disco y diccionarios
        de capas generados por componentes interactivos de Gradio.

        :param image_input: Instancia de PIL.Image, arreglo NumPy, ruta de archivo o dict de Gradio.
        :type image_input: Any
        :return: Objeto imagen PIL debidamente instanciado.
        :rtype: Image.Image
        :raises ValueError: Si la imagen es nula o el formato no es compatible.
        """
        # Validación de presencia: la imagen no debe ser None
        if image_input is None:
            raise ValueError("Error de captura: No se proporcionó ninguna fotografía de superficie.")

        # Si el input ya es una instancia de Pillow, se retorna directamente
        if isinstance(image_input, Image.Image):
            return image_input

        # Si el input proviene como diccionario estructurado desde Gradio (Image Editor / Layers)
        if isinstance(image_input, dict):
            if "composite" in image_input and image_input["composite"] is not None:
                return self.validate_and_load(image_input["composite"])
            if "background" in image_input and image_input["background"] is not None:
                return self.validate_and_load(image_input["background"])

        # Si el input proviene de Gradio como matriz de NumPy (RGB o escala de grises)
        if isinstance(image_input, np.ndarray):
            # Conversión de la matriz numérica de píxeles a objeto PIL
            return Image.fromarray(image_input)

        # Si se trata de una ruta en disco en formato string
        if isinstance(image_input, str):
            # Apertura segura del archivo desde el sistema de archivos
            return Image.open(image_input)

        # Si no corresponde a ninguno de los tipos soportados, se levanta excepción
        raise ValueError(f"Formato no soportado para preprocesamiento: {type(image_input)}")

    def preprocess_for_inference(self, pil_image: Image.Image) -> Dict[str, Any]:
        """
        Ejecuta la tubería de transformación para la red neuronal (Fase 2):
        1. Conversión de espacio de color RGB a escala de grises (1 canal).
        2. Redimensionamiento a 200x200 con interpolación bilineal de alta fidelidad.
        3. Normalización flotante de luminancia en el rango [0.0, 1.0].
        4. Cálculo de métricas estadísticas de superficie (media, desviación estándar, contraste).

        :param pil_image: Imagen cruda capturada del espécimen.
        :type pil_image: Image.Image
        :return: Diccionario con la imagen procesada en PIL, el tensor normalizado y métricas.
        :rtype: Dict[str, Any]
        """
        # Paso 1: Conversión a escala de grises ('L') como establece el estándar NEU
        grayscale_image = pil_image.convert("L")

        # Paso 2: Redimensionamiento espacial al tamaño canónico (200x200 píxeles)
        resized_image = grayscale_image.resize(self.target_size, Image.Resampling.BILINEAR)

        # Paso 3: Conversión a arreglo NumPy para operaciones matemáticas vectorizadas
        pixel_array = np.array(resized_image, dtype=np.float32)

        # Paso 4: Normalización Min-Max al rango unitario [0.0, 1.0] para redes convolucionales
        normalized_tensor = pixel_array / 255.0

        # Paso 5: Cálculo de métricas de calidad de imagen para control de sensor
        mean_luminance = float(np.mean(pixel_array))
        std_contrast = float(np.std(pixel_array))
        min_val = float(np.min(pixel_array))
        max_val = float(np.max(pixel_array))

        # Paso 6: Verificación de subexposición o sobreexposición lumínica en la captura
        is_poor_lighting = (mean_luminance < 30.0) or (mean_luminance > 225.0) or (std_contrast < 15.0)

        # Empaquetado de artefactos procesados
        return {
            "processed_pil": resized_image,
            "normalized_tensor": normalized_tensor,
            "mean_luminance": mean_luminance,
            "std_contrast": std_contrast,
            "dynamic_range": (min_val, max_val),
            "is_poor_lighting": is_poor_lighting,
            "dimensions": self.target_size
        }

