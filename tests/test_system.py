# ==============================================================================
# PROYECTO 10: DETECTOR INTELIGENTE DE DEFECTOS SUPERFICIALES EN PIEZAS INDUSTRIALES
# INSTITUCIÓN: UNIVERSIDAD TECNOLÓGICA DE QUERÉTARO (UTEQ)
# MODULO: tests/test_system.py
# ==============================================================================
# RESUMEN DEL ARCHIVO:
# Suite integral de pruebas unitarias y de integración para validar la robustez
# matemática y lógica de todo el sistema de inspección:
# 1. Pipeline de preprocesamiento de imágenes (ImagePreprocessor).
# 2. Motor perceptual desacoplado (MockDefectClassifier).
# 3. Motor determinista de reglas heurísticas de triaje (TriageRuleEngine).
# 4. Controladores de eventos de la interfaz Gradio (Callbacks de app.py).
# ==============================================================================

import unittest
import numpy as np
from PIL import Image

from core.schemas import (
    DefectCategory,
    ComponentCriticality,
    TriagePriority,
    DefectPrediction,
    TriageResult
)
from core.preprocessor import ImagePreprocessor
from core.classifier import MockDefectClassifier
from core.rules_engine import TriageRuleEngine
import app


class TestImagePreprocessor(unittest.TestCase):
    """
    Pruebas unitarias para el componente de normalización y transformación de imágenes.
    """

    def setUp(self):
        """Inicializa una instancia limpia del preprocesador antes de cada prueba."""
        self.preprocessor = ImagePreprocessor(target_size=(200, 200))

    def test_null_input_raises_value_error(self):
        """Verifica que suministrar None levante una excepción controlada."""
        with self.assertRaises(ValueError):
            self.preprocessor.validate_and_load(None)

    def test_numpy_array_conversion(self):
        """Verifica que una matriz NumPy RGB se convierta correctamente a PIL.Image."""
        dummy_array = np.zeros((100, 100, 3), dtype=np.uint8)
        img = self.preprocessor.validate_and_load(dummy_array)
        self.assertIsInstance(img, Image.Image)

    def test_gradio_dict_payload_conversion(self):
        """Verifica que diccionarios estructurados de Gradio (editor de capas) sean procesados."""
        dummy_pil = Image.new("RGB", (50, 50), color="gray")
        dict_payload = {"composite": dummy_pil, "background": None}
        loaded = self.preprocessor.validate_and_load(dict_payload)
        self.assertEqual(loaded.size, (50, 50))

    def test_canonical_dimensions_and_normalization(self):
        """Verifica que la salida preprocesada cumpla con el estándar NEU (200x200, escala de grises [0, 1])."""
        raw_image = Image.new("RGB", (450, 300), color="blue")
        result = self.preprocessor.preprocess_for_inference(raw_image)

        self.assertEqual(result["dimensions"], (200, 200))
        self.assertEqual(result["processed_pil"].size, (200, 200))
        self.assertEqual(result["processed_pil"].mode, "L")

        tensor = result["normalized_tensor"]
        self.assertEqual(tensor.shape, (200, 200))
        self.assertTrue(0.0 <= np.min(tensor) <= np.max(tensor) <= 1.0)


class TestMockDefectClassifier(unittest.TestCase):
    """
    Pruebas unitarias para el clasificador simulado calibrado de Fase 1.
    """

    def setUp(self):
        """Inicializa el clasificador y una imagen de prueba."""
        self.classifier = MockDefectClassifier()
        self.test_img = Image.new("L", (200, 200), color=128)

    def test_prediction_output_structure(self):
        """Verifica que la salida sea una instancia válida de DefectPrediction."""
        pred = self.classifier.predict(self.test_img)
        self.assertIsInstance(pred, DefectPrediction)
        self.assertIsInstance(pred.defect_class, DefectCategory)
        self.assertTrue(0.0 <= pred.confidence <= 100.0)
        self.assertTrue(pred.is_simulated)

    def test_reproducibility_by_pixel_hash(self):
        """Verifica que la misma imagen produzca exactamente la misma predicción consistente."""
        pred_1 = self.classifier.predict(self.test_img)
        pred_2 = self.classifier.predict(self.test_img)
        self.assertEqual(pred_1.defect_class, pred_2.defect_class)
        self.assertEqual(pred_1.confidence, pred_2.confidence)

    def test_probability_distribution_sum(self):
        """Verifica que la suma de las probabilidades simuladas sea aproximadamente 100%."""
        pred = self.classifier.predict(self.test_img)
        total_prob = sum(pred.raw_probabilities.values())
        self.assertAlmostEqual(total_prob, 100.0, places=1)


class TestTriageRuleEngine(unittest.TestCase):
    """
    Pruebas unitarias para el motor determinista de reglas heurísticas de triaje.
    """

    def test_conforming_surface_high_confidence(self):
        """Superficie conforme con alta confianza debe prescribir Prioridad Baja y liberación."""
        pred = DefectPrediction(defect_class=DefectCategory.SUPERFICIE_CONFORME, confidence=95.0)
        triage = TriageRuleEngine.evaluate(pred, ComponentCriticality.CRITICA)

        self.assertEqual(triage.priority, TriagePriority.BAJA)
        self.assertIn("Liberar lote", triage.recommendation)
        self.assertFalse(triage.requires_human_authorization)

    def test_crack_on_critical_component(self):
        """Grieta en componente crítico debe activar Prioridad Urgente, cuarentena y firma obligatoria."""
        pred = DefectPrediction(defect_class=DefectCategory.GRIETA, confidence=94.0)
        triage = TriageRuleEngine.evaluate(pred, ComponentCriticality.CRITICA)

        self.assertEqual(triage.priority, TriagePriority.URGENTE)
        self.assertTrue(triage.requires_human_authorization)
        self.assertIn("cuarentena", triage.recommendation.lower())

    def test_crack_on_medium_criticality(self):
        """Grieta en componente de criticidad media debe prescribir Prioridad Alta."""
        pred = DefectPrediction(defect_class=DefectCategory.GRIETA, confidence=92.0)
        triage = TriageRuleEngine.evaluate(pred, ComponentCriticality.MEDIA)

        self.assertEqual(triage.priority, TriagePriority.ALTA)
        self.assertTrue(triage.requires_human_authorization)

    def test_scratch_on_low_criticality(self):
        """Rayadura en componente no crítico debe asignarse como Prioridad Baja."""
        pred = DefectPrediction(defect_class=DefectCategory.RAYADURA, confidence=91.0)
        triage = TriageRuleEngine.evaluate(pred, ComponentCriticality.BAJA)

        self.assertEqual(triage.priority, TriagePriority.BAJA)
        self.assertIn("Liberar pieza", triage.recommendation)

    def test_low_confidence_escalation(self):
        """Confianza menor al umbral de incertidumbre (80%) debe escalar supervisión humana."""
        pred = DefectPrediction(defect_class=DefectCategory.SUPERFICIE_CONFORME, confidence=72.0)
        triage = TriageRuleEngine.evaluate(pred, ComponentCriticality.BAJA)

        self.assertEqual(triage.priority, TriagePriority.MEDIA)
        self.assertTrue(triage.requires_human_authorization)


class TestAppCallbacks(unittest.TestCase):
    """
    Pruebas de integración sobre los controladores de eventos de Gradio.
    """

    def test_on_analyze_surface_without_image(self):
        """Validar que pulsar Analizar sin imagen retorne mensaje de error amigable."""
        out = app.on_analyze_surface(None, "LOT-1", "TST-1", "Media")
        self.assertEqual(out[0], "Error: Sin imagen")
        self.assertEqual(out[1], "0.00%")
        self.assertEqual(out[2], "N/A")

    def test_on_reset_form_clears_all(self):
        """Validar que el reinicio de formulario restablezca los 10 componentes."""
        out = app.on_reset_form()
        self.assertEqual(len(out), 10)
        self.assertIsNone(out[0])  # image_input debe ser None


if __name__ == "__main__":
    unittest.main()
