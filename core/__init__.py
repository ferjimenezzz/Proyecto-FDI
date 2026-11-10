# ==============================================================================
# PROYECTO 10: DETECTOR INTELIGENTE DE DEFECTOS SUPERFICIALES EN PIEZAS INDUSTRIALES
# INSTITUCIÓN: UNIVERSIDAD TECNOLÓGICA DE QUERÉTARO (UTEQ)
# MODULO: core/__init__.py
# ==============================================================================
# RESUMEN DEL ARCHIVO:
# Inicializador del paquete core del sistema de inspección visual de calidad.
# Exporta de manera unificada los contratos y componentes principales:
# esquemas de datos, preprocesador de imagen, clasificadores y motor de reglas.
# ==============================================================================

from core.schemas import (
    DefectCategory,
    ComponentCriticality,
    TriagePriority,
    DefectPrediction,
    TriageResult
)
from core.preprocessor import ImagePreprocessor
from core.classifier import (
    BaseDefectClassifier,
    MockDefectClassifier,
    Phase2DeepLearningClassifier
)
from core.rules_engine import TriageRuleEngine

__all__ = [
    "DefectCategory",
    "ComponentCriticality",
    "TriagePriority",
    "DefectPrediction",
    "TriageResult",
    "ImagePreprocessor",
    "BaseDefectClassifier",
    "MockDefectClassifier",
    "Phase2DeepLearningClassifier",
    "TriageRuleEngine"
]

