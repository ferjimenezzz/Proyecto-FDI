# ==============================================================================
# PROYECTO 10: DETECTOR INTELIGENTE DE DEFECTOS SUPERFICIALES EN PIEZAS INDUSTRIALES
# INSTITUCIÓN: UNIVERSIDAD TECNOLÓGICA DE QUERÉTARO (UTEQ)
# MODULO: core/schemas.py
# ==============================================================================
# RESUMEN DEL ARCHIVO:
# Este módulo define las estructuras de datos fundamentales, enumeraciones tipadas
# y modelos de transferencia de datos (DTO) utilizados en todo el sistema.
# Centraliza la definición de categorías de defectos, niveles de criticidad de componentes,
# prioridades de triaje de inspección y los contratos de resultados del clasificador
# y del motor de reglas determinista.
# ==============================================================================

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from datetime import datetime


class DefectCategory(str, Enum):
    """
    Enumeración formal de las categorías de defectos superficiales analizadas
    en superficies metálicas industriales, basadas en la taxonomía de la base de
    datos NEU Surface Defect Database (Song & Yan) y la consigna del Proyecto 10.
    """
    GRIETA = "Grieta"
    PICADURA = "Picadura"
    DEFECTO_LAMINADO = "Defecto de laminado"
    RAYADURA = "Rayadura"
    SUPERFICIE_CONFORME = "Superficie Conforme (Sin Defecto)"


class ComponentCriticality(str, Enum):
    """
    Nivel de criticidad funcional del componente o pieza dentro del ensamblaje industrial.
    Determina la tolerancia a fallos y pondera la prioridad final en el motor de reglas.
    """
    BAJA = "Baja"
    MEDIA = "Media"
    CRITICA = "Crítica"


class TriagePriority(str, Enum):
    """
    Niveles de prioridad de inspección y triaje operativo en piso de planta.
    """
    BAJA = "Baja"
    MEDIA = "Media"
    ALTA = "Alta"
    URGENTE = "Urgente"


@dataclass
class DefectPrediction:
    """
    Estructura de datos que encapsula el resultado de la inferencia perceptual
    (simulada en Fase 1, red neuronal profunda en Fase 2).

    Attributes:
        defect_class (DefectCategory): Clase de defecto identificada con mayor probabilidad.
        confidence (float): Nivel de confianza porcentual de la predicción (0.0 a 100.0).
        raw_probabilities (Dict[str, float]): Distribución de probabilidades sobre todas las clases.
        timestamp (str): Marca de tiempo ISO del momento en que se ejecutó la inferencia.
        is_simulated (bool): Bandera booleana que indica si el resultado proviene de simulación.
    """
    defect_class: DefectCategory
    confidence: float
    raw_probabilities: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    is_simulated: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializa la instancia a un diccionario nativo para exportación o telemetría.
        
        :return: Diccionario con los campos estructurados de la predicción.
        :rtype: Dict[str, Any]
        """
        return {
            "defect_class": self.defect_class.value,
            "confidence": f"{self.confidence:.2f}%",
            "raw_probabilities": {k: f"{v:.2f}%" for k, v in self.raw_probabilities.items()},
            "timestamp": self.timestamp,
            "is_simulated": self.is_simulated
        }


@dataclass
class TriageResult:
    """
    Estructura de datos que encapsula el dictamen prescriptivo emitido por el
    motor de reglas deterministas (Sistemas Basados en Conocimiento).

    Attributes:
        defect_category (DefectCategory): Categoría del defecto evaluado.
        confidence (float): Confianza porcentual del modelo perceptual.
        criticality (ComponentCriticality): Nivel de criticidad contextual de la pieza.
        priority (TriagePriority): Prioridad asignada por la matriz de reglas IF-THEN.
        recommendation (str): Instrucción operativa prescriptiva para el operador de calidad.
        requires_human_authorization (bool): Indica si se requiere validación obligatoria de supervisor.
        lot_id: Código o identificador de lote industrial analizado.
        test_id: Identificador de caso o espécimen de prueba opcional.
    """
    defect_category: DefectCategory
    confidence: float
    criticality: ComponentCriticality
    priority: TriagePriority
    recommendation: str
    requires_human_authorization: bool
    lot_id: str = "N/A"
    test_id: str = "N/A"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializa el resultado de triaje a un diccionario para registro de auditoría.
        
        :return: Diccionario con la auditoría completa de la decisión.
        :rtype: Dict[str, Any]
        """
        return {
            "lot_id": self.lot_id,
            "test_id": self.test_id,
            "defect_category": self.defect_category.value,
            "confidence": f"{self.confidence:.2f}%",
            "criticality": self.criticality.value,
            "priority": self.priority.value,
            "recommendation": self.recommendation,
            "requires_human_authorization": self.requires_human_authorization,
            "timestamp": self.timestamp
        }

