# ==============================================================================
# PROYECTO 10: DETECTOR INTELIGENTE DE DEFECTOS SUPERFICIALES EN PIEZAS INDUSTRIALES
# INSTITUCIÓN: UNIVERSIDAD TECNOLÓGICA DE QUERÉTARO (UTEQ)
# MODULO: core/rules_engine.py
# ==============================================================================
# RESUMEN DEL ARCHIVO:
# Este módulo implementa el motor determinista de reglas de triaje industrial
# (Sistemas Basados en Conocimiento / Heurísticas de Negocio) conforme al
# enfoque híbrido estipulado en la Sección 7.2 del reporte técnico UTEQ.
# Recibe la clase estimada por el modelo perceptual y su confianza, cruzándolas
# con la criticidad del componente para calcular determinísticamente la
# prioridad de inspección y la prescripción operativa en piso de planta.
# ==============================================================================

from typing import Tuple, Dict, Any, Optional
from core.schemas import (
    DefectCategory,
    ComponentCriticality,
    TriagePriority,
    DefectPrediction,
    TriageResult
)


class TriageRuleEngine:
    """
    Motor de inferencia basado en reglas condicionales (IF-THEN) para la toma
    de decisiones operativas en control de calidad metalmecánica.
    """

    # Umbral mínimo de confianza estadística para decisiones directas
    CONFIDENCE_UNCERTAINTY_THRESHOLD: float = 80.0

    @classmethod
    def evaluate(
        cls,
        prediction: DefectPrediction,
        criticality: ComponentCriticality,
        lot_id: str = "N/A",
        test_id: str = "N/A"
    ) -> TriageResult:
        """
        Ejecuta la evaluación cruzada entre la predicción visual y el contexto de la pieza.

        :param prediction: Resultado de inferencia emitido por el clasificador.
        :type prediction: DefectPrediction
        :param criticality: Nivel de criticidad funcional del componente (Baja, Media, Crítica).
        :type criticality: ComponentCriticality
        :param lot_id: Código identificador del lote de fabricación.
        :type lot_id: str
        :param test_id: Identificador de prueba o caso de inspección.
        :type test_id: str
        :return: Dictamen prescriptivo con prioridad y recomendación de planta.
        :rtype: TriageResult
        """
        defect = prediction.defect_class
        confidence = prediction.confidence
        requires_human_auth = False

        # Regla 1: Superficie Conforme (Sin Defectos Detectados)
        if defect == DefectCategory.SUPERFICIE_CONFORME:
            if confidence >= cls.CONFIDENCE_UNCERTAINTY_THRESHOLD:
                priority = TriagePriority.BAJA
                recommendation = (
                    "Liberar lote para la siguiente estación de maquinado o ensamble. "
                    "Superficie homogénea dentro de tolerancias de rugosidad."
                )
            else:
                # Si la confianza es dudosa aun cuando aparenta estar conforme
                priority = TriagePriority.MEDIA
                requires_human_auth = True
                recommendation = (
                    "Superficie aparentemente conforme pero con baja confianza de sensor (< 80%). "
                    "Se solicita confirmación visual rápida por inspector en línea."
                )

        # Regla 2: Defectos Estructuralmente Críticos (Grietas y Defectos de Laminado)
        elif defect in (DefectCategory.GRIETA, DefectCategory.DEFECTO_LAMINADO):
            if criticality == ComponentCriticality.CRITICA:
                priority = TriagePriority.URGENTE
                requires_human_auth = True
                recommendation = (
                    "ALERTA CRÍTICA: Discontinuidad estructural profunda en componente de alta responsabilidad. "
                    "Detener procesamiento inmediatamente. Poner lote en cuarentena y remitir a laboratorio de NDT "
                    "(Ensayos No Destructivos por líquidos penetrantes o ultrasonido)."
                )
            elif criticality == ComponentCriticality.MEDIA:
                priority = TriagePriority.ALTA
                requires_human_auth = True
                recommendation = (
                    "Prioridad Alta: Defecto tipo " + defect.value + " en componente de criticidad media. "
                    "Segregar pieza del lote, detener flujo continuo y notificar al supervisor de calidad."
                )
            else:  # Criticidad Baja
                priority = TriagePriority.MEDIA
                recommendation = (
                    "Falla detectada en pieza no crítica. Separar pieza para inspección visual "
                    "y evaluar posible tolerancia antes de rechazo definitivo."
                )

        # Regla 3: Picaduras Superficiales / Porosidades
        elif defect == DefectCategory.PICADURA:
            if criticality == ComponentCriticality.CRITICA:
                priority = TriagePriority.ALTA
                requires_human_auth = True
                recommendation = (
                    "Picaduras detectadas en componente crítico. Posible riesgo de fatiga superficial. "
                    "Separar espécimen para medición de profundidad con rugosímetro/perfilómetro óptico."
                )
            elif criticality == ComponentCriticality.MEDIA:
                priority = TriagePriority.MEDIA
                recommendation = (
                    "Picaduras localizadas. Derivar a inspección intermedia para validar "
                    "si el defecto desaparece en la fase posterior de rectificado o pulido."
                )
            else:
                priority = TriagePriority.BAJA
                recommendation = (
                    "Picaduras menores tolerables en pieza de baja criticidad. "
                    "Aprobar con observación en bitácora de lote."
                )

        # Regla 4: Rayaduras Superficiales
        elif defect == DefectCategory.RAYADURA:
            if criticality == ComponentCriticality.CRITICA:
                priority = TriagePriority.ALTA
                recommendation = (
                    "Rayadura en componente de alta exigencia. Enviar a estación de retrabajo / pulido de precisión "
                    "para verificar que no comprometa el sello hermético o la tolerancia geométrica."
                )
            elif criticality == ComponentCriticality.MEDIA:
                priority = TriagePriority.MEDIA
                recommendation = (
                    "Rayadura superficial moderada. Derivar a estación de pulido estándar. "
                    "No requiere cuarentena de lote."
                )
            else:
                priority = TriagePriority.BAJA
                recommendation = (
                    "Rayadura estética no funcional. Liberar pieza para continuar con el flujo normal de producción."
                )

        # Fallback de seguridad operativa
        else:
            priority = TriagePriority.MEDIA
            requires_human_auth = True
            recommendation = (
                "Anomalía no catalogada. Requerir evaluación manual exhaustiva por el jefe de turno de calidad."
            )

        # Modificador Heurístico por Umbral de Incertidumbre
        # Si la confianza del modelo probabilístico es menor al 80%, se escala la supervisión
        if confidence < cls.CONFIDENCE_UNCERTAINTY_THRESHOLD and priority == TriagePriority.BAJA:
            priority = TriagePriority.MEDIA
            requires_human_auth = True
            recommendation += " [NOTA TÉCNICA: Se eleva a prioridad Media por baja certidumbre en captura de imagen]."

        return TriageResult(
            defect_category=defect,
            confidence=confidence,
            criticality=criticality,
            priority=priority,
            recommendation=recommendation,
            requires_human_authorization=requires_human_auth,
            lot_id=lot_id or "SIN-LOTE",
            test_id=test_id or "N/A"
        )

