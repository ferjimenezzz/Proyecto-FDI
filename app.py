# ==============================================================================
# PROYECTO 10: DETECTOR INTELIGENTE DE DEFECTOS SUPERFICIALES EN PIEZAS INDUSTRIALES
# INSTITUCIÓN: UNIVERSIDAD TECNOLÓGICA DE QUERÉTARO (UTEQ)
# INGENIERÍA EN TECNOLOGÍAS DE LA INFORMACIÓN E INNOVACIÓN DIGITAL
# ASIGNATURA: FUNDAMENTOS DE INTELIGENCIA ARTIFICIAL
# MODULO: app.py
# ==============================================================================
# RESUMEN DEL ARCHIVO:
# Este archivo constituye la aplicación principal y punto de entrada de la
# interfaz gráfica de usuario desarrollada con Python y Gradio Blocks.
# Implementa la experiencia interactiva completa definida en el Reporte Técnico:
# - Carga y previsualización de fotografías metalmecánicas (JPG/PNG).
# - Registro de metadatos de trazabilidad (Lote, ID de prueba y Criticidad).
# - Disparo de análisis perceptual desacoplado (Mock Fase 1 / CNN Fase 2).
# - Evaluación determinista mediante Motor de Reglas Heurísticas de Calidad.
# - Despliegue de priorización de triaje y prescripción operativa de piso.
# - Sistema de diseño visual Material Design Industrial Dark Theme.
# ==============================================================================

import os
# Desactivación estricta de telemetría externa para arranque offline inmediato y seguro
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import json
from typing import Tuple, Dict, Any, Optional
import gradio as gr
from PIL import Image

# Importación de componentes del núcleo modular
from core.schemas import (
    DefectCategory,
    ComponentCriticality,
    TriagePriority,
    DefectPrediction,
    TriageResult
)
from core.classifier import BaseDefectClassifier, MockDefectClassifier
from core.rules_engine import TriageRuleEngine
from core.preprocessor import ImagePreprocessor
from core.sample_generator import generate_samples

# Asegurar la existencia de imágenes de muestra para evaluación inmediata
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")
if not os.path.exists(SAMPLES_DIR) or len(os.listdir(SAMPLES_DIR)) == 0:
    generate_samples(SAMPLES_DIR)

# Instanciación de los componentes de negocio (Inversión de dependencias)
# Para la Fase 1 se inyecta MockDefectClassifier; para la Fase 2 bastará inyectar Phase2DeepLearningClassifier
ACTIVE_CLASSIFIER: BaseDefectClassifier = MockDefectClassifier()
IMAGE_PREPROCESSOR: ImagePreprocessor = ImagePreprocessor()

# Variable de estado en memoria para almacenar la última predicción realizada
# Esto permite que el botón "Calcular prioridad" opere sobre el resultado de "Analizar"
last_prediction_state: Optional[DefectPrediction] = None


# ------------------------------------------------------------------------------
# CONTROLADORES DE EVENTOS DE INTERFAZ (CALLBACKS)
# ------------------------------------------------------------------------------

def on_analyze_surface(
    image_input: Optional[Image.Image],
    lot_code: str,
    test_id: str,
    criticality_val: str
) -> Tuple[str, str, str, str, str, str]:
    """
    Controlador para el botón "Analizar superficie".
    Valida la presencia de la imagen, ejecuta la inferencia perceptual y
    evalúa automáticamente la prioridad inicial mediante el motor de reglas.

    :param image_input: Fotografía capturada o cargada por el operador.
    :type image_input: Optional[Image.Image]
    :param lot_code: Código de lote ingresado en el campo de texto.
    :type lot_code: str
    :param test_id: Identificador opcional de la prueba o espécimen.
    :type test_id: str
    :param criticality_val: Valor del radio button de criticidad ("Baja", "Media", "Crítica").
    :type criticality_val: str
    :return: Tupla con (Defecto, Confianza, Prioridad Texto, Prioridad HTML, Recomendación, Auditoría JSON).
    :rtype: Tuple[str, str, str, str, str, str]
    """
    global last_prediction_state

    # 1. Validación de entrada: verificar que el operador haya cargado una imagen
    if image_input is None:
        empty_badge = (
            "<div class='status-badge badge-neutral'>"
            "<span>⚠️ SIN CAPTURA</span>"
            "</div>"
        )
        return (
            "Error: Sin imagen",
            "0.00%",
            "N/A",
            empty_badge,
            "Por favor cargue una fotografía de la superficie metálica antes de analizar.",
            "{}"
        )

    try:
        # 2. Ejecución de la inferencia en el clasificador activo
        prediction = ACTIVE_CLASSIFIER.predict(image_input)
        last_prediction_state = prediction

        # 3. Mapeo seguro del valor de criticidad al enum fuertemente tipado
        try:
            criticality_enum = ComponentCriticality(criticality_val)
        except ValueError:
            criticality_enum = ComponentCriticality.MEDIA

        # 4. Evaluación en el motor de reglas determinista de triaje
        triage = TriageRuleEngine.evaluate(
            prediction=prediction,
            criticality=criticality_enum,
            lot_id=lot_code.strip() or "LOT-GENERAL",
            test_id=test_id.strip() or "TST-SN"
        )

        # 5. Generación de insignia HTML semántica para la prioridad
        priority_badge = _build_priority_badge(triage.priority)

        # 6. Serialización de telemetría y auditoría técnica
        audit_payload = {
            "auditoria_calidad": triage.to_dict(),
            "distribucion_probabilidades": prediction.to_dict()["raw_probabilities"],
            "modo_operativo": "Simulación Calibrada (Fase 1 UTEQ - Preparada para Fase 2 CNN)"
        }
        audit_json = json.dumps(audit_payload, indent=2, ensure_ascii=False)

        return (
            prediction.defect_class.value,
            f"{prediction.confidence:.2f}%",
            triage.priority.value,
            priority_badge,
            triage.recommendation,
            audit_json
        )

    except Exception as exc:
        err_badge = (
            "<div class='status-badge badge-urgent'>"
            "<span>ERROR DE PROCESAMIENTO</span>"
            "</div>"
        )
        return (
            "Excepción de inferencia",
            "0.00%",
            "Error",
            err_badge,
            f"Error interno al procesar imagen: {str(exc)}",
            "{}"
        )


def on_calculate_priority(
    current_defect_str: str,
    confidence_str: str,
    criticality_val: str,
    lot_code: str,
    test_id: str
) -> Tuple[str, str, str, str]:
    """
    Controlador específico para el botón independiente "Calcular prioridad" (Sección 7.4).
    Permite al operador recalcular la decisión determinista si modifica la criticidad
    sin necesidad de re-clasificar la fotografía.

    :param current_defect_str: Texto actual del defecto estimado.
    :type current_defect_str: str
    :param confidence_str: Texto del porcentaje de confianza actual.
    :type confidence_str: str
    :param criticality_val: Nivel de criticidad seleccionado.
    :type criticality_val: str
    :param lot_code: Identificador de lote.
    :type lot_code: str
    :param test_id: Identificador de prueba.
    :type test_id: str
    :return: Tupla con (Prioridad Texto, Prioridad HTML, Recomendación Operativa, Auditoría JSON).
    :rtype: Tuple[str, str, str, str]
    """
    global last_prediction_state

    # Si aún no se ha ejecutado un análisis o el campo está vacío
    if not current_defect_str or current_defect_str in ("Sin análisis", "Error: Sin imagen", ""):
        neutral_badge = (
            "<div class='status-badge badge-neutral'>"
            "<span>SIN DATOS</span>"
            "</div>"
        )
        return (
            "N/A",
            neutral_badge,
            "Debe ejecutar primero el análisis de superficie antes de calcular la prioridad.",
            "{}"
        )

    # Conversión del texto de defecto al Enum
    try:
        defect_enum = DefectCategory(current_defect_str)
    except ValueError:
        defect_enum = DefectCategory.SUPERFICIE_CONFORME

    # Extracción del float de confianza
    try:
        cleaned_conf = float(confidence_str.replace("%", "").strip())
    except ValueError:
        cleaned_conf = 90.0

    # Conversión de la criticidad
    try:
        criticality_enum = ComponentCriticality(criticality_val)
    except ValueError:
        criticality_enum = ComponentCriticality.MEDIA

    # Construcción o recuperación de la predicción
    if last_prediction_state is not None:
        prediction_to_use = last_prediction_state
    else:
        prediction_to_use = DefectPrediction(
            defect_class=defect_enum,
            confidence=cleaned_conf
        )

    # Recálculo determinista con la nueva combinación de criticidad
    triage = TriageRuleEngine.evaluate(
        prediction=prediction_to_use,
        criticality=criticality_enum,
        lot_id=lot_code.strip() or "LOT-GENERAL",
        test_id=test_id.strip() or "TST-SN"
    )

    priority_badge = _build_priority_badge(triage.priority)
    audit_payload = {
        "recálculo_prioridad": triage.to_dict(),
        "accion": "Evaluación bajo políticas de control de calidad"
    }

    return (
        triage.priority.value,
        priority_badge,
        triage.recommendation,
        json.dumps(audit_payload, indent=2, ensure_ascii=False)
    )


def on_reset_form() -> Tuple[None, str, str, str, str, str, str, str, str, str]:
    """
    Controlador para el botón "Nueva pieza" (Reset completo).
    Restaura todos los controles a sus valores iniciales limpios.

    :return: Tupla con los 10 componentes reseteados a su estado por defecto.
    :rtype: Tuple[None, str, str, str, str, str, str, str, str, str]
    """
    global last_prediction_state
    last_prediction_state = None

    default_badge = (
        "<div class='status-badge badge-neutral'>"
        "<span>PENDIENTE DE ANÁLISIS</span>"
        "</div>"
    )

    return (
        None,                   # image_input
        "LOT-2026-X4B",         # lot_input
        "TST-00101",            # test_id_input
        "Media",                # criticality_input
        "",                     # defect_output
        "",                     # confidence_output
        "",                     # priority_output
        default_badge,          # priority_html
        "",                     # recommendation_output
        "{}"                    # audit_output
    )


def _build_priority_badge(priority: TriagePriority) -> str:
    """
    Genera el fragmento HTML con diseño Material Design para el badge semántico
    de prioridad de triaje.

    :param priority: Nivel de prioridad asignado.
    :type priority: TriagePriority
    :return: Cadena HTML con estilo embebido.
    :rtype: str
    """
    css_class_map = {
        TriagePriority.BAJA: "badge-low",
        TriagePriority.MEDIA: "badge-medium",
        TriagePriority.ALTA: "badge-high",
        TriagePriority.URGENTE: "badge-urgent"
    }
    icon_map = {
        TriagePriority.BAJA: "🟢",
        TriagePriority.MEDIA: "🟡",
        TriagePriority.ALTA: "🟠",
        TriagePriority.URGENTE: "🔴"
    }
    css_class = css_class_map.get(priority, "badge-neutral")
    icon = icon_map.get(priority, "⚪")

    return f"""
    <div class="priority-badge-container">
        <div class="status-badge {css_class}">
            <span class="badge-icon">{icon}</span>
            <span class="badge-text">PRIORIDAD: {priority.value.upper()}</span>
        </div>
    </div>
    """


# ------------------------------------------------------------------------------
# HOJA DE ESTILOS CSS - MATERIAL DESIGN INDUSTRIAL DARK THEME
# ------------------------------------------------------------------------------

CUSTOM_CSS = """
/* Reset y variables de color - Material Design Dark Industrial */
:root {
    --md-sys-color-background: #0F1115;
    --md-sys-color-surface: #171A21;
    --md-sys-color-surface-container: #1E222B;
    --md-sys-color-surface-high: #262B36;
    --md-sys-color-primary: #3B82F6;
    --md-sys-color-primary-hover: #2563EB;
    --md-sys-color-border: #2D333F;
    --md-sys-color-text-primary: #F1F5F9;
    --md-sys-color-text-secondary: #94A3B8;
    --md-sys-color-success: #10B981;
    --md-sys-color-warning: #F59E0B;
    --md-sys-color-danger: #EF4444;
}

body, .gradio-container {
    background-color: var(--md-sys-color-background) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: var(--md-sys-color-text-primary) !important;
}

/* Encabezado Institucional */
.header-container {
    background: linear-gradient(135deg, #1E222B 0%, #15181F 100%);
    border: 1px solid var(--md-sys-color-border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}

.header-title {
    font-size: 1.55rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0;
    letter-spacing: -0.02em;
}

.header-subtitle {
    font-size: 0.90rem;
    color: var(--md-sys-color-text-secondary);
    margin-top: 6px;
    margin-bottom: 0;
}

.header-tag {
    display: inline-block;
    background-color: rgba(59, 130, 246, 0.15);
    color: #60A5FA;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 9999px;
    border: 1px solid rgba(59, 130, 246, 0.3);
    margin-top: 10px;
}

/* Tarjetas y Contenedores */
.panel-card {
    background-color: var(--md-sys-color-surface) !important;
    border: 1px solid var(--md-sys-color-border) !important;
    border-radius: 12px !important;
    padding: 18px !important;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.25) !important;
}

.panel-section-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #E2E8F0;
    border-bottom: 1px solid var(--md-sys-color-border);
    padding-bottom: 8px;
    margin-bottom: 14px;
}

/* Estilo para los botones de acción */
.btn-analyze {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
.btn-analyze:hover {
    background-color: #1D4ED8 !important;
    box-shadow: 0 0 14px rgba(37, 99, 235, 0.4) !important;
}

.btn-priority {
    background-color: #475569 !important;
    color: #F8FAFC !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: 1px solid #64748B !important;
}
.btn-priority:hover {
    background-color: #334155 !important;
}

.btn-reset {
    background-color: transparent !important;
    color: #94A3B8 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}
.btn-reset:hover {
    background-color: rgba(148, 163, 184, 0.1) !important;
    color: #F1F5F9 !important;
}

/* Insignias Semánticas de Prioridad de Triaje */
.priority-badge-container {
    margin: 8px 0 16px 0;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.03em;
    width: 100%;
    justify-content: center;
}

.badge-low {
    background-color: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.4);
}

.badge-medium {
    background-color: rgba(245, 158, 11, 0.15);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.4);
}

.badge-high {
    background-color: rgba(249, 115, 22, 0.15);
    color: #FB923C;
    border: 1px solid rgba(249, 115, 22, 0.4);
}

.badge-urgent {
    background-color: rgba(239, 68, 68, 0.2);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.5);
    animation: pulse 2s infinite;
}

.badge-neutral {
    background-color: rgba(100, 116, 139, 0.15);
    color: #94A3B8;
    border: 1px solid rgba(100, 116, 139, 0.3);
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.75; }
    100% { opacity: 1; }
}

/* Footer de acreditación */
.footer-text {
    font-size: 0.80rem;
    color: var(--md-sys-color-text-secondary);
    text-align: center;
    margin-top: 30px;
    border-top: 1px solid var(--md-sys-color-border);
    padding-top: 15px;
}
"""


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# CONSTRUCCIÓN DE LA APLICACIÓN GRADIO
# ------------------------------------------------------------------------------

def get_theme() -> gr.Theme:
    """
    Construye la configuración del tema visual Material Design Industrial Dark.

    :return: Instancia configurada de gr.themes.Soft con paleta de alto contraste.
    :rtype: gr.Theme
    """
    return gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]
    ).set(
        body_background_fill="#0F1115",
        block_background_fill="#171A21",
        block_border_color="#2D333F",
        input_background_fill="#1E222B",
        input_border_color="#333944"
    )


def build_app() -> gr.Blocks:
    """
    Construye y ensambla la interfaz gráfica interactiva con Gradio Blocks.
    
    :return: Instancia configurada de gr.Blocks lista para lanzamiento (launch).
    :rtype: gr.Blocks
    """
    with gr.Blocks(title="Detector de Defectos Superficiales | UTEQ") as demo:

        # 1. Encabezado Institucional del Proyecto
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="header-container">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <h1 class="header-title">Detector Inteligente de Defectos Superficiales en Piezas Industriales</h1>
                            <p class="header-subtitle">
                                Universidad Tecnológica de Querétaro &bull; Ingeniería en Tecnologías de la Información &bull; Fundamentos de IA
                            </p>
                        </div>
                        <span class="header-tag">ETAPA 1: PROTOTIPO & TRIAGE</span>
                    </div>
                </div>
                """)

        # 2. Distribución Principal en Dos Columnas (Entrada vs Resultados)
        with gr.Row(equal_height=False):

            # COLUMNA IZQUIERDA: Entrada de Operador y Contexto de la Pieza
            with gr.Column(scale=5, elem_classes=["panel-card"]):
                gr.HTML("<div class='panel-section-title'>📥 1. Entrada del Operador / Contexto de la Pieza</div>")

                # Carga de la fotografía (JPG/PNG)
                image_input = gr.Image(
                    label="Fotografía de la Pieza",
                    type="pil",
                    sources=["upload", "clipboard"],
                    elem_id="image_preview_box"
                )

                # Identificadores de pieza
                with gr.Row():
                    lot_input = gr.Textbox(
                        label="Código / Lote",
                        value="LOT-2026-X4B",
                        placeholder="Ej. LOT-2026-X4B",
                        scale=3
                    )
                    test_id_input = gr.Textbox(
                        label="ID Prueba (Opcional)",
                        value="TST-00101",
                        placeholder="Ej. TST-00101",
                        scale=2
                    )

                # Nivel de Criticidad del Componente
                criticality_input = gr.Radio(
                    choices=["Baja", "Media", "Crítica"],
                    value="Media",
                    label="Criticidad del Componente",
                    info="Afecta directamente la prioridad y las políticas de segregación en piso de planta."
                )

                # Botones de Acción (Sección 7.4 del Reporte Técnico)
                with gr.Row():
                    btn_analyze = gr.Button(
                        "🔍 Analizar superficie",
                        variant="primary",
                        elem_classes=["btn-analyze"],
                        scale=3
                    )
                    btn_priority = gr.Button(
                        "⚖️ Calcular prioridad",
                        elem_classes=["btn-priority"],
                        scale=2
                    )
                    btn_reset = gr.Button(
                        "🔄 Nueva pieza",
                        elem_classes=["btn-reset"],
                        scale=2
                    )

                # Galería de Especímenes Sintéticos Precargados para prueba rápida
                sample_files = [
                    [os.path.join(SAMPLES_DIR, "muestra_grieta.png"), "LOT-2026-CRK", "TST-00001", "Crítica"],
                    [os.path.join(SAMPLES_DIR, "muestra_picadura.png"), "LOT-2026-PIT", "TST-00002", "Media"],
                    [os.path.join(SAMPLES_DIR, "muestra_rayadura.png"), "LOT-2026-SCR", "TST-00003", "Baja"],
                    [os.path.join(SAMPLES_DIR, "muestra_conforme.png"), "LOT-2026-OK0", "TST-00004", "Baja"]
                ]
                
                gr.Examples(
                    examples=sample_files,
                    inputs=[image_input, lot_input, test_id_input, criticality_input],
                    label="Especímenes Industriales de Demostración (Haga clic para cargar)",
                    cache_examples=False
                )

            # COLUMNA DERECHA: Resultados Visibles y Prescripción Operativa
            with gr.Column(scale=5, elem_classes=["panel-card"]):
                gr.HTML("<div class='panel-section-title'>📊 2. Resultados de Diagnóstico y Triaje</div>")

                # Diagnóstico perceptual
                with gr.Row():
                    defect_output = gr.Textbox(
                        label="Defecto Estimado",
                        placeholder="Pendiente de análisis...",
                        interactive=False,
                        scale=3
                    )
                    confidence_output = gr.Textbox(
                        label="Confianza (Simulada)",
                        placeholder="--%",
                        interactive=False,
                        scale=2
                    )

                # Campo de Prioridad de Inspección (Fiel a Figura 3 y 4 del Reporte)
                priority_output = gr.Textbox(
                    label="Prioridad de Inspección",
                    placeholder="Pendiente de análisis...",
                    interactive=False
                )

                # Insignia Visual Semántica de Prioridad de Triaje
                priority_html = gr.HTML(
                    value="""
                    <div class="priority-badge-container">
                        <div class="status-badge badge-neutral">
                            <span>⚪ PENDIENTE DE ANÁLISIS</span>
                        </div>
                    </div>
                    """
                )

                # Recomendación Prescriptiva
                recommendation_output = gr.Textbox(
                    label="Recomendación Operativa",
                    placeholder="El motor de reglas deterministas emitirá la recomendación de piso de planta...",
                    lines=3,
                    interactive=False
                )

                # Sección Colapsable para Telemetría y Preparación Fase 2
                with gr.Accordion("🛠️ Auditoría de Decisión & Extensibilidad Fase 2", open=False):
                    gr.Markdown(
                        "**Nota de Arquitectura:** En la **Fase 2**, este módulo consumirá el tensor "
                        "normalizado $(1, 1, 200, 200)$ generado por `ImagePreprocessor` sobre una red convolucional "
                        "(CNN) entrenada con el dataset NEU. El motor de reglas permanecerá idéntico."
                    )
                    audit_output = gr.Code(
                        label="Trazabilidad y Auditoría (JSON)",
                        language="json",
                        value="{}"
                    )

        # 3. Pie de Página Institucional
        gr.HTML("""
        <div class="footer-text">
            Equipo del Proyecto: Kevin Cruz Hernández &bull; Montserrat Oliva Rodríguez &bull; Citlali Vite Merino &bull; Fernando Jiménez Ángeles<br>
            Docente: Ernesto Chavero Navarrete &bull; Octubre 2026 &bull; UTEQ | FUNDAMENTOS DE IA &bull; PROYECTO 10
        </div>
        """)

        # ----------------------------------------------------------------------
        # CONEXIÓN DE EVENTOS DE COMPONENTES
        # ----------------------------------------------------------------------

        # Evento 1: Botón "Analizar superficie"
        btn_analyze.click(
            fn=on_analyze_surface,
            inputs=[image_input, lot_input, test_id_input, criticality_input],
            outputs=[defect_output, confidence_output, priority_output, priority_html, recommendation_output, audit_output]
        )

        # Evento 2: Botón "Calcular prioridad" (Evaluación independiente sin re-analizar)
        btn_priority.click(
            fn=on_calculate_priority,
            inputs=[defect_output, confidence_output, criticality_input, lot_input, test_id_input],
            outputs=[priority_output, priority_html, recommendation_output, audit_output]
        )

        # Evento 3: Botón "Nueva pieza" (Reset integral)
        btn_reset.click(
            fn=on_reset_form,
            inputs=[],
            outputs=[
                image_input,
                lot_input,
                test_id_input,
                criticality_input,
                defect_output,
                confidence_output,
                priority_output,
                priority_html,
                recommendation_output,
                audit_output
            ]
        )

    return demo


# Punto de entrada para ejecución directa
if __name__ == "__main__":
    app = build_app()
    # Lanzamiento local aplicando el tema y estilos en launch() conforme a Gradio 6
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=get_theme(),
        css=CUSTOM_CSS,
        show_error=True
    )

