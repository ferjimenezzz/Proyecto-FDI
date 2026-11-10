# ==============================================================================
# PROYECTO 10: DETECTOR INTELIGENTE DE DEFECTOS SUPERFICIALES EN PIEZAS INDUSTRIALES
# INSTITUCIÓN: UNIVERSIDAD TECNOLÓGICA DE QUERÉTARO (UTEQ)
# INGENIERÍA EN TECNOLOGÍAS DE LA INFORMACIÓN E INNOVACIÓN DIGITAL
# ASIGNATURA: FUNDAMENTOS DE INTELIGENCIA ARTIFICIAL
# ==============================================================================

## 1. Resumen Ejecutivo y Alcance
Este repositorio contiene la implementación integral del prototipo interactivo desarrollado en **Python + Gradio** para el **Proyecto 10: Detector inteligente de defectos superficiales en piezas industriales**.

El sistema cumple rigurosamente con los requisitos de la **Primera Etapa (Planteamiento conceptual, recorrido de interfaz y triaje heurístico)**, incorporando una arquitectura limpia desacoplada (**SOLID / Strategy Pattern**) que deja el pipeline completamente preparado para conectar el modelo de **Deep Learning (CNN / YOLO / ONNX)** en la **Fase 2** sin requerir modificaciones en la interfaz ni en el motor de negocio.

---

## 2. Arquitectura del Sistema (Enfoque Híbrido)

Conforme a lo especificado en la Sección 7.2 del Reporte Técnico de la UTEQ:

```
[ Fotografía JPG/PNG ] + [ Lote, ID, Criticidad ]
               │
               ▼
┌────────────────────────────────────────────────────────┐
│ 1. PIPELINE DE PREPROCESAMIENTO (ImagePreprocessor)    │
│    - Conversión a Escala de Grises ('L')               │
│    - Normalización canónica a 200x200 px (Dataset NEU) │
│    - Tensor flotante [0.0, 1.0] listo para CNN         │
│    - Métricas estadísticas de contraste y luminancia   │
└──────────────────────────────────┬─────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────┐
│ 2. COMPONENTE PERCEPTUAL (BaseDefectClassifier)        │
│    ├─ MockDefectClassifier (Activo en Fase 1)          │
│    │  Inferencia probabilística calibrada y consistente│
│    └─ Phase2DeepLearningClassifier (Slot Fase 2)       │
│       Inyección directa de pesos PyTorch / ONNX        │
└──────────────────────────────────┬─────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────┐
│ 3. MOTOR DETERMINISTA DE REGLAS (TriageRuleEngine)     │
│    - Cruce: Defecto + Confianza + Criticidad           │
│    - Matriz de Políticas de Calidad IF-THEN            │
│    - Asignación de Prioridad: Baja, Media, Alta, Urgente│
│    - Generación de Prescripción Operativa de Piso      │
└──────────────────────────────────┬─────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────┐
│ 4. INTERFAZ GRADIO (Material Design Industrial Dark)   │
│    - Paneles de control y visualización de resultados  │
│    - Botones: Analizar, Calcular prioridad, Nueva pieza│
│    - Telemetría y auditoría de trazabilidad en JSON    │
└────────────────────────────────────────────────────────┘
```

---

## 3. Estructura del Proyecto

```
Proyecto FDI/
├── core/
│   ├── __init__.py            # Exportación unificada de paquetes
│   ├── schemas.py             # Enums tipados, DefectPrediction y TriageResult
│   ├── preprocessor.py        # Normalización a 200x200 px y escala de grises (NEU)
│   ├── classifier.py          # Patrón Strategy: Mock Fase 1 y CNN Fase 2
│   ├── rules_engine.py        # Motor determinista de reglas IF-THEN
│   └── sample_generator.py    # Generador de especímenes sintéticos de prueba
├── samples/                   # Especímenes sintéticos generados (200x200 px)
│   ├── muestra_grieta.png
│   ├── muestra_picadura.png
│   ├── muestra_rayadura.png
│   └── muestra_conforme.png
├── app.py                     # Aplicación principal Gradio Blocks
└── README.md                  # Documentación técnica exhaustiva
```

---

## 4. Requisitos y Puesta en Marcha

### Prerrequisitos
* Python 3.10 o superior (compatible con Python 3.13 / 3.14).
* Paquetes de dependencias: `gradio` y `pillow`.

### Instalación
```bash
python -m pip install gradio pillow
```

### Ejecución de la Aplicación
Para iniciar el servidor local de Gradio:
```bash
python app.py
```

La interfaz se desplegará localmente en:
`http://127.0.0.1:7860`

---

## 5. Recorrido de Operación (Workflow del Inspector)

1. **Carga de Espécimen:**
   * El usuario arrastra o selecciona una fotografía en la caja `Fotografía de la Pieza` (o hace clic sobre cualquiera de los ejemplos precargados en la parte inferior).
2. **Contexto de Producción:**
   * Se registra el `Código / Lote` (ej. `LOT-2026-X4B`).
   * Se asigna el `ID Prueba` (ej. `TST-00101`).
   * Se selecciona la `Criticidad del Componente`: `Baja`, `Media` o `Crítica`.
3. **Ejecución del Análisis:**
   * Clic en **`🔍 Analizar superficie`**: el clasificador genera el `Defecto Estimado` y la `Confianza (%)`, evaluando de inmediato la prioridad y recomendación.
4. **Recálculo Heurístico Independiente (Sección 7.4):**
   * Si el operador modifica la criticidad de la pieza sin alterar la imagen, puede pulsar **`⚖️ Calcular prioridad`** para re-evaluar la decisión determinista al instante.
5. **Reinicio de Ciclo:**
   * El botón **`🔄 Nueva pieza`** limpia todos los campos y restaura el formulario a su estado de espera.

---

## 6. Transición a la Fase 2 (Deep Learning)

Para incorporar un modelo real entrenado (ej. ResNet o MobileNet entrenado con la base *NEU Surface Defect Database*):

1. En `core/classifier.py`, complete la carga de pesos en `Phase2DeepLearningClassifier`:
   ```python
   # Ejemplo con ONNX Runtime:
   import onnxruntime as ort
   self.session = ort.InferenceSession("weights/neu_classifier.onnx")
   ```
2. En `app.py`, reemplace la inicialización del clasificador activo:
   ```python
   # De:
   ACTIVE_CLASSIFIER = MockDefectClassifier()
   # A:
   ACTIVE_CLASSIFIER = Phase2DeepLearningClassifier("weights/neu_classifier.onnx")
   ```
El pipeline de preprocesamiento, la interfaz de Gradio y el motor de triaje continuarán funcionando sin ninguna otra modificación requerida.

