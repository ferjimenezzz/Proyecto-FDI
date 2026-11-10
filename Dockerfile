# ==============================================================================
# PROYECTO 10: DETECTOR INTELIGENTE DE DEFECTOS SUPERFICIALES EN PIEZAS INDUSTRIALES
# INSTITUCIÓN: UNIVERSIDAD TECNOLÓGICA DE QUERÉTARO (UTEQ)
# ARCHIVO: Dockerfile
# ==============================================================================
# RESUMEN DEL ARCHIVO:
# Archivo de construcción de imagen de contenedor Docker para el despliegue
# reproducible de la aplicación Gradio en cualquier infraestructura en la nube
# (Hugging Face Spaces, Render, AWS ECS, GCP Cloud Run o servidores on-premise).
# ==============================================================================

FROM python:3.12-slim

# Evitar generación de archivos de bytecode .pyc y forzar volcado de logs a stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED="False" \
    HF_HUB_DISABLE_TELEMETRY="1" \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT=7860

# Definir directorio de trabajo
WORKDIR /app

# Instalar dependencias esenciales del sistema operativo
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar manifiesto de dependencias e instalarlas en el entorno Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    gradio==6.31.0 \
    pillow==12.3.0 \
    numpy>=1.26.0

# Copiar el código fuente completo del proyecto
COPY . /app

# Crear usuario sin privilegios root para cumplimiento de directrices de seguridad
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Exponer el puerto estándar de Gradio
EXPOSE 7860

# Comando de arranque del servidor
CMD ["python", "app.py"]
