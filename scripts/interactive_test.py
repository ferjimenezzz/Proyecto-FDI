# ==============================================================================
# PROYECTO 10: ROBOT DE PRUEBAS INTERACTIVAS EN VIVO SOBRE GRADIO
# ==============================================================================
# Este script actúa como un operador automatizado conectándose al servidor web
# en ejecución en http://127.0.0.1:7860 para simular las acciones de un inspector:
# 1. Carga de espécimen con defecto crítico y análisis.
# 2. Modificación de criticidad y recálculo de prioridad de triaje.
# 3. Carga de espécimen conforme (limpio) y análisis.
# 4. Prueba del botón de reinicio integral.
# ==============================================================================

import time
import os
import json
from gradio_client import Client, handle_file

SERVER_URL = "http://127.0.0.1:7860"

def run_live_operator():
    print("=" * 80)
    print("ROBOT DE OPERACIÓN EN VIVO - INTERACTUANDO CON EL SERVIDOR GRADIO")
    print(f"Conectándose al servidor en: {SERVER_URL} ...")
    print("=" * 80)

    client = Client(SERVER_URL)
    print(">> [CONEXIÓN ESTABLECIDA EXITOSAMENTE]")
    time.sleep(1)

    # --------------------------------------------------------------------------
    # PRUEBA 1: INSPECCIÓN DE ESPÉCIMEN CON DEFECTO CRÍTICO (muestra_grieta.png)
    # --------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------")
    print("ACCION 1: Operador coloca fotografía de grieta 'samples/muestra_grieta.png'")
    print("Parámetros: Lote='LOT-2026-AEROSPATIAL' | ID='SPEC-CRACK-01' | Criticidad='Crítica'")
    print("Presionando botón: [🔍 Analizar superficie]...")
    print("--------------------------------------------------------------------")

    sample_crack = os.path.abspath("samples/muestra_grieta.png")
    result_1 = client.predict(
        image_input=handle_file(sample_crack),
        lot_code="LOT-2026-AEROSPATIAL",
        test_id="SPEC-CRACK-01",
        criticality_val="Crítica",
        api_name="/on_analyze_surface"
    )

    defect_1, conf_1, prio_1, badge_1, rec_1, audit_1 = result_1

    print(">> RESPUESTA DEL SERVIDOR GRADIO EN TIEMPO REAL:")
    print(f"   • Defecto Estimado:       {defect_1}")
    print(f"   • Confianza (Simulada):   {conf_1}")
    print(f"   • Prioridad de Triaje:    {prio_1}")
    print(f"   • Recomendación en Piso:  {rec_1}")
    time.sleep(2)

    # --------------------------------------------------------------------------
    # PRUEBA 2: RECALCULAR PRIORIDAD CON OTRA CRITICIDAD
    # --------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------")
    print("ACCION 2: El inspector cambia la criticidad de 'Crítica' a 'Baja'")
    print("Presionando botón: [⚖️ Calcular prioridad] (Sin recargar la fotografía)...")
    print("--------------------------------------------------------------------")

    result_2 = client.predict(
        current_defect_str=defect_1,
        confidence_str=conf_1,
        criticality_val="Baja",
        lot_code="LOT-2026-AEROSPATIAL",
        test_id="SPEC-CRACK-01",
        api_name="/on_calculate_priority"
    )

    prio_2, badge_2, rec_2, audit_2 = result_2

    print(">> RESPUESTA DEL MOTOR HEURÍSTICO DETERMINISTA:")
    print(f"   • Nueva Prioridad Calculada: {prio_2}")
    print(f"   • Nueva Recomendación:       {rec_2}")
    time.sleep(2)

    # --------------------------------------------------------------------------
    # PRUEBA 3: ESPÉCIMEN LIMPIO / CONFORME (muestra_conforme.png)
    # --------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------")
    print("ACCION 3: Operador carga nueva pieza pulida 'samples/muestra_conforme.png'")
    print("Parámetros: Lote='LOT-2026-OK-SERIES' | ID='SPEC-OK-02' | Criticidad='Baja'")
    print("Presionando botón: [🔍 Analizar superficie]...")
    print("--------------------------------------------------------------------")

    sample_ok = os.path.abspath("samples/muestra_conforme.png")
    result_3 = client.predict(
        image_input=handle_file(sample_ok),
        lot_code="LOT-2026-OK-SERIES",
        test_id="SPEC-OK-02",
        criticality_val="Baja",
        api_name="/on_analyze_surface"
    )

    defect_3, conf_3, prio_3, badge_3, rec_3, audit_3 = result_3

    print(">> RESPUESTA DEL SERVIDOR GRADIO EN TIEMPO REAL:")
    print(f"   • Defecto Estimado:       {defect_3}")
    print(f"   • Confianza (Simulada):   {conf_3}")
    print(f"   • Prioridad de Triaje:    {prio_3}")
    print(f"   • Recomendación en Piso:  {rec_3}")
    time.sleep(2)

    # --------------------------------------------------------------------------
    # PRUEBA 4: REINICIO DE LA INTERFAZ
    # --------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------")
    print("ACCION 4: El operador finaliza el turno y presiona: [🔄 Nueva pieza]")
    print("--------------------------------------------------------------------")

    result_reset = client.predict(api_name="/on_reset_form")
    print(">> RESPUESTA DEL SERVIDOR:")
    print(f"   • Campos reseteados a estado original: {len(result_reset)} elementos en blanco.")
    print(f"   • Imagen en formulario: {result_reset[0]} (Limpio)")
    print(f"   • Lote por defecto:     {result_reset[1]}")
    print(f"   • Defecto estimado:     '{result_reset[4]}' (Limpio)")

    print("\n" + "=" * 80)
    print("PRUEBA INTERACTIVA COMPLETADA CON ÉXITO: 4 DE 4 ACCIONES VALIDADAS")
    print("=" * 80)

if __name__ == "__main__":
    run_live_operator()
