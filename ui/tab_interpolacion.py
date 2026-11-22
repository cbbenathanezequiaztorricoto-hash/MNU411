import json
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from engine.interpolacion import (
    InterpolacionEngine,
    interpolacion_diferencias_divididas,
    interpolacion_lineal,
)
from engine.pdf_generator import PDFReportEngine


def _render_geogebra_interpolacion(
    x_pts, y_pts, x_eval, y_eval, expr_ggb: str | None, titulo: str
):
    """Renderiza el Laboratorio Gráfico GeoGebra (Módulo 4) para Interpolación."""
    x_all = list(x_pts) + [x_eval]
    y_all = list(y_pts) + [y_eval]
    x_min, x_max = min(x_all), max(x_all)
    y_min, y_max = min(y_all), max(y_all)

    dx = max((x_max - x_min) * 0.25, 1.0)
    dy = max((y_max - y_min) * 0.25, 1.0)

    comandos = []
    if expr_ggb:
        comandos.extend(
            [
                f"f(x) = {expr_ggb}",
                "SetColor(f, 0, 102, 204)",
                "SetLineThickness(f, 4)",
            ]
        )

    for idx, (xi, yi) in enumerate(zip(x_pts, y_pts), start=1):
        comandos.append(f"P_{idx} = ({xi}, {yi})")
        comandos.append(f"SetColor(P_{idx}, 30, 30, 30)")
        comandos.append(f"SetPointSize(P_{idx}, 4)")

    comandos.append(f"PEval = ({x_eval}, {y_eval})")
    comandos.append("SetColor(PEval, 220, 38, 38)")
    comandos.append("SetPointSize(PEval, 6)")
    comandos.append("ShowLabel(PEval, true)")
    comandos.append(
        f"SetCoordSystem({x_min - dx}, {x_max + dx}, {y_min - dy}, {y_max + dy})"
    )

    comandos_json = json.dumps(comandos)

    html_ggb = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <script src="https://www.geogebra.org/apps/deployggb.js"></script>
    </head>
    <body style="margin:0; padding:0; background-color:#ffffff;">
        <div id="ggb-element" style="width:100%; height:520px; border:1px solid #e2e8f0; border-radius:8px;"></div>
        <script>
            var params = {{
                "appName": "graphing",
                "width": "100%",
                "height": 520,
                "showToolBar": true,
                "showAlgebraInput": true,
                "showMenuBar": false,
                "enableRightClick": true,
                "language": "es",
                "appletOnLoad": function(api) {{
                    var cmds = {comandos_json};
                    cmds.forEach(function(cmd) {{
                        api.evalCommand(cmd);
                    }});
                }}
            }};
            var applet = new GGBApplet(params, true);
            window.addEventListener("load", function() {{
                applet.inject('ggb-element');
            }});
        </script>
    </body>
    </html>
    """
    st.subheader(f"Laboratorio Gráfico GeoGebra — {titulo}")
    components.html(html_ggb, height=540)


def render_tab_interpolacion():
    st.header("Módulo 5: Interpolación Funcional y Aproximación Polinomial")
    st.caption(
        "Basado en Nieves Hurtado & Domínguez Sánchez (Cap. 5) — Diferencias Finitas, "
        "Lagrange, Diferencias Divididas e Interpolación Lineal."
    )

    metodo = st.selectbox(
        "Seleccione la Pieza / Método de Interpolación:",
        [
            "5.1.A — Newton hacia Adelante (Diferencias finitas, h constante)",
            "5.2.B — Newton hacia Atrás (Diferencias finitas, h constante)",
            "5.3.C — Polinomio de Lagrange",
            "5.4.C — Diferencias Divididas de Newton (Paso variable o constante)",
            "5.5.D — Interpolación Lineal (Simple y Segmentaria por tramos)",
        ],
        key="sb_metodo_interpolacion",
    )

    col_izq, col_der = st.columns([1.1, 1.3])

    with col_izq:
        st.subheader("1. Tabla de Puntos $(x_i, y_i)$")
        df_default = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0],
                "y": [2.0, 4.5, 3.8, 7.2],
            }
        )
        df_puntos = st.data_editor(
            df_default,
            num_rows="dynamic",
            use_container_width=True,
            key="editor_puntos_interpolacion",
        )

        x_eval = st.number_input(
            "Valor a evaluar ($x_{eval}$):",
            value=2.5,
            step=0.1,
            format="%.6f",
            key="x_eval_interpolacion",
        )

        calcular = st.button(
            "Calcular Interpolación",
            type="primary",
            use_container_width=True,
            key="btn_calcular_interpolacion",
        )

    if not calcular:
        return

    df_limpio = df_puntos.dropna(subset=["x", "y"])
    x_vals = df_limpio["x"].astype(float).tolist()
    y_vals = df_limpio["y"].astype(float).tolist()

    # -------------------------------------------------------------------------
    # PIEZAS 5.1.A, 5.2.B Y 5.3.C (InterpolacionEngine)
    # -------------------------------------------------------------------------
    if metodo.startswith(("5.1.A", "5.2.B", "5.3.C")):
        try:
            if metodo.startswith("5.1.A"):
                res = InterpolacionEngine.newton_adelante(x_vals, y_vals, x_eval)
            elif metodo.startswith("5.2.B"):
                res = InterpolacionEngine.newton_atras(x_vals, y_vals, x_eval)
            else:
                res = InterpolacionEngine.lagrange(x_vals, y_vals, x_eval)

            # Obtener polinomio simbólico equivalente para GeoGebra
            res_poli = interpolacion_diferencias_divididas(x_vals, y_vals, x_eval)
            expr_ggb = res_poli["polinomio_geogebra"] if res_poli.get("exito") else None

            with col_der:
                st.subheader(f"2. Resultados — {res['metodo']}")
                st.metric(
                    label=f"Valor Interpolado P({x_eval:.6g})",
                    value=f"{res['solucion']:.6f}",
                )
                if "h" in res:
                    st.info(
                        f"**Nodo base $x_0$:** `{res['x0']}` | "
                        f"**Paso $h$:** `{res['h']}` | **Parámetro $P$:** `{res['P']}`"
                    )
                elif "tipo_evaluacion" in res:
                    st.info(f"**Tipo de evaluación:** {res['tipo_evaluacion']}")

            st.divider()
            st.subheader("3. Tabla de Diferencias / Desglose de Términos")
            st.dataframe(res["iteraciones"], use_container_width=True)

            if "terminos" in res:
                with st.expander("Ver aporte por orden polinomial"):
                    st.dataframe(res["terminos"], use_container_width=True)

            pdf_bytes = PDFReportEngine.generar_reporte_pdf(
                titulo_metodo=res["metodo"],
                entradas_dict={"x": x_vals, "f(x)": y_vals, "x_eval": x_eval},
                resultado_str=f"P({x_eval}) = {res['solucion']}",
                iteraciones_list=res.get("iteraciones", []),
            )
            st.download_button(
                label="📄 Exportar Cálculo como PDF",
                data=pdf_bytes,
                file_name=f"Reporte_{res['metodo'].replace(' ', '_')}.pdf",
                mime="application/pdf",
                key="btn_pdf_interp_clasica",
            )

            _render_geogebra_interpolacion(
                x_vals, y_vals, x_eval, res["solucion"], expr_ggb, res["metodo"]
            )
        except Exception as e:
            st.error(f"Error en el cálculo: {e}")

    # -------------------------------------------------------------------------
    # PIEZA 5.4.C: DIFERENCIAS DIVIDIDAS
    # -------------------------------------------------------------------------
    elif metodo.startswith("5.4.C"):
        res = interpolacion_diferencias_divididas(x_vals, y_vals, x_eval)
        if not res["exito"]:
            st.error(res["mensaje"])
            return

        with col_der:
            st.subheader("2. Resultados — Diferencias Divididas")
            st.metric(
                label=f"Valor Interpolado P({res['x_eval']:.6g})",
                value=f"{res['resultado']:.8f}",
            )
            if res["es_extrapolacion"]:
                st.warning(
                    f"Advertencia: x = {res['x_eval']} está fuera del rango nodal "
                    f"[{res['rango_x'][0]}, {res['rango_x'][1]}] (Extrapolación)."
                )

            st.markdown("**Polinomio en Base de Newton:**")
            st.latex(f"P_{{{res['grado']}}}(x) = {res['polinomio_newton_latex']}")

            st.markdown("**Polinomio Canónico Expandido:**")
            st.latex(f"P_{{{res['grado']}}}(x) = {res['polinomio_expandido_latex']}")

        st.divider()
        st.subheader("3. Tabla de Diferencias Divididas")
        n = len(res["x"])
        cols_tabla = ["x_i", "f[x_i]"] + [f"Orden {j}" for j in range(1, n)]
        matriz_vista = np.column_stack([res["x"], res["tabla_diferencias"]])
        df_tabla = pd.DataFrame(matriz_vista, columns=cols_tabla)
        st.dataframe(df_tabla.style.format("{:.6f}"), use_container_width=True)

        with st.expander("Ver cálculo paso a paso de diferencias divididas"):
            for p in res["pasos"]:
                st.code(f"{p['formula']} = {p['valor']:.8f}", language="text")

        pdf_bytes = PDFReportEngine.generar_reporte_pdf(
            titulo_metodo="Diferencias Divididas de Newton (Pieza 5.4.C)",
            entradas_dict={
                "x": res["x"],
                "y": res["y"],
                "x_eval": res["x_eval"],
                "Polinomio": res["polinomio_geogebra"],
            },
            resultado_str=f"P({res['x_eval']}) = {res['resultado']:.8f}",
            iteraciones_list=res.get("pasos", []),
        )
        st.download_button(
            label="📄 Exportar Cálculo como PDF",
            data=pdf_bytes,
            file_name="Reporte_Diferencias_Divididas.pdf",
            mime="application/pdf",
            key="btn_pdf_dif_div",
        )

        _render_geogebra_interpolacion(
            res["x"],
            res["y"],
            res["x_eval"],
            res["resultado"],
            res["polinomio_geogebra"],
            "Diferencias Divididas de Newton",
        )

    # -------------------------------------------------------------------------
    # PIEZA 5.5.D: INTERPOLACIÓN LINEAL
    # -------------------------------------------------------------------------
    elif metodo.startswith("5.5.D"):
        res = interpolacion_lineal(x_vals, y_vals, x_eval)
        if not res["exito"]:
            st.error(res["mensaje"])
            return

        with col_der:
            st.subheader("2. Resultados — Interpolación Lineal")
            st.metric(
                label=f"Valor Interpolado S({res['x_eval']:.6g})",
                value=f"{res['resultado']:.8f}",
            )
            if res["es_extrapolacion"]:
                st.warning(f"Advertencia: {res['advertencia_extrapolacion']}.")

            tramo = res["tramo_activo"]
            st.info(
                f"**Tramo activo:** Tramo #{tramo['tramo']} en $[{tramo['x0']}, {tramo['x1']}]$  \n"
                f"**Ecuación:** `{tramo['ecuacion_texto']}`"
            )

        st.divider()
        st.subheader("3. Ecuaciones de Todos los Tramos Lineales")
        df_tramos = pd.DataFrame(res["segmentos"])[
            ["tramo", "x0", "x1", "y0", "y1", "pendiente", "ecuacion_texto"]
        ]
        st.dataframe(df_tramos, use_container_width=True)

        tramos_pdf = [
            {
                "Tramo": s["tramo"],
                "Intervalo": f"[{s['x0']}, {s['x1']}]",
                "Pendiente": round(s["pendiente"], 6),
                "Ecuación": s["ecuacion_texto"],
            }
            for s in res["segmentos"]
        ]
        pdf_bytes = PDFReportEngine.generar_reporte_pdf(
            titulo_metodo="Interpolación Lineal Segmentaria (Pieza 5.5.D)",
            entradas_dict={"x": res["x"], "y": res["y"], "x_eval": res["x_eval"]},
            resultado_str=f"S({res['x_eval']}) = {res['resultado']:.8f}",
            iteraciones_list=tramos_pdf,
        )
        st.download_button(
            label="📄 Exportar Cálculo como PDF",
            data=pdf_bytes,
            file_name="Reporte_Interpolacion_Lineal.pdf",
            mime="application/pdf",
            key="btn_pdf_interp_lineal",
        )

        _render_geogebra_interpolacion(
            res["x"],
            res["y"],
            res["x_eval"],
            res["resultado"],
            res["polinomio_geogebra"],
            "Interpolación Lineal",
        )
