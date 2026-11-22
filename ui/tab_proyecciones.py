import json
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from engine.pdf_generator import PDFReportEngine
from engine.proyecciones import (
    comparar_modelos_proyeccion,
    pronostico_series_tiempo,
    regresion_exponencial,
    regresion_lineal_simple,
    regresion_logaritmica,
    regresion_polinomial,
    regresion_potencial,
)


def _render_geogebra_proyeccion(
    x_pts,
    y_pts,
    x_eval: float,
    y_eval: float,
    expr_ggb: str,
    titulo: str,
    curvas_extra: list | None = None,
):
    """Renderiza el Laboratorio Gráfico GeoGebra (Módulo 4) para el Módulo 6."""
    x_all = list(x_pts) + [x_eval]
    y_all = list(y_pts) + [y_eval]
    x_min, x_max = min(x_all), max(x_all)
    y_min, y_max = min(y_all), max(y_all)

    dx = max((x_max - x_min) * 0.25, 1.0)
    dy = max((y_max - y_min) * 0.25, 1.0)

    comandos = [
        f"f(x) = {expr_ggb}",
        "SetColor(f, 0, 102, 204)",
        "SetLineThickness(f, 4)",
    ]

    if curvas_extra:
        colores = [(16, 185, 129), (245, 158, 11), (139, 92, 246), (100, 116, 139)]
        for idx, (_, expr_c) in enumerate(curvas_extra):
            r, g, b = colores[idx % len(colores)]
            fn_name = f"g_{idx+1}"
            comandos.append(f"{fn_name}(x) = {expr_c}")
            comandos.append(f"SetColor({fn_name}, {r}, {g}, {b})")
            comandos.append(f"SetLineThickness({fn_name}, 2)")
            comandos.append(f"SetLineStyle({fn_name}, 1)")

    for idx, (xi, yi) in enumerate(zip(x_pts, y_pts), start=1):
        comandos.append(f"D_{idx} = ({xi}, {yi})")
        comandos.append(f"SetColor(D_{idx}, 30, 30, 30)")
        comandos.append(f"SetPointSize(D_{idx}, 4)")

    comandos.append(f"PProy = ({x_eval}, {y_eval})")
    comandos.append("SetColor(PProy, 220, 38, 38)")
    comandos.append("SetPointSize(PProy, 6)")
    comandos.append("ShowLabel(PProy, true)")
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
        <div id="ggb-proy" style="width:100%; height:520px; border:1px solid #e2e8f0; border-radius:8px;"></div>
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
                applet.inject('ggb-proy');
            }});
        </script>
    </body>
    </html>
    """
    st.subheader(f"Laboratorio Gráfico GeoGebra — {titulo}")
    components.html(html_ggb, height=540)


def render_tab_proyecciones():
    st.header("Módulo 6: Ajuste de Curvas, Regresión y Proyecciones")
    st.caption(
        "Basado en Nieves Hurtado & Domínguez Sánchez (Cap. 6) — Mínimos Cuadrados, "
        "Linealización de Modelos No Lineales, Comparativa R² y Series de Tiempo."
    )

    modelo_sel = st.selectbox(
        "Seleccione la Pieza / Modelo de Ajuste:",
        [
            "6.1.A — Regresión Lineal Simple (y = a0 + a1·x)",
            "6.2.B — Regresión Polinomial de Grado m",
            "6.3.C — Regresión Exponencial (y = a·e^(b·x))",
            "6.4.D — Regresión Potencial (y = a·x^b)",
            "6.5.E — Regresión Logarítmica (y = a + b·ln(x)) y Comparador Global",
            "6.6.F — Series de Tiempo (PMS, PMP, Suavizamiento Exponencial, Holt y Estacional)",
        ],
        key="sb_modelo_proyecciones",
    )

    col_izq, col_der = st.columns([1.1, 1.3])

    # Parámetros por defecto
    grado_poli = 2
    x_eval = 7.0
    cod_ts = "SES"
    ventana_k = 3
    horizonte_h = 3
    alpha = 0.35
    beta = 0.15
    periodo_L = 4
    pesos_list = [0.2, 0.3, 0.5]

    with col_izq:
        st.subheader("1. Serie de Datos Experimentales $(x_i, y_i)$")
        df_default = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "y": [2.5, 4.1, 6.8, 10.9, 17.5, 28.2],
            }
        )
        df_datos = st.data_editor(
            df_default,
            num_rows="dynamic",
            use_container_width=True,
            key="editor_datos_proyecciones",
        )

        # Controles específicos ANTES del botón Calcular
        if modelo_sel.startswith("6.6.F"):
            sub_ts = st.selectbox(
                "Técnica de Serie de Tiempo:",
                [
                    "SES — Suavizamiento Exponencial Simple",
                    "HOLT — Suavizamiento Doble de Holt (Tendencia)",
                    "PMS — Promedio Móvil Simple",
                    "PMP — Promedio Móvil Ponderado",
                    "ESTACIONAL — Descomposición Estacional Multiplicativa",
                ],
                key="submetodo_ts",
            )
            cod_ts = sub_ts.split(" — ")[0]

            c_p1, c_p2 = st.columns(2)
            ventana_k = c_p1.number_input(
                "Ventana k (PMS):", min_value=1, max_value=15, value=3, step=1
            )
            horizonte_h = c_p2.number_input(
                "Periodos futuros (h):", min_value=1, max_value=24, value=3, step=1
            )

            if cod_ts in ("SES", "HOLT"):
                alpha = st.slider("Constante de nivel (α):", 0.05, 1.0, 0.35, 0.05)
            if cod_ts == "HOLT":
                beta = st.slider("Constante de tendencia (β):", 0.05, 1.0, 0.15, 0.05)
            if cod_ts == "PMP":
                pesos_str = st.text_input(
                    "Pesos (del más antiguo al más reciente, separados por coma):",
                    "0.2, 0.3, 0.5",
                )
                try:
                    pesos_list = [
                        float(p.strip()) for p in pesos_str.split(",") if p.strip()
                    ]
                except ValueError:
                    st.error("Formato inválido en los pesos. Use números separados por coma.")
                    return
            if cod_ts == "ESTACIONAL":
                periodo_L = st.number_input(
                    "Longitud del ciclo estacional (L):",
                    min_value=2,
                    max_value=12,
                    value=4,
                    step=1,
                )
        else:
            if modelo_sel.startswith("6.2.B") or modelo_sel.startswith("6.5.E"):
                grado_poli = st.number_input(
                    "Grado del polinomio (m):",
                    min_value=1,
                    max_value=8,
                    value=2,
                    step=1,
                    key="grado_poli_proy",
                )

            x_eval = st.number_input(
                "Valor a proyectar ($x_{eval}$):",
                value=7.0,
                step=0.5,
                format="%.6f",
                key="x_eval_proyecciones",
            )

        ejecutar = st.button(
            "Calcular Ajuste y Proyección",
            type="primary",
            use_container_width=True,
            key="btn_calcular_proyecciones",
        )

    if not ejecutar:
        return

    df_limpio = df_datos.dropna(subset=["x", "y"])
    x_vals = df_limpio["x"].astype(float).tolist()
    y_vals = df_limpio["y"].astype(float).tolist()

    # -------------------------------------------------------------------------
    # RAMA A: PIEZA 6.6.F — SERIES DE TIEMPO
    # -------------------------------------------------------------------------
    if modelo_sel.startswith("6.6.F"):
        res_ts = pronostico_series_tiempo(
            y_vals=y_vals,
            submetodo=cod_ts,
            ventana_k=int(ventana_k),
            pesos=pesos_list,
            alpha=float(alpha),
            beta=float(beta),
            horizonte_h=int(horizonte_h),
            periodo_estacional=int(periodo_L),
            t_vals=x_vals,
        )

        if not res_ts["exito"]:
            st.error(res_ts["mensaje"])
            return

        m_ts = res_ts["metricas"]
        with col_der:
            st.subheader(f"2. Resultados — {res_ts['metodo']}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                f"Pronóstico t={res_ts['t_futuro'][0]:.4g}",
                f"{res_ts['resultado']:.4f}",
            )
            m2.metric("MAD", f"{m_ts['mad']:.4f}")
            m3.metric("RMSE", f"{m_ts['rmse']:.4f}")
            m4.metric("MAPE (%)", f"{m_ts['mape']:.2f}%")

            if "indices_estacionales" in res_ts["detalles_extra"]:
                st.info(
                    f"**Tendencia base:** `{res_ts['detalles_extra']['tendencia_base']}`  \n"
                    f"**Índices Estacionales:** `{res_ts['detalles_extra']['indices_estacionales']}`"
                )

            st.markdown("**Tabla de Pronósticos Futuros:**")
            df_fut = pd.DataFrame(
                {
                    "Periodo Futuro (t)": res_ts["t_futuro"],
                    "Pronóstico ŷ_t": res_ts["pronosticos_futuros"],
                }
            )
            st.dataframe(df_fut.style.format("{:.4f}"), use_container_width=True)

        st.divider()
        st.subheader("3. Histórico: Observado vs. Suavizado / Pronosticado")
        y_arr = np.array(res_ts["y"])
        y_aj = np.array(res_ts["y_ajuste"], dtype=float)
        df_hist_ts = pd.DataFrame(
            {
                "Periodo (t)": res_ts["t"],
                "Valor Real (y_t)": y_arr,
                "Pronóstico Histórico (ŷ_t)": y_aj,
                "Error (y_t - ŷ_t)": y_arr - y_aj,
            }
        )
        st.dataframe(df_hist_ts.style.format("{:.4f}"), use_container_width=True)

        # Exportación PDF con PDFReportEngine
        iters_pdf = [
            {
                "Periodo_t": round(float(tf), 4),
                "Pronostico": round(float(pf), 6),
            }
            for tf, pf in zip(res_ts["t_futuro"], res_ts["pronosticos_futuros"])
        ]
        pdf_bytes = PDFReportEngine.generar_reporte_pdf(
            titulo_metodo=res_ts["metodo"],
            entradas_dict={
                "Observaciones (n)": len(y_vals),
                "Horizonte h": int(horizonte_h),
                "MAD / RMSE / MAPE": f"{m_ts['mad']:.4f} / {m_ts['rmse']:.4f} / {m_ts['mape']:.2f}%",
            },
            resultado_str=f"Pronóstico siguiente y({res_ts['t_futuro'][0]}) = {res_ts['resultado']:.6f}",
            iteraciones_list=iters_pdf,
        )
        st.download_button(
            label="📄 Exportar Cálculo como PDF",
            data=pdf_bytes,
            file_name="Reporte_Series_de_Tiempo.pdf",
            mime="application/pdf",
            key="btn_pdf_series_tiempo",
        )

        _render_geogebra_proyeccion(
            res_ts["t"],
            res_ts["y"],
            res_ts["t_futuro"][0],
            res_ts["resultado"],
            res_ts["funcion_geogebra"],
            res_ts["metodo"],
        )
        return

    # -------------------------------------------------------------------------
    # RAMA B: PIEZAS 6.1.A – 6.5.E (REGRESIÓN POR MÍNIMOS CUADRADOS)
    # -------------------------------------------------------------------------
    if modelo_sel.startswith("6.1.A"):
        res = regresion_lineal_simple(x_vals, y_vals, x_eval)
    elif modelo_sel.startswith("6.2.B"):
        res = regresion_polinomial(x_vals, y_vals, int(grado_poli), x_eval)
    elif modelo_sel.startswith("6.3.C"):
        res = regresion_exponencial(x_vals, y_vals, x_eval)
    elif modelo_sel.startswith("6.4.D"):
        res = regresion_potencial(x_vals, y_vals, x_eval)
    else:
        res = regresion_logaritmica(x_vals, y_vals, x_eval)

    if not res["exito"]:
        st.error(res["mensaje"])
        if not modelo_sel.startswith("6.5.E"):
            return

    if res["exito"]:
        m_info = res["metricas"]
        with col_der:
            st.subheader(f"2. Resultados — {res['metodo']}")
            c1, c2, c3 = st.columns(3)
            c1.metric(
                f"Proyección y({res['x_eval']:.4g})", f"{res['resultado']:.6f}"
            )
            c2.metric("Coef. Determinación R²", f"{m_info['r2']:.6f}")
            c3.metric("Error Estándar s_{y/x}", f"{m_info['error_estandar']:.6f}")

            if res.get("es_extrapolacion"):
                st.warning(
                    f"Extrapolación: x = {res['x_eval']} está fuera del rango muestral "
                    f"[{res['rango_x'][0]}, {res['rango_x'][1]}]."
                )
            if res.get("advertencia"):
                st.warning(res["advertencia"])

            st.markdown("**Ecuación de Ajuste Obtenida:**")
            st.latex(res["ecuacion_latex"])

        st.divider()
        st.subheader("3. Tabla de Residuos y Valores Estimados")
        y_arr = np.array(res["y"])
        y_est_arr = np.array(res["y_estimado"])
        residuos = y_arr - y_est_arr

        df_residuos = pd.DataFrame(
            {
                "x_i": res["x"],
                "y_i (Observado)": y_arr,
                "ŷ_i (Modelo)": y_est_arr,
                "Residuo e_i = y_i - ŷ_i": residuos,
                "Residuo² (e_i²)": residuos ** 2,
            }
        )
        st.dataframe(df_residuos.style.format("{:.6f}"), use_container_width=True)

        if "matriz_normal" in res:
            with st.expander("Ver Sistema de Ecuaciones Normales [A]{a} = {B}"):
                df_mat = pd.DataFrame(res["matriz_normal"])
                df_mat["= B"] = res["vector_normal"]
                st.dataframe(df_mat.style.format("{:.6f}"), use_container_width=True)

        # Exportación PDF con PDFReportEngine
        tabla_pdf = [
            {
                "x_i": round(float(xi), 4),
                "y_i": round(float(yi), 4),
                "y_est": round(float(ye), 4),
                "Residuo": round(float(ei), 6),
            }
            for xi, yi, ye, ei in zip(res["x"], y_arr, y_est_arr, residuos)
        ]
        pdf_bytes = PDFReportEngine.generar_reporte_pdf(
            titulo_metodo=res["metodo"],
            entradas_dict={
                "Ecuación": res["funcion_geogebra"],
                "x_eval": res["x_eval"],
                "R²": f"{m_info['r2']:.6f}",
                "Error Estándar": f"{m_info['error_estandar']:.6f}",
            },
            resultado_str=f"y({res['x_eval']}) = {res['resultado']:.6f}",
            iteraciones_list=tabla_pdf,
        )
        st.download_button(
            label="📄 Exportar Cálculo como PDF",
            data=pdf_bytes,
            file_name=f"Reporte_{res['metodo'][:15].replace(' ', '_')}.pdf",
            mime="application/pdf",
            key="btn_pdf_proyecciones",
        )

    # Comparador Global en 6.5.E
    curvas_comparativas = None
    if modelo_sel.startswith("6.5.E"):
        st.divider()
        st.subheader(
            "4. Comparador Global de Modelos de Proyección (Piezas 6.1.A – 6.5.E)"
        )
        comp = comparar_modelos_proyeccion(x_vals, y_vals, x_eval, int(grado_poli))
        if comp["exito"]:
            mejor = comp["mejor_modelo"]
            st.success(
                f"**Mejor modelo según R²:** {mejor['modelo']} | "
                f"**R² = {mejor['r2']:.6f}** | **Proyección en x={x_eval}: {mejor['proyeccion']:.6f}**"
            )
            df_rank = pd.DataFrame(
                [
                    {
                        "Modelo": item["modelo"],
                        "R²": item["r2"],
                        "Error Estándar s_{y/x}": item["error_estandar"],
                        "Suma Residuos Sr": item["sr"],
                        f"Proyección y({x_eval})": item["proyeccion"],
                        "Expresión": item["funcion_geogebra"],
                    }
                    for item in comp["ranking"]
                ]
            )
            st.dataframe(df_rank, use_container_width=True)

            if comp["descartados"]:
                with st.expander("Modelos descartados por restricciones de dominio"):
                    for d in comp["descartados"]:
                        st.write(f"- **{d['modelo']}**: {d['motivo']}")

            if not res["exito"]:
                res = mejor["detalle"]

            curvas_comparativas = [
                (item["modelo"], item["funcion_geogebra"])
                for item in comp["ranking"]
                if item["funcion_geogebra"] != res["funcion_geogebra"]
            ]

    if res["exito"]:
        _render_geogebra_proyeccion(
            res["x"],
            res["y"],
            res["x_eval"],
            res["resultado"],
            res["funcion_geogebra"],
            res["metodo"],
            curvas_extra=curvas_comparativas,
        )
