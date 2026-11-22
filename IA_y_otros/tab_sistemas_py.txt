import numpy as np
import pandas as pd
import streamlit as st
from engine.pdf_generator import PDFReportEngine
from engine.sistemas_lineales import SistemasLinealesEngine


def render_tab_sistemas():
    st.header("Módulo 3: Sistemas de Ecuaciones Lineales (A · x = b)")

    metodo = st.selectbox(
        "Seleccione el Método Numérico:",
        [
            "Eliminación Gaussiana con Pivotaje Parcial",
            "Método de Gauss-Jordan con Pivotaje Parcial",
            "Método Iterativo de Jacobi",
            "Método Iterativo de Gauss-Seidel",
        ],
        key="sb_metodo_sistemas",
    )

    st.markdown("---")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("Configuración del Sistema")
        n = st.number_input(
            "Dimensiones del sistema (n x n):",
            min_value=2,
            max_value=6,
            value=3,
            step=1,
            key="ni_dimension_sistemas",
        )

        st.markdown("**Matriz A y Vector b:**")
        st.caption(
            "Ingrese los coeficientes de A y los términos independientes b."
        )

        if n == 3:
            default_matrix = [
                [10.0, 1.0, 2.0, 3.0],
                [4.0, 6.0, -1.0, 9.0],
                [-2.0, 3.0, 8.0, 51.0],
            ]
            cols = ["x1", "x2", "x3", "b"]
        else:
            default_matrix = np.zeros((n, n + 1)).tolist()
            cols = [f"x{i+1}" for i in range(n)] + ["b"]

        df_ab = pd.DataFrame(default_matrix, columns=cols)
        df_edited = st.data_editor(
            df_ab,
            use_container_width=True,
            num_rows="fixed",
            key=f"editor_ab_{n}",
        )

        # Parámetros adicionales para métodos iterativos
        if metodo in [
            "Método Iterativo de Jacobi",
            "Método Iterativo de Gauss-Seidel",
        ]:
            st.markdown("---")
            st.subheader("Parámetros Iterativos")

            tipo_error = st.radio(
                "Criterio de Evaluación de Error:",
                [
                    "Residuo del sistema (|b - Ax|)",
                    "Distancia entre iteraciones (||x_new - x_old||)",
                ],
                key="radio_tipo_error_sistemas",
            )

            col_tol, col_max = st.columns(2)
            with col_tol:
                tol = st.number_input(
                    "Tolerancia de error (E):",
                    value=0.0001,
                    format="%.5f",
                    step=0.0001,
                    key="ni_tol_sistemas",
                )
            with col_max:
                max_iter = st.slider(
                    "Máximo de iteraciones:",
                    min_value=1,
                    max_value=100,
                    value=30,
                    key="slider_max_iter_sistemas",
                )

            st.markdown("**Vector Inicial x⁽⁰⁾:**")
            col_x0 = st.columns(n)
            x0_vals = []
            for i in range(n):
                with col_x0[i]:
                    val_init = (
                        0.3
                        if (n == 3 and i == 0)
                        else (
                            1.5
                            if (n == 3 and i == 1)
                            else (6.375 if (n == 3 and i == 2) else 0.0)
                        )
                    )
                    val = st.number_input(
                        f"x{i+1}⁰",
                        value=val_init,
                        format="%.3f",
                        key=f"x0_sistemas_{i}",
                    )
                    x0_vals.append(val)

        btn_resolver = st.button(
            "Resolver Sistema", type="primary", key="btn_resolver_sistemas"
        )

    with col2:
        st.subheader("Resultados y Procedimiento")

        if btn_resolver:
            try:
                ab_matrix = df_edited.to_numpy(dtype=float)
                A = ab_matrix[:, :n]
                b = ab_matrix[:, n]

                res = None
                t_err = (
                    "residuo" if "Residuo" in tipo_error else "distancia"
                ) if metodo in [
                    "Método Iterativo de Jacobi",
                    "Método Iterativo de Gauss-Seidel",
                ] else None

                if metodo == "Eliminación Gaussiana con Pivotaje Parcial":
                    res = SistemasLinealesEngine.eliminacion_gaussiana(A, b)
                elif metodo == "Método de Gauss-Jordan con Pivotaje Parcial":
                    res = SistemasLinealesEngine.gauss_jordan(A, b)
                elif metodo == "Método Iterativo de Jacobi":
                    res = SistemasLinealesEngine.jacobi(
                        A, b, x0_vals, tol, int(max_iter), t_err
                    )
                elif metodo == "Método Iterativo de Gauss-Seidel":
                    res = SistemasLinealesEngine.gauss_seidel(
                        A, b, x0_vals, tol, int(max_iter), t_err
                    )

                if res:
                    st.success(f"**Método:** {res['metodo']}")

                    st.markdown("### Vector Solución (x):")
                    sol_df = pd.DataFrame(
                        [res["solucion"]],
                        columns=[f"x{i+1}" for i in range(n)],
                    )
                    st.dataframe(sol_df, use_container_width=True)

                    if "pasos" in res:
                        st.markdown("### Operaciones Renglón / Pasos:")
                        for p in res["pasos"]:
                            st.caption(f"• {p}")

                    if "iteraciones" in res:
                        st.markdown("### Tabla de Iteraciones:")
                        st.dataframe(
                            res["iteraciones"],
                            use_container_width=True,
                        )

                    # Exportación PDF (Módulo 4 del Mapa de Piezas)
                    st.markdown("---")
                    # Dentro de ui/tab_sistemas.py al presionar el botón de calcular:
                    pdf_bytes = PDFReportEngine.generar_reporte_pdf(
                        titulo_metodo=res["metodo"],
                        entradas_dict={"Dimensión del sistema": f"{n}x{n}"},
                        resultado_str=f"Solución x = {res['solucion']}",
                        iteraciones_list=res.get("iteraciones", []),
                        pasos_matriz_list=res.get("pasos_matriz", []),  # <--- Envía los pasos matriciales paso a paso
                    )

                    st.download_button(
                        label="📄 Exportar Cálculo como PDF",
                        data=pdf_bytes,
                        file_name=f"Reporte_{res['metodo'].replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        key="btn_pdf_sistemas",
                    )

            except Exception as e:
                st.error(
                    f"Error al resolver el sistema. Detalle técnico: {e}"
                )