import os
import streamlit as st
from engine.pdf_generator import PDFReportEngine
from engine.no_lineales import NoLinealesEngine


def render_tab_no_lineales():
    st.header("Módulo 2: Ecuaciones No Lineales")

    gif_map = {
        "Método de Bisección": "media/biseccion.gif",
        "Método de Regula Falsi (Posición Falsa)": "media/falsi.gif",
        "Método de la Secante": "media/secante.gif",
        "Newton-Raphson (1er Orden)": "media/newton.gif",
        "Newton-Raphson (2do Orden)": "media/newton.gif",
        "Método de Punto Fijo": "media/metodo26.gif",
    }
    # Selector del método numérico
    metodo = st.selectbox(
        "Seleccione el Método Numérico:",
        list(gif_map.keys()),
        key="sb_metodo_no_lineales",
    )
    st.markdown("---")
    # Distribución en 2 Columnas (Columna Izquierda: Parámetros + GIF | Columna Derecha: GeoGebra + Resultados)
    col_izq, col_der = st.columns([1, 1])

    with col_izq:
        st.subheader("1. Interpretación Geométrica")
        # Cargar el GIF correspondiente si existe en la carpeta media/
        gif_path = gif_map.get(metodo)
        if gif_path and os.path.exists(gif_path):
            st.image(
                gif_path,
                caption=f"Comportamiento dinámico: {metodo} (Créditos: @juliomulero)",
                use_container_width=True,
            )
        else:
            st.info("Animación conceptual no disponible para este método.")
        st.markdown("---")
        st.subheader("2. Parámetros de Entrada")
        # 1. PUNTO FIJO
                # Formulario según el método seleccionado
        if metodo == "Método de Punto Fijo":
            st.latex(r"x_{i+1} = g(x_i)")
            func_str = st.text_input(
                "Ingrese g(x):",
                value="(x**2 - exp(x)) / 5",
                key="input_g_pf",
            )
            x0 = st.number_input(
                "Valor inicial (x₀):", value=0.0, format="%.4f", key="x0_pf"
            )

        # 2. BISECCIÓN Y REGULA FALSI
        elif metodo in [
            "Método de Bisección",
            "Método de Regula Falsi (Posición Falsa)",
        ]:
            if metodo == "Método de Bisección":
                st.latex(r"x_M = \frac{a + b}{2}")
            else:
                st.latex(
                    r"x_M = b - \frac{f(b) \cdot (b - a)}{f(b) - f(a)}"
                )

            func_str = st.text_input(
                "Ingrese f(x):",
                value="x**3 - x - 2",
                placeholder="Ej: x**3 - x - 2",
            )
            col_a, col_b = st.columns(2)
            with col_a:
                val_a = st.number_input(
                    "Extremo inferior (a / x₀):", value=1.0, format="%.4f"
                )
            with col_b:
                val_b = st.number_input(
                    "Extremo superior (b / x₁):", value=2.0, format="%.4f"
                )

        # 3. SECANTE
        elif metodo == "Método de la Secante":
            st.latex(
                r"x_{n+1} = x_n - \frac{f(x_n) \cdot (x_n - x_{n-1})}{f(x_n) - f(x_{n-1})}"
            )
            func_str = st.text_input(
                "Ingrese f(x):",
                value="x**2 - 5*x - exp(x)",
                placeholder="Ej: x**2 - 5*x - exp(x)",
            )
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                x_prev = st.number_input(
                    "Punto inicial (x_{n-1}):", value=0.0, format="%.4f"
                )
            with col_s2:
                x_curr = st.number_input(
                    "Punto actual (x_n):", value=1.0, format="%.4f"
                )

        # 4. NEWTON-RAPHSON 1ER Y 2DO ORDEN
        elif metodo == "Newton-Raphson (1er Orden)":
            st.latex(r"x_{i+1} = x_i - \frac{f(x_i)}{f'(x_i)}")
            func_str = st.text_input(
                "Ingrese f(x):",
                value="x**2 - 5*x - exp(x)",
                placeholder="Ej: x**2 - 5*x - exp(x)",
            )
            x0 = st.number_input(
                "Valor inicial (x₀):", value=0.0, format="%.4f"
            )

        else:  # Newton-Raphson 2do Orden
            st.latex(
                r"x_{i+1} = x_i - \frac{f(x_i) \cdot f'(x_i)}{[f'(x_i)]^2 - f(x_i) \cdot f''(x_i)}"
            )
            func_str = st.text_input(
                "Ingrese f(x):",
                value="x**2 - 5*x - exp(x)",
                placeholder="Ej: x**2 - 5*x - exp(x)",
            )
            x0 = st.number_input(
                "Valor inicial (x₀):", value=0.0, format="%.4f"
            )

        # Parámetros comunes de convergencia
        col_tol, col_max = st.columns(2)
        with col_tol:
            tol = st.number_input(
                "Tolerancia de error (E):",
                value=0.001,
                format="%.5f",
                step=0.0001,
            )
        with col_max:
            max_iter = st.slider(
                "Máximo de iteraciones:",
                min_value=1,
                max_value=100,
                value=30,
            )

        btn_calcular = st.button("Ejecutar Método Numérico", type="primary")

    with col_der:
        st.subheader("3. Laboratorio Gráfico (GeoGebra)")
        st.iframe("https://www.geogebra.org/classic", height=380)

        if btn_calcular:
            st.markdown("---")
            try:
                res = None
                if metodo == "Método de Punto Fijo":
                    res = NoLinealesEngine.punto_fijo(
                        func_str, x0, tol, int(max_iter)
                    )
                elif metodo == "Método de Bisección":
                    res = NoLinealesEngine.biseccion(
                        func_str, val_a, val_b, tol, int(max_iter)
                    )
                elif metodo == "Método de Regula Falsi (Posición Falsa)":
                    res = NoLinealesEngine.regula_falsi(
                        func_str, val_a, val_b, tol, int(max_iter)
                    )
                elif metodo == "Método de la Secante":
                    res = NoLinealesEngine.secante(
                        func_str, x_prev, x_curr, tol, int(max_iter)
                    )
                elif metodo == "Newton-Raphson (1er Orden)":
                    res = NoLinealesEngine.newton_raphson_1(
                        func_str, x0, tol, int(max_iter)
                    )
                elif metodo == "Newton-Raphson (2do Orden)":
                    res = NoLinealesEngine.newton_raphson_2(
                        func_str, x0, tol, int(max_iter)
                    )

                if res and res["convergio"]:
                    st.success(f"**Raíz hallada:** `{res['raiz']:.6f}`")
                else:
                    st.warning("No se alcanzó la convergencia en el límite de iteraciones.")

                # if res and "iteraciones" in res:
                #     st.dataframe(res["iteraciones"], use_container_width=True)
                # Mostrar tabla de iteraciones
                if res and "iteraciones" in res:
                    st.subheader("Tabla de Iteraciones")
                    st.dataframe(res["iteraciones"], width="stretch")
                # Dentro de ui/tab_no_lineales.py después de calcular:
                pdf_bytes = PDFReportEngine.generar_reporte_pdf(
                    titulo_metodo=res["metodo"],
                    entradas_dict={"Función evaluada": func_str, "Tolerancia": tol},
                    resultado_str=f"Raíz = {res['raiz']}",
                    iteraciones_list=res.get("iteraciones", []),
                )

                st.download_button(
                    label="📄 Exportar Cálculo como PDF",
                    data=pdf_bytes,
                    file_name=f"Reporte_{res['metodo'].replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    key="btn_pdf_no_lineales",
                )
            except Exception as e:
                st.error(f"Error en el cálculo: {e}")