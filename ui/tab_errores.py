import streamlit as st
from engine.errores import ErroresEngine


def render_tab_errores():
    st.header("Módulo 1: Análisis de Errores")

    col_opt1, col_opt2 = st.columns([2, 1])
    with col_opt1:
        opcion_error = st.selectbox(
            "Seleccione el tipo de cálculo de error:",
            [
                "Error absoluto",
                "Error relativo",
                "Error porcentual",
                "Error de redondeo",
                "Error de truncamiento (Serie de Taylor)",
                "Error en funciones (Propagación)",
            ],
        )
    with col_opt2:
        num_dec = st.number_input(
            "Decimales de precisión para visualización:",
            min_value=1,
            max_value=12,
            value=6,
            step=1,
        )

    st.markdown("---")

    # 1. ERROR ABSOLUTO, RELATIVO Y PORCENTUAL
    if opcion_error in ["Error absoluto", "Error relativo", "Error porcentual"]:
        col1, col2 = st.columns(2)
        with col1:
            val_x = st.text_input(
                "Valor real (x):", value="1.206539823", placeholder="Ej: 1.206539823"
            )
        with col2:
            val_x_bar = st.text_input(
                "Valor aproximado (x̄):", value="1.206433", placeholder="Ej: 1.206433"
            )

        if val_x.strip() and val_x_bar.strip():
            try:
                x_num = float(val_x)
                x_bar_num = float(val_x_bar)

                if opcion_error == "Error absoluto":
                    st.latex(rf"|E_A| = |x - \bar{{x}}| = |{x_num} - {x_bar_num}|")
                    e_a = ErroresEngine.error_absoluto(x_num, x_bar_num)
                    st.success(f"**Error Absoluto (|E_A|):** `{e_a:.{num_dec}f}`")

                elif opcion_error == "Error relativo":
                    st.latex(
                        rf"|E_R| = \frac{{|E_A|}}{{|x|}} = \frac{{|{x_num} - {x_bar_num}|}}{{|{x_num}|}}"
                    )
                    e_r = ErroresEngine.error_relativo(x_num, x_bar_num)
                    st.success(f"**Error Relativo (|E_R|):** `{e_r:.{num_dec}f}`")

                elif opcion_error == "Error porcentual":
                    st.latex(
                        rf"|E_P| = |E_R| \cdot 100 = \frac{{|{x_num} - {x_bar_num}|}}{{|{x_num}|}} \cdot 100"
                    )
                    e_p = ErroresEngine.error_porcentual(x_num, x_bar_num)
                    st.success(f"**Error Porcentual (|E_P|):** `{e_p:.{num_dec}f}%`")

            except ValueError as e:
                st.error(
                    f"Error de entrada: Verifique que ambos valores sean numéricos. {e}"
                )

    # 2. ERROR DE REDONDEO
    elif opcion_error == "Error de redondeo":
        col1, col2, col3 = st.columns(3)
        with col1:
            val_x = st.text_input("Valor real (x):", value="3.14159265")
        with col2:
            val_x_bar = st.text_input("Valor aproximado (x̄):", value="3.1416")
        with col3:
            val_d = st.number_input(
                "Cifras significativas (d):", min_value=1, value=4, step=1
            )

        st.latex(rf"|E_R^d| = |x - \bar{{x}}| \le \frac{{1}}{{2}} \cdot 10^{{-d}}")

        cota = ErroresEngine.cota_redondeo(int(val_d))
        st.info(
            f"**Cota máxima de error teórica para d = {val_d}:** `{cota:.{num_dec}f}`"
        )

        if val_x.strip() and val_x_bar.strip():
            try:
                x_num = float(val_x)
                x_bar_num = float(val_x_bar)
                e_a = ErroresEngine.error_absoluto(x_num, x_bar_num)

                st.markdown(f"**Error absoluto real:** `{e_a:.{num_dec}f}`")
                if e_a <= cota:
                    st.success(
                        f"¡Cumple! El error absoluto ({e_a:.{num_dec}f}) es menor o igual a la cota teórica ({cota:.{num_dec}f})."
                    )
                else:
                    st.error(
                        f"No cumple. El error absoluto ({e_a:.{num_dec}f}) supera la cota teórica ({cota:.{num_dec}f})."
                    )
            except ValueError:
                st.error("Ingrese valores numéricos válidos.")

    # 3. ERROR DE TRUNCAMIENTO CON SERIE DE TAYLOR
    elif opcion_error == "Error de truncamiento (Serie de Taylor)":
        col1, col2 = st.columns(2)
        with col1:
            func_str = st.text_input(
                "Ingrese f(x):", value="exp(x)", placeholder="Ej: exp(x), sin(x)"
            )
            val_x0 = st.number_input("Punto de expansión (x₀):", value=0.0)
        with col2:
            val_x = st.number_input("Valor a evaluar (x):", value=0.5)
            val_d = st.number_input(
                "Orden / Términos de Taylor (d):", min_value=1, value=3, step=1
            )

        if st.button("Calcular Serie de Taylor"):
            try:
                res = ErroresEngine.error_truncamiento_taylor(
                    func_str, val_x0, val_x, int(val_d)
                )

                st.markdown("**1. Polinomio de Taylor resultante $P_d(x)$:**")
                st.latex(rf"P_{{{val_d}}}(x) = {res['polinomio_latex']}")

                st.markdown("**2. Evaluación y Comparación:**")
                st.latex(
                    rf"\text{{Valor Real }} f({val_x}) = {res['val_real']:.{num_dec}f}"
                )
                st.latex(
                    rf"\text{{Valor Aprox. }} P_{{{val_d}}}({val_x}) = {res['val_aprox']:.{num_dec}f}"
                )
                st.latex(
                    rf"|E_T^{{{val_d}}}| = |{res['val_real']:.{num_dec}f} - {res['val_aprox']:.{num_dec}f}| = {res['error_truncamiento']:.{num_dec}f}"
                )

                if res["cumple_cota"]:
                    st.success(
                        f"¡Cumple! El error de truncamiento ({res['error_truncamiento']:.{num_dec}f}) es menor o igual a la cota teórica ({res['cota_teorica']:.{num_dec}f})."
                    )
                else:
                    st.error(
                        f"No cumple. Supera la cota teórica ({res['cota_teorica']:.{num_dec}f}). Se requieren más términos."
                    )
            except Exception as e:
                st.error(f"Error al procesar la función: {e}")

    # 4. PROPAGACIÓN DE ERROR MULTIVARIABLE
    elif opcion_error == "Error en funciones (Propagación)":
        func_expr_str = st.text_input(
            "Ingrese la función f(x1, x2, ...):",
            value="3*x1*x2**3",
            placeholder="Ej: 3*x1*x2**3",
        )

        if func_expr_str.strip():
            try:
                import sympy as sp

                expr = sp.sympify(func_expr_str)
                variables = sorted(list(expr.free_symbols), key=lambda s: s.name)

                if not variables:
                    st.warning("Ingrese una función con variables (ej. x1, x2).")
                else:
                    st.markdown(
                        "**Variables detectadas:** "
                        + ", ".join([f"`{v.name}`" for v in variables])
                    )

                    cols = st.columns(len(variables))
                    valores_nominales = {}
                    errores_vars = {}

                    for idx, var in enumerate(variables):
                        with cols[idx]:
                            val_v = st.text_input(
                                f"Valor ({var.name}):",
                                value="2.0" if idx == 0 else "3.0",
                            )
                            err_v = st.text_input(
                                f"Error EA({var.name}):", value="0.01"
                            )
                            if val_v.strip() and err_v.strip():
                                valores_nominales[var.name] = float(val_v)
                                errores_vars[var.name] = float(err_v)

                    if len(valores_nominales) == len(variables):
                        res = ErroresEngine.propagacion_error(
                            func_expr_str, valores_nominales, errores_vars
                        )

                        st.markdown("---")
                        st.markdown("**1. Valoración de la función nominal:**")
                        st.latex(
                            rf"f = {sp.latex(expr)} = {res['valor_nominal']:.{num_dec}f}"
                        )

                        st.markdown("**2. Derivadas Parciales y Aportes de Error:**")
                        for d in res["detalles"]:
                            st.latex(
                                rf"\frac{{\partial f}}{{\partial {d['variable']}}} = {d['derivada_latex']} \quad \Rightarrow \quad |{d['valor_evaluado']:.{num_dec}f}| \cdot {d['error_variable']} = {d['aporte_error']:.{num_dec}f}"
                            )

                        st.markdown("**3. Resultado Final con Intervalo de Error:**")
                        st.success(
                            f"**Valor Final:** `{res['valor_nominal']:.{num_dec}f} ± {res['error_total']:.{num_dec}f}`"
                        )
            except Exception as e:
                st.error(f"Error en la evaluación simbólica: {e}")