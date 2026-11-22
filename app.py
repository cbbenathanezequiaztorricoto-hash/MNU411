import streamlit as st
from ui.tab_errores import render_tab_errores
from ui.tab_interpolacion import render_tab_interpolacion
from ui.tab_no_lineales import render_tab_no_lineales
from ui.tab_proyecciones import render_tab_proyecciones
from ui.tab_sistemas import render_tab_sistemas

# Configuración inicial de la página
st.set_page_config(
    page_title="MNU411 - Métodos Numéricos",
    page_icon="media/metodo26.gif",
    layout="wide",
)

# Título y Cabecera de la Aplicación
st.title("MNU411")
st.caption(
    "Plataforma interactiva para el análisis de errores, ecuaciones no lineales, "
    "sistemas lineales, interpolación funcional y proyecciones."
)

# Pestañas de navegación principal (Módulos 1 al 6)
tab_guia, tab_err, tab_nonlin, tab_sys, tab_interp, tab_proy = st.tabs(
    [
        "Guía de Símbolos",
        "Análisis de Errores",
        "Ecuaciones No Lineales",
        "Sistemas Lineales (A·x = b)",
        "Interpolación Funcional",
        "Proyecciones y Series de Tiempo",
    ]
)

with tab_guia:
    st.header("Guía de Sintaxis para Ingreso de Funciones")
    st.markdown(
        r"""
    | Categoría | Operación | Sintaxis en la App | Ejemplo Matemático |
    | :--- | :--- | :--- | :--- |
    | **Básicas** | Suma / Resta | `+` / `-` | `x + 5` |
    | | Multiplicación | `*` | `5*x` |
    | | División | `/` | `x / 2` |
    | | Potencia ($x^n$) | `**` | `x**2` para $x^2$ |
    | **Exponenciales y Raíces** | Exponencial ($e^x$) | `exp(x)` | `exp(x)` para $e^x$ |
    | | Raíz cuadrada ($\sqrt{x}$) | `sqrt(x)` | `sqrt(x)` para $\sqrt{x}$ |
    | **Trigonométricas** | Seno / Coseno / Tangente | `sin(x)`, `cos(x)`, `tan(x)` | `sin(x)` para $\sin(x)$ |
    | **Logarítmicas** | Logaritmo natural ($\ln x$) | `log(x)` | `log(x)` para $\ln(x)$ |
    """
    )

with tab_err:
    render_tab_errores()

with tab_nonlin:
    render_tab_no_lineales()

with tab_sys:
    render_tab_sistemas()

with tab_interp:
    render_tab_interpolacion()

with tab_proy:
    render_tab_proyecciones()
