import numpy as np
import sympy as sp


def _validar_datos_regresion(x_vals, y_vals, min_puntos: int = 2):
    """
    Valida dimensiones, ausencia de NaN/Inf y dispersión mínima en X
    para evitar sistemas normales singulares.
    """
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        return False, "Los vectores X e Y deben ser unidimensionales y de igual longitud.", None, None

    if len(x) < min_puntos:
        return False, f"Se requieren al menos {min_puntos} pares de datos para este ajuste.", None, None

    if np.any(np.isnan(x)) or np.any(np.isnan(y)) or np.any(np.isinf(x)) or np.any(np.isinf(y)):
        return False, "Los datos contienen valores NaN o infinitos.", None, None

    if np.ptp(x) < 1e-15:
        return False, "Error de estabilidad: todos los valores de X son idénticos (varianza nula).", None, None

    return True, "OK", x, y


def _calcular_metricas_ajuste(y_real: np.ndarray, y_est: np.ndarray, num_parametros: int) -> dict:
    """
    Calcula St (suma total de cuadrados), Sr (suma de residuos al cuadrado),
    coeficiente de determinación R^2, correlación r y error estándar s_{y/x}
    en el espacio original de la variable dependiente Y.
    """
    n = len(y_real)
    y_prom = float(np.mean(y_real))
    st = float(np.sum((y_real - y_prom) ** 2))
    sr = float(np.sum((y_real - y_est) ** 2))

    if st < 1e-15:
        r2 = 1.0 if sr < 1e-15 else 0.0
    else:
        r2 = float(1.0 - (sr / st))

    r_pearson = float(np.sqrt(max(0.0, min(1.0, r2))))

    grados_libertad = n - num_parametros
    if grados_libertad > 0:
        error_estandar = float(np.sqrt(sr / grados_libertad))
    else:
        error_estandar = float("nan")

    return {
        "st": st,
        "sr": sr,
        "r2": r2,
        "r": r_pearson,
        "error_estandar": error_estandar,
        "grados_libertad": grados_libertad,
    }


def _verificar_extrapolacion(x: np.ndarray, x_eval: float) -> tuple[bool, tuple[float, float]]:
    x_min, x_max = float(np.min(x)), float(np.max(x))
    return bool(x_eval < x_min or x_eval > x_max), (x_min, x_max)


# =============================================================================
# PIEZA 6.1.A: REGRESIÓN LINEAL SIMPLE (MÍNIMOS CUADRADOS)
# =============================================================================
def regresion_lineal_simple(x_vals, y_vals, x_eval: float) -> dict:
    """
    Ajusta el modelo y = a_0 + a_1 * x mediante ecuaciones normales
    (Nieves & Domínguez, Sec. 6.1).
    """
    valido, msg, x, y = _validar_datos_regresion(x_vals, y_vals, min_puntos=2)
    if not valido:
        return {"exito": False, "mensaje": msg}

    n = len(x)
    sum_x = float(np.sum(x))
    sum_y = float(np.sum(y))
    sum_x2 = float(np.sum(x ** 2))
    sum_xy = float(np.sum(x * y))
    sum_y2 = float(np.sum(y ** 2))

    denom = n * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-15:
        return {"exito": False, "mensaje": "Sistema normal singular (determinante nulo)."}

    a1 = (n * sum_xy - sum_x * sum_y) / denom
    a0 = (sum_y - a1 * sum_x) / n

    y_est = a0 + a1 * x
    metricas = _calcular_metricas_ajuste(y, y_est, num_parametros=2)
    if a1 < 0:
        metricas["r"] = -metricas["r"]

    y_proj = float(a0 + a1 * x_eval)
    es_extrap, rango_x = _verificar_extrapolacion(x, x_eval)

    signo = "+" if a1 >= 0 else "-"
    latex_ec = f"y = {a0:.6g} {signo} {abs(a1):.6g}x"
    ggb_ec = f"{a0:.10g} + ({a1:.10g})*x"

    return {
        "exito": True,
        "mensaje": "Regresión lineal calculada correctamente.",
        "metodo": "Regresión Lineal Simple (Pieza 6.1.A)",
        "x": x.tolist(),
        "y": y.tolist(),
        "y_estimado": y_est.tolist(),
        "x_eval": float(x_eval),
        "resultado": y_proj,
        "parametros": {"a0": float(a0), "a1": float(a1)},
        "sumatorias": {
            "n": n,
            "sum_x": sum_x,
            "sum_y": sum_y,
            "sum_x2": sum_x2,
            "sum_xy": sum_xy,
            "sum_y2": sum_y2,
        },
        "matriz_normal": [[float(n), sum_x], [sum_x, sum_x2]],
        "vector_normal": [sum_y, sum_xy],
        "metricas": metricas,
        "ecuacion_latex": latex_ec,
        "funcion_geogebra": ggb_ec,
        "es_extrapolacion": es_extrap,
        "rango_x": rango_x,
    }


# =============================================================================
# PIEZA 6.2.B: REGRESIÓN POLINOMIAL DE GRADO m
# =============================================================================
def regresion_polinomial(x_vals, y_vals, grado: int, x_eval: float) -> dict:
    """
    Ajusta un polinomio P_m(x) = a_0 + a_1*x + ... + a_m*x^m por mínimos cuadrados
    resolviendo el sistema normal [A]{a} = {B} (Nieves & Domínguez, Sec. 6.2).
    """
    if grado < 1:
        return {"exito": False, "mensaje": "El grado del polinomio debe ser mayor o igual a 1."}

    valido, msg, x, y = _validar_datos_regresion(x_vals, y_vals, min_puntos=grado + 1)
    if not valido:
        return {"exito": False, "mensaje": msg}

    n = len(x)
    m = int(grado)
    dim = m + 1

    # Construcción de la matriz normal simétrica A (dim x dim) y vector B
    A = np.zeros((dim, dim), dtype=float)
    B = np.zeros(dim, dtype=float)

    potencias_x = {k: float(np.sum(x ** k)) for k in range(2 * m + 1)}
    for j in range(dim):
        for k in range(dim):
            A[j, k] = potencias_x[j + k]
        B[j] = float(np.sum((x ** j) * y))

    cond = float(np.linalg.cond(A))
    if cond > 1e14:
        return {
            "exito": False,
            "mensaje": f"Matriz normal mal condicionada o singular (cond = {cond:.2e}). Reduzca el grado.",
        }

    coeficientes = np.linalg.solve(A, B)

    # Evaluación mediante esquema de Horner (Sec. 2.10)
    def evaluar_horner(val):
        acc = coeficientes[-1]
        for c in range(m - 1, -1, -1):
            acc = acc * val + coeficientes[c]
        return acc

    y_est = np.array([evaluar_horner(xi) for xi in x], dtype=float)
    y_proj = float(evaluar_horner(x_eval))
    metricas = _calcular_metricas_ajuste(y, y_est, num_parametros=dim)
    es_extrap, rango_x = _verificar_extrapolacion(x, x_eval)

    # Construcción simbólica con SymPy
    x_sym = sp.Symbol("x")
    expr = sum(sp.Float(coeficientes[k], 8) * (x_sym ** k) for k in range(dim))
    latex_ec = f"y = {sp.latex(expr)}"
    ggb_ec = " + ".join(f"({coeficientes[k]:.10g})*x^{k}" for k in range(dim))

    advertencia = ""
    if m == n - 1:
        advertencia = "Grado m = n - 1: el ajuste degenera en interpolación exacta (0 grados de libertad)."
    elif cond > 1e9:
        advertencia = f"Advertencia de condicionamiento elevado en matriz normal (cond = {cond:.2e})."

    return {
        "exito": True,
        "mensaje": "Regresión polinomial calculada correctamente.",
        "metodo": f"Regresión Polinomial de Grado {m} (Pieza 6.2.B)",
        "grado": m,
        "x": x.tolist(),
        "y": y.tolist(),
        "y_estimado": y_est.tolist(),
        "x_eval": float(x_eval),
        "resultado": y_proj,
        "coeficientes": [float(c) for c in coeficientes],
        "matriz_normal": A.tolist(),
        "vector_normal": B.tolist(),
        "numero_condicion": cond,
        "metricas": metricas,
        "ecuacion_latex": latex_ec,
        "funcion_geogebra": ggb_ec,
        "es_extrapolacion": es_extrap,
        "rango_x": rango_x,
        "advertencia": advertencia,
    }


# =============================================================================
# PIEZA 6.3.C: REGRESIÓN EXPONENCIAL (y = a * e^(b*x))
# =============================================================================
def regresion_exponencial(x_vals, y_vals, x_eval: float) -> dict:
    """
    Ajusta el modelo no lineal y = a * exp(b * x) mediante transformación
    logarítmica ln(y) = ln(a) + b*x (Nieves & Domínguez, Sec. 6.3).
    """
    valido, msg, x, y = _validar_datos_regresion(x_vals, y_vals, min_puntos=2)
    if not valido:
        return {"exito": False, "mensaje": msg}

    if np.any(y <= 0):
        return {
            "exito": False,
            "mensaje": "Error de dominio: el modelo exponencial y = a·e^(bx) requiere que todos los valores de Y sean estrictamente positivos (y_i > 0).",
        }

    z = np.log(y)
    lin = regresion_lineal_simple(x, z, x_eval)
    if not lin["exito"]:
        return {"exito": False, "mensaje": lin["mensaje"]}

    a0 = lin["parametros"]["a0"]
    b = lin["parametros"]["a1"]
    a = float(np.exp(a0))

    y_est = a * np.exp(b * x)
    y_proj = float(a * np.exp(b * x_eval))
    metricas = _calcular_metricas_ajuste(y, y_est, num_parametros=2)
    metricas["r2_linealizado"] = lin["metricas"]["r2"]
    es_extrap, rango_x = _verificar_extrapolacion(x, x_eval)

    latex_ec = f"y = {a:.6g} e^{{{b:.6g}x}}"
    ggb_ec = f"({a:.10g}) * exp(({b:.10g}) * x)"

    return {
        "exito": True,
        "mensaje": "Regresión exponencial calculada correctamente.",
        "metodo": "Regresión Exponencial (Pieza 6.3.C)",
        "x": x.tolist(),
        "y": y.tolist(),
        "z_transformado": z.tolist(),
        "y_estimado": y_est.tolist(),
        "x_eval": float(x_eval),
        "resultado": y_proj,
        "parametros": {"a": a, "b": float(b), "ln_a": float(a0)},
        "metricas": metricas,
        "ecuacion_latex": latex_ec,
        "funcion_geogebra": ggb_ec,
        "es_extrapolacion": es_extrap,
        "rango_x": rango_x,
    }


# =============================================================================
# PIEZA 6.4.D: REGRESIÓN POTENCIAL (y = a * x^b)
# =============================================================================
def regresion_potencial(x_vals, y_vals, x_eval: float) -> dict:
    """
    Ajusta el modelo potencial y = a * x^b mediante doble transformación
    logarítmica ln(y) = ln(a) + b*ln(x) (Nieves & Domínguez, Sec. 6.3).
    """
    valido, msg, x, y = _validar_datos_regresion(x_vals, y_vals, min_puntos=2)
    if not valido:
        return {"exito": False, "mensaje": msg}

    if np.any(x <= 0) or np.any(y <= 0):
        return {
            "exito": False,
            "mensaje": "Error de dominio: el modelo potencial y = a·x^b requiere X > 0 e Y > 0 en todos los puntos.",
        }

    if x_eval <= 0:
        return {
            "exito": False,
            "mensaje": "Error de dominio: x_eval debe ser estrictamente positivo (x_eval > 0) para evaluar y = a·x^b.",
        }

    w = np.log(x)
    z = np.log(y)
    lin = regresion_lineal_simple(w, z, float(np.log(x_eval)))
    if not lin["exito"]:
        return {"exito": False, "mensaje": lin["mensaje"]}

    a0 = lin["parametros"]["a0"]
    b = lin["parametros"]["a1"]
    a = float(np.exp(a0))

    y_est = a * (x ** b)
    y_proj = float(a * (x_eval ** b))
    metricas = _calcular_metricas_ajuste(y, y_est, num_parametros=2)
    metricas["r2_linealizado"] = lin["metricas"]["r2"]
    es_extrap, rango_x = _verificar_extrapolacion(x, x_eval)

    latex_ec = f"y = {a:.6g} x^{{{b:.6g}}}"
    ggb_ec = f"({a:.10g}) * x^({b:.10g})"

    return {
        "exito": True,
        "mensaje": "Regresión potencial calculada correctamente.",
        "metodo": "Regresión Potencial (Pieza 6.4.D)",
        "x": x.tolist(),
        "y": y.tolist(),
        "w_transformado": w.tolist(),
        "z_transformado": z.tolist(),
        "y_estimado": y_est.tolist(),
        "x_eval": float(x_eval),
        "resultado": y_proj,
        "parametros": {"a": a, "b": float(b), "ln_a": float(a0)},
        "metricas": metricas,
        "ecuacion_latex": latex_ec,
        "funcion_geogebra": ggb_ec,
        "es_extrapolacion": es_extrap,
        "rango_x": rango_x,
    }


# =============================================================================
# PIEZA 6.5.E: REGRESIÓN LOGARÍTMICA (y = a + b*ln(x)) Y COMPARADOR DE MODELOS
# =============================================================================
def regresion_logaritmica(x_vals, y_vals, x_eval: float) -> dict:
    """
    Ajusta el modelo logarítmico y = a + b * ln(x) mediante sustitución w = ln(x).
    """
    valido, msg, x, y = _validar_datos_regresion(x_vals, y_vals, min_puntos=2)
    if not valido:
        return {"exito": False, "mensaje": msg}

    if np.any(x <= 0):
        return {
            "exito": False,
            "mensaje": "Error de dominio: el modelo logarítmico y = a + b·ln(x) requiere X > 0 en todos los puntos.",
        }

    if x_eval <= 0:
        return {
            "exito": False,
            "mensaje": "Error de dominio: x_eval debe ser estrictamente positivo (x_eval > 0).",
        }

    w = np.log(x)
    lin = regresion_lineal_simple(w, y, float(np.log(x_eval)))
    if not lin["exito"]:
        return {"exito": False, "mensaje": lin["mensaje"]}

    a = lin["parametros"]["a0"]
    b = lin["parametros"]["a1"]

    y_est = a + b * np.log(x)
    y_proj = float(a + b * np.log(x_eval))
    metricas = _calcular_metricas_ajuste(y, y_est, num_parametros=2)
    es_extrap, rango_x = _verificar_extrapolacion(x, x_eval)

    signo = "+" if b >= 0 else "-"
    latex_ec = f"y = {a:.6g} {signo} {abs(b):.6g} \\ln(x)"
    ggb_ec = f"({a:.10g}) + ({b:.10g}) * ln(x)"

    return {
        "exito": True,
        "mensaje": "Regresión logarítmica calculada correctamente.",
        "metodo": "Regresión Logarítmica (Pieza 6.5.E)",
        "x": x.tolist(),
        "y": y.tolist(),
        "w_transformado": w.tolist(),
        "y_estimado": y_est.tolist(),
        "x_eval": float(x_eval),
        "resultado": y_proj,
        "parametros": {"a": float(a), "b": float(b)},
        "metricas": metricas,
        "ecuacion_latex": latex_ec,
        "funcion_geogebra": ggb_ec,
        "es_extrapolacion": es_extrap,
        "rango_x": rango_x,
    }


def comparar_modelos_proyeccion(x_vals, y_vals, x_eval: float, grado_poli: int = 2) -> dict:
    """
    Ejecuta todos los modelos aplicables (6.1.A a 6.5.E), ordena los válidos por
    coeficiente de determinación R^2 descendente y selecciona el mejor modelo de proyección.
    """
    candidatos = [
        ("6.1.A Lineal", regresion_lineal_simple(x_vals, y_vals, x_eval)),
        (f"6.2.B Polinomial (g={grado_poli})", regresion_polinomial(x_vals, y_vals, grado_poli, x_eval)),
        ("6.3.C Exponencial", regresion_exponencial(x_vals, y_vals, x_eval)),
        ("6.4.D Potencial", regresion_potencial(x_vals, y_vals, x_eval)),
        ("6.5.E Logarítmico", regresion_logaritmica(x_vals, y_vals, x_eval)),
    ]

    validos = []
    descartados = []

    for nombre, res in candidatos:
        if res.get("exito"):
            validos.append(
                {
                    "modelo": nombre,
                    "ecuacion_latex": res["ecuacion_latex"],
                    "funcion_geogebra": res["funcion_geogebra"],
                    "proyeccion": res["resultado"],
                    "r2": res["metricas"]["r2"],
                    "error_estandar": res["metricas"]["error_estandar"],
                    "sr": res["metricas"]["sr"],
                    "detalle": res,
                }
            )
        else:
            descartados.append({"modelo": nombre, "motivo": res.get("mensaje", "No aplicable")})

    if not validos:
        return {
            "exito": False,
            "mensaje": "Ningún modelo pudo ajustarse a los datos proporcionados.",
            "descartados": descartados,
        }

    # Ordenar por R^2 descendente (y menor Sr en caso de empate)
    validos.sort(key=lambda item: (-item["r2"], item["sr"]))

    return {
        "exito": True,
        "mensaje": "Comparativa de modelos completada.",
        "mejor_modelo": validos[0],
        "ranking": validos,
        "descartados": descartados,
    }
# =============================================================================
# PIEZA 6.6.F: ANÁLISIS Y PRONÓSTICO DE SERIES DE TIEMPO
# =============================================================================
def _metricas_series_tiempo(y_real: np.ndarray, y_pron: np.ndarray) -> dict:
    """
    Calcula métricas estándar de error en series de tiempo sobre los periodos
    donde existe pronóstico válido (no NaN): MAD, MSE, RMSE y MAPE (%).
    """
    mask = ~np.isnan(y_pron)
    yr = y_real[mask]
    yp = y_pron[mask]

    if len(yr) == 0:
        return {"mad": float("nan"), "mse": float("nan"), "rmse": float("nan"), "mape": float("nan")}

    errores = yr - yp
    abs_err = np.abs(errores)
    mad = float(np.mean(abs_err))
    mse = float(np.mean(errores ** 2))
    rmse = float(np.sqrt(mse))

    # MAPE evitando división por cero si algún y_t == 0
    mask_no_cero = np.abs(yr) > 1e-12
    if np.any(mask_no_cero):
        mape = float(np.mean(abs_err[mask_no_cero] / np.abs(yr[mask_no_cero])) * 100.0)
    else:
        mape = float("nan")

    return {"mad": mad, "mse": mse, "rmse": rmse, "mape": mape}


def pronostico_series_tiempo(
    y_vals,
    submetodo: str = "SES",
    ventana_k: int = 3,
    pesos: list | None = None,
    alpha: float = 0.3,
    beta: float = 0.1,
    horizonte_h: int = 1,
    periodo_estacional: int = 4,
    t_vals=None,
) -> dict:
    """
    Ejecuta pronósticos de series de tiempo univariadas y_1, y_2, ..., y_n:
      - 'PMS': Promedio Móvil Simple de orden k
      - 'PMP': Promedio Móvil Ponderado de orden k (pesos suman 1)
      - 'SES': Suavizamiento Exponencial Simple (alpha)
      - 'HOLT': Suavizamiento Exponencial Doble de Holt (alpha, beta)
      - 'ESTACIONAL': Descomposición Multiplicativa (Tendencia Lineal + Índices Estacionales)
    """
    y = np.asarray(y_vals, dtype=float)
    if y.ndim != 1 or len(y) < 3:
        return {"exito": False, "mensaje": "La serie de tiempo requiere al menos 3 observaciones."}
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        return {"exito": False, "mensaje": "La serie contiene valores NaN o infinitos."}

    n = len(y)
    t = np.arange(1, n + 1, dtype=float) if t_vals is None else np.asarray(t_vals, dtype=float)
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
    t_futuro = np.array([t[-1] + (h * dt) for h in range(1, horizonte_h + 1)], dtype=float)

    y_ajuste = np.full(n, np.nan, dtype=float)
    pronosticos_futuros = np.zeros(horizonte_h, dtype=float)
    detalles_extra = {}

    # 1. PROMEDIO MÓVIL SIMPLE (PMS)
    if submetodo == "PMS":
        k = int(ventana_k)
        if k < 1 or k >= n:
            return {"exito": False, "mensaje": f"La ventana k debe cumplir 1 <= k < {n}."}
        for i in range(k, n):
            y_ajuste[i] = np.mean(y[i - k : i])
        # Pronóstico futuro iterativo (o nivel constante del último bloque)
        serie_ext = list(y)
        for h in range(horizonte_h):
            val_next = float(np.mean(serie_ext[-k:]))
            pronosticos_futuros[h] = val_next
            serie_ext.append(val_next)
        nombre_metodo = f"Promedio Móvil Simple (k = {k})"

    # 2. PROMEDIO MÓVIL PONDERADO (PMP)
    elif submetodo == "PMP":
        if not pesos or len(pesos) < 2:
            return {"exito": False, "mensaje": "Debe proporcionar al menos 2 pesos para el PMP."}
        w = np.asarray(pesos, dtype=float)
        if np.any(w < 0) or np.sum(w) <= 0:
            return {"exito": False, "mensaje": "Los pesos deben ser no negativos y sumar > 0."}
        w = w / np.sum(w)  # Normalización automática a suma = 1
        k = len(w)
        if k >= n:
            return {"exito": False, "mensaje": f"El número de pesos ({k}) debe ser menor que n ({n})."}
        for i in range(k, n):
            y_ajuste[i] = float(np.dot(y[i - k : i], w))
        serie_ext = list(y)
        for h in range(horizonte_h):
            val_next = float(np.dot(serie_ext[-k:], w))
            pronosticos_futuros[h] = val_next
            serie_ext.append(val_next)
        detalles_extra["pesos_normalizados"] = w.tolist()
        nombre_metodo = f"Promedio Móvil Ponderado (k = {k})"

    # 3. SUAVIZAMIENTO EXPONENCIAL SIMPLE (SES)
    elif submetodo == "SES":
        if not (0.0 < alpha <= 1.0):
            return {"exito": False, "mensaje": "El parámetro alpha debe estar en (0, 1]."}
        y_ajuste[0] = y[0]
        for i in range(1, n):
            y_ajuste[i] = alpha * y[i - 1] + (1.0 - alpha) * y_ajuste[i - 1]
        ultimo_pron = float(alpha * y[-1] + (1.0 - alpha) * y_ajuste[-1])
        pronosticos_futuros[:] = ultimo_pron
        nombre_metodo = f"Suavizamiento Exponencial Simple (α = {alpha:.3f})"

    # 4. SUAVIZAMIENTO EXPONENCIAL DOBLE DE HOLT (NIVEL + TENDENCIA)
    elif submetodo == "HOLT":
        if not (0.0 < alpha <= 1.0) or not (0.0 <= beta <= 1.0):
            return {"exito": False, "mensaje": "Los parámetros alpha y beta deben estar en (0, 1]."}
        L = np.zeros(n, dtype=float)
        T = np.zeros(n, dtype=float)
        L[0] = y[0]
        T[0] = y[1] - y[0]
        y_ajuste[0] = y[0]
        for i in range(1, n):
            y_ajuste[i] = L[i - 1] + T[i - 1]
            L[i] = alpha * y[i] + (1.0 - alpha) * (L[i - 1] + T[i - 1])
            T[i] = beta * (L[i] - L[i - 1]) + (1.0 - beta) * T[i - 1]
        for h in range(1, horizonte_h + 1):
            pronosticos_futuros[h - 1] = L[-1] + h * T[-1]
        detalles_extra["nivel_final"] = float(L[-1])
        detalles_extra["tendencia_final"] = float(T[-1])
        nombre_metodo = f"Suavizamiento Doble de Holt (α = {alpha:.2f}, β = {beta:.2f})"

    # 5. DESCOMPOSICIÓN ESTACIONAL MULTIPLICATIVA (TENDENCIA + ÍNDICES ESTACIONALES)
    elif submetodo == "ESTACIONAL":
        L_per = int(periodo_estacional)
        if L_per < 2 or n < 2 * L_per:
            return {
                "exito": False,
                "mensaje": f"Se requieren al menos 2 ciclos completos (n >= {2 * L_per}) para periodo L = {L_per}.",
            }
        if np.any(y <= 0):
            return {
                "exito": False,
                "mensaje": "La descomposición multiplicativa requiere valores positivos (y_t > 0).",
            }
        # Tendencia lineal base sobre índice temporal 1..n
        idx_t = np.arange(1, n + 1, dtype=float)
        tend_lin = regresion_lineal_simple(idx_t, y, float(n + 1))
        a0 = tend_lin["parametros"]["a0"]
        a1 = tend_lin["parametros"]["a1"]
        tendencia_hist = a0 + a1 * idx_t

        # Razón y_t / Tendencia_t agrupada por estación (0 .. L_per - 1)
        razones = y / tendencia_hist
        indices_brutos = np.zeros(L_per, dtype=float)
        for s in range(L_per):
            indices_brutos[s] = float(np.mean(razones[s::L_per]))
        # Normalizar para que los índices sumen exactamente L_per
        indices_est = indices_brutos * (L_per / np.sum(indices_brutos))

        for i in range(n):
            y_ajuste[i] = tendencia_hist[i] * indices_est[i % L_per]

        for h in range(1, horizonte_h + 1):
            t_idx_fut = n + h
            est_idx = (t_idx_fut - 1) % L_per
            pronosticos_futuros[h - 1] = (a0 + a1 * t_idx_fut) * indices_est[est_idx]

        detalles_extra["indices_estacionales"] = [round(float(val), 4) for val in indices_est]
        detalles_extra["tendencia_base"] = f"T(t) = {a0:.4f} + {a1:.4f}·t"
        nombre_metodo = f"Descomposición Estacional Multiplicativa (L = {L_per})"

    else:
        return {"exito": False, "mensaje": f"Submétodo de serie de tiempo no reconocido: {submetodo}"}

    metricas = _metricas_series_tiempo(y, y_ajuste)

    # Polinomio segmentario para visualizar la curva suavizada/pronosticada en GeoGebra
    t_comb = np.concatenate([t, t_futuro])
    y_comb = np.concatenate([np.where(np.isnan(y_ajuste), y, y_ajuste), pronosticos_futuros])
    tramos_ggb = []
    for k in range(len(t_comb) - 1):
        m_k = (y_comb[k + 1] - y_comb[k]) / (t_comb[k + 1] - t_comb[k])
        b_k = y_comb[k] - m_k * t_comb[k]
        tramos_ggb.append((t_comb[k + 1], f"({m_k:.8g}*x + ({b_k:.8g}))"))

    ggb_expr = tramos_ggb[-1][1]
    for k in range(len(tramos_ggb) - 2, -1, -1):
        ggb_expr = f"If(x <= {tramos_ggb[k][0]}, {tramos_ggb[k][1]}, {ggb_expr})"

    return {
        "exito": True,
        "mensaje": "Pronóstico de serie de tiempo calculado correctamente.",
        "metodo": f"Series de Tiempo — {nombre_metodo} (Pieza 6.6.F)",
        "t": t.tolist(),
        "y": y.tolist(),
        "y_ajuste": y_ajuste.tolist(),
        "t_futuro": t_futuro.tolist(),
        "pronosticos_futuros": pronosticos_futuros.tolist(),
        "resultado": float(pronosticos_futuros[0]),
        "x_eval": float(t_futuro[0]),
        "metricas": metricas,
        "detalles_extra": detalles_extra,
        "funcion_geogebra": ggb_expr,
    }
