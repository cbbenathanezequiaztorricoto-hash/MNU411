import sympy as sp
import numpy as np

class ErroresEngine:
    @staticmethod
    def error_absoluto(x: float, x_bar: float) -> float:
        """Calcula el Error Absoluto: |x - x̄|"""
        return abs(x - x_bar)

    @staticmethod
    def error_relativo(x: float, x_bar: float) -> float:
        """Calcula el Error Relativo: |x - x̄| / |x|"""
        if x == 0:
            raise ValueError("El valor real (x) no puede ser cero para el error relativo.")
        return abs(x - x_bar) / abs(x)

    @staticmethod
    def error_porcentual(x: float, x_bar: float) -> float:
        """Calcula el Error Porcentual: Error Relativo * 100"""
        return ErroresEngine.error_relativo(x, x_bar) * 100.0

    @staticmethod
    def cota_redondeo(d: int) -> float:
        """Calcula la Cota Máxima Teórica de Redondeo: (1/2) * 10^-d"""
        if d < 1:
            raise ValueError("Las cifras significativas (d) deben ser al menos 1.")
        return 0.5 * (10.0 ** -d)

    @staticmethod
    def error_truncamiento_taylor(func_str: str, x0: float, x_eval: float, orden: int):
        """
        Calcula el Polinomio de Taylor y el Error de Truncamiento para f(x).
        Retorna: (polinomio_latex, val_real, val_aprox, error_truncamiento, cota_teorica)
        """
        x_sym = sp.Symbol('x')
        expr = sp.sympify(func_str)
        
        # 1. Valor Real
        val_real = float(expr.subs(x_sym, x_eval))
        
        # 2. Polinomio de Taylor de orden 'd'
        taylor_poly = expr.series(x_sym, x0, n=orden + 1).removeO()
        val_aprox = float(taylor_poly.subs(x_sym, x_eval))
        
        # 3. Error y Cota Teórica
        e_t = abs(val_real - val_aprox)
        cota_teorica = 1.0 * (10.0 ** -orden)
        
        return {
            "polinomio_sympy": taylor_poly,
            "polinomio_latex": sp.latex(taylor_poly),
            "val_real": val_real,
            "val_aprox": val_aprox,
            "error_truncamiento": e_t,
            "cota_teorica": cota_teorica,
            "cumple_cota": e_t <= cota_teorica
        }

    @staticmethod
    def propagacion_error(func_str: str, valores_dict: dict, errores_dict: dict):
        """
        Calcula la propagación de error en funciones multivariable mediante Derivadas Parciales.
        Formula: E_A(total) = Suma( |∂f/∂xi| * E_A(xi) )
        """
        expr = sp.sympify(func_str)
        variables = sorted(list(expr.free_symbols), key=lambda s: s.name)
        
        # Mapeo de variables
        val_pares = [(v, valores_dict[v.name]) for v in variables]
        
        # Valor nominal de la función original
        y_0 = float(expr.subs(val_pares))
        
        e_a_total = 0.0
        detalles_derivadas = []
        
        for v in variables:
            df_var = sp.diff(expr, v)
            df_val = float(df_var.subs(val_pares))
            err_var = errores_dict[v.name]
            
            aporte = abs(df_val) * err_var
            e_a_total += aporte
            
            detalles_derivadas.append({
                "variable": v.name,
                "derivada_sympy": df_var,
                "derivada_latex": sp.latex(df_var),
                "valor_evaluado": df_val,
                "error_variable": err_var,
                "aporte_error": aporte
            })
            
        return {
            "valor_nominal": y_0,
            "error_total": e_a_total,
            "detalles": detalles_derivadas,
            "resultado_intervalo": f"{y_0} ± {e_a_total}"
        }