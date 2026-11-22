import math
import sympy as sp


class NoLinealesEngine:

    @staticmethod
    def _evaluar(expr, x_sym, valor: float) -> float:
        """Función auxiliar para evaluar expresiones de SymPy a float de forma segura."""
        res = expr.evalf(subs={x_sym: valor})
        return float(res)

    # -------------------------------------------------------------------------
    # 2.1 MÉTODO DE PUNTO FIJO: x_{i+1} = g(x_i)
    # -------------------------------------------------------------------------
    @staticmethod
    def punto_fijo(
        func_g_str: str, x0: float, tol: float = 0.001, max_iter: int = 30
    ):
        x_sym = sp.Symbol("x")
        g_expr = sp.sympify(func_g_str)

        iteraciones = []
        x_curr = x0
        convergio = False

        for i in range(max_iter):
            x_next = NoLinealesEngine._evaluar(g_expr, x_sym, x_curr)
            error = abs(x_next - x_curr)

            iteraciones.append(
                {
                    "Iteracion": i,
                    "x_i": x_curr,
                    "x_i+1": x_next,
                    "Error": error,
                }
            )

            if error <= tol:
                convergio = True
                break
            x_curr = x_next

        return {
            "metodo": "Punto Fijo",
            "g_latex": sp.latex(g_expr),
            "raiz": x_curr if convergio else None,
            "convergio": convergio,
            "iteraciones": iteraciones,
        }

    # -------------------------------------------------------------------------
    # 2.2 MÉTODO DE BISECCIÓN
    # -------------------------------------------------------------------------
    @staticmethod
    def biseccion(
        func_f_str: str,
        a: float,
        b: float,
        tol: float = 0.001,
        max_iter: int = 30,
    ):
        x_sym = sp.Symbol("x")
        f_expr = sp.sympify(func_f_str)

        fa = NoLinealesEngine._evaluar(f_expr, x_sym, a)
        fb = NoLinealesEngine._evaluar(f_expr, x_sym, b)

        if fa * fb > 0:
            raise ValueError(
                f"El intervalo [{a}, {b}] no presenta cambio de signo: f(a)={fa:.4f}, f(b)={fb:.4f}"
            )

        # Estimación teórica de iteraciones: n = (ln(b - a) - ln(tol)) / ln(2)
        n_teorico = math.ceil((math.log(abs(b - a)) - math.log(tol)) / math.log(2))

        iteraciones = []
        x_m_prev = a
        convergio = False

        for i in range(1, max_iter + 1):
            x_m = (a + b) / 2.0
            f_xm = NoLinealesEngine._evaluar(f_expr, x_sym, x_m)
            error = abs(b - a) if i == 1 else abs(x_m - x_m_prev)

            iteraciones.append(
                {
                    "Iteracion": i,
                    "a": a,
                    "b": b,
                    "x_M": x_m,
                    "f(x_M)": f_xm,
                    "Error": error,
                }
            )

            if error <= tol or abs(f_xm) == 0:
                convergio = True
                break

            fa = NoLinealesEngine._evaluar(f_expr, x_sym, a)
            if fa * f_xm < 0:
                b = x_m
            else:
                a = x_m

            x_m_prev = x_m

        return {
            "metodo": "Bisección",
            "f_latex": sp.latex(f_expr),
            "n_teorico": n_teorico,
            "raiz": iteraciones[-1]["x_M"] if convergio else None,
            "convergio": convergio,
            "iteraciones": iteraciones,
        }

    # -------------------------------------------------------------------------
    # 2.3 MÉTODO DE REGULA FALSI (POSICIÓN FALSA)
    # -------------------------------------------------------------------------
    @staticmethod
    def regula_falsi(
        func_f_str: str,
        x0: float,
        x1: float,
        tol: float = 0.001,
        max_iter: int = 30,
    ):
        x_sym = sp.Symbol("x")
        f_expr = sp.sympify(func_f_str)

        a, b = x0, x1
        fa = NoLinealesEngine._evaluar(f_expr, x_sym, a)
        fb = NoLinealesEngine._evaluar(f_expr, x_sym, b)

        if fa * fb > 0:
            raise ValueError(
                f"El intervalo [{a}, {b}] no presenta cambio de signo."
            )

        iteraciones = []
        x_m_prev = a
        convergio = False

        for i in range(1, max_iter + 1):
            fa = NoLinealesEngine._evaluar(f_expr, x_sym, a)
            fb = NoLinealesEngine._evaluar(f_expr, x_sym, b)

            denominador = fb - fa
            if denominador == 0:
                raise ZeroDivisionError("División por cero en Regula Falsi.")

            x_m = b - (fb * (b - a)) / denominador
            f_xm = NoLinealesEngine._evaluar(f_expr, x_sym, x_m)
            error = abs(x_m - x_m_prev) if i > 1 else abs(b - a)

            iteraciones.append(
                {
                    "Iteracion": i,
                    "x0 (a)": a,
                    "x1 (b)": b,
                    "x_M": x_m,
                    "f(x_M)": f_xm,
                    "Error": error,
                }
            )

            if error <= tol or abs(f_xm) == 0:
                convergio = True
                break

            if fa * f_xm < 0:
                b = x_m
            else:
                a = x_m

            x_m_prev = x_m

        return {
            "metodo": "Regula Falsi",
            "f_latex": sp.latex(f_expr),
            "raiz": iteraciones[-1]["x_M"] if convergio else None,
            "convergio": convergio,
            "iteraciones": iteraciones,
        }

    # -------------------------------------------------------------------------
    # 2.4 MÉSTATIC DE LA SECANTE
    # -------------------------------------------------------------------------
    @staticmethod
    def secante(
        func_f_str: str,
        x_prev: float,
        x_curr: float,
        tol: float = 0.001,
        max_iter: int = 30,
    ):
        x_sym = sp.Symbol("x")
        f_expr = sp.sympify(func_f_str)

        iteraciones = []
        convergio = False

        for i in range(1, max_iter + 1):
            f_prev = NoLinealesEngine._evaluar(f_expr, x_sym, x_prev)
            f_curr = NoLinealesEngine._evaluar(f_expr, x_sym, x_curr)

            denominador = f_curr - f_prev
            if denominador == 0:
                raise ZeroDivisionError(
                    "División por cero en Secante: f(x_n) - f(x_{n-1}) = 0"
                )

            x_next = x_curr - (f_curr * (x_curr - x_prev)) / denominador
            error = abs(x_next - x_curr)

            iteraciones.append(
                {
                    "Iteracion": i,
                    "x_{n-1}": x_prev,
                    "x_n": x_curr,
                    "f(x_n)": f_curr,
                    "x_{n+1}": x_next,
                    "Error": error,
                }
            )

            if error <= tol:
                convergio = True
                break

            x_prev = x_curr
            x_curr = x_next

        return {
            "metodo": "Secante",
            "f_latex": sp.latex(f_expr),
            "raiz": x_curr if convergio else None,
            "convergio": convergio,
            "iteraciones": iteraciones,
        }

    # -------------------------------------------------------------------------
    # 2.5 NEWTON-RAPHSON 1ER Y 2DO ORDEN (Derivación simbólica implícita)
    # -------------------------------------------------------------------------
    @staticmethod
    def newton_raphson_1(
        func_f_str: str, x0: float, tol: float = 0.001, max_iter: int = 30
    ):
        x_sym = sp.Symbol("x")
        f_expr = sp.sympify(func_f_str)
        f_d1_expr = sp.diff(f_expr, x_sym)

        iteraciones = []
        x_curr = x0
        convergio = False

        for i in range(1, max_iter + 1):
            f_val = NoLinealesEngine._evaluar(f_expr, x_sym, x_curr)
            fd1_val = NoLinealesEngine._evaluar(f_d1_expr, x_sym, x_curr)

            if fd1_val == 0:
                raise ZeroDivisionError(
                    "La primera derivada se hizo cero. Indeterminación."
                )

            x_next = x_curr - (f_val / fd1_val)
            error = abs(x_next - x_curr)

            iteraciones.append(
                {
                    "Iteracion": i,
                    "x_i": x_curr,
                    "f(x_i)": f_val,
                    "f'(x_i)": fd1_val,
                    "x_{i+1}": x_next,
                    "Error": error,
                }
            )

            if error <= tol:
                convergio = True
                break

            x_curr = x_next

        return {
            "metodo": "Newton-Raphson (1er Orden)",
            "f_latex": sp.latex(f_expr),
            "fd1_latex": sp.latex(f_d1_expr),
            "raiz": x_curr if convergio else None,
            "convergio": convergio,
            "iteraciones": iteraciones,
        }

    @staticmethod
    def newton_raphson_2(
        func_f_str: str, x0: float, tol: float = 0.001, max_iter: int = 30
    ):
        x_sym = sp.Symbol("x")
        f_expr = sp.sympify(func_f_str)
        f_d1_expr = sp.diff(f_expr, x_sym)
        f_d2_expr = sp.diff(f_d1_expr, x_sym)

        iteraciones = []
        x_curr = x0
        convergio = False

        for i in range(1, max_iter + 1):
            f_val = NoLinealesEngine._evaluar(f_expr, x_sym, x_curr)
            fd1_val = NoLinealesEngine._evaluar(f_d1_expr, x_sym, x_curr)
            fd2_val = NoLinealesEngine._evaluar(f_d2_expr, x_sym, x_curr)

            denominador = (fd1_val**2) - (f_val * fd2_val)
            if denominador == 0:
                raise ZeroDivisionError(
                    "Denominador nulo en Newton-Raphson de 2do Orden."
                )

            x_next = x_curr - ((f_val * fd1_val) / denominador)
            error = abs(x_next - x_curr)

            iteraciones.append(
                {
                    "Iteracion": i,
                    "x_i": x_curr,
                    "f(x_i)": f_val,
                    "f'(x_i)": fd1_val,
                    "f''(x_i)": fd2_val,
                    "x_{i+1}": x_next,
                    "Error": error,
                }
            )

            if error <= tol:
                convergio = True
                break

            x_curr = x_next

        return {
            "metodo": "Newton-Raphson (2do Orden)",
            "f_latex": sp.latex(f_expr),
            "fd1_latex": sp.latex(f_d1_expr),
            "fd2_latex": sp.latex(f_d2_expr),
            "raiz": x_curr if convergio else None,
            "convergio": convergio,
            "iteraciones": iteraciones,
        }