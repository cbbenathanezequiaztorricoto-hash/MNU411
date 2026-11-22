import numpy as np

class SistemasLinealesEngine:

    # -------------------------------------------------------------------------
    # 3.1 ELIMINACIÓN GAUSSIANA CON PIVOTAJE PARCIAL
    # -------------------------------------------------------------------------
    @staticmethod
    def eliminacion_gaussiana(A, b):
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float).reshape(-1, 1)
        n = len(b)
        Ab = np.hstack([A, b])

        pasos_matriz = []
        pasos_matriz.append({
            "paso": "Matriz Aumentada Inicial [A|b]",
            "formula": "Estado inicial",
            "matriz": [[round(val, 4) for val in row] for row in Ab.tolist()],
        })

        paso_num = 1
        for k in range(n - 1):
            # Pivotaje Parcial
            max_row = np.argmax(np.abs(Ab[k:, k])) + k
            if Ab[max_row, k] == 0:
                raise ValueError("El sistema no tiene solución única (pivote nulo).")
            if max_row != k:
                Ab[[k, max_row]] = Ab[[max_row, k]]
                pasos_matriz.append({
                    "paso": f"Paso {paso_num}: Pivotaje Parcial",
                    "formula": f"Intercambio Fila L_{k+1} <-> Fila L_{max_row+1}",
                    "matriz": [[round(val, 4) for val in row] for row in Ab.tolist()],
                })
                paso_num += 1

            # Eliminación hacia adelante
            for i in range(k + 1, n):
                mult = Ab[i, k] / Ab[k, k]
                Ab[i, k:] -= mult * Ab[k, k:]
                pasos_matriz.append({
                    "paso": f"Paso {paso_num}: Eliminación de A[{i+1}][{k+1}]",
                    "formula": f"L_{i+1} = L_{i+1} - ({mult:.4f}) * L_{k+1}",
                    "matriz": [[round(val, 4) for val in row] for row in Ab.tolist()],
                })
                paso_num += 1

        # Sustitución hacia atrás
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            if Ab[i, i] == 0:
                raise ZeroDivisionError("Cero en la diagonal durante la sustitución hacia atrás.")
            x[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1 : n], x[i + 1 : n])) / Ab[i, i]

        return {
            "metodo": "Eliminación Gaussiana con Pivotaje Parcial",
            "matriz_triangular": [[round(val, 4) for val in row] for row in Ab[:, :-1].tolist()],
            "solucion": [round(val, 4) for val in x.tolist()],
            "pasos_matriz": pasos_matriz,
        }

    # -------------------------------------------------------------------------
    # 3.2 MÉTODO DE GAUSS-JORDAN CON PIVOTAJE PARCIAL
    # -------------------------------------------------------------------------
    @staticmethod
    def gauss_jordan(A, b):
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float).reshape(-1, 1)
        n = len(b)
        Ab = np.hstack([A, b])

        pasos_matriz = []
        pasos_matriz.append({
            "paso": "Matriz Aumentada Inicial [A|b]",
            "formula": "Estado inicial",
            "matriz": [[round(val, 4) for val in row] for row in Ab.tolist()],
        })

        paso_num = 1
        for k in range(n):
            # Pivotaje Parcial
            max_row = np.argmax(np.abs(Ab[k:, k])) + k
            if Ab[max_row, k] == 0:
                raise ValueError("El sistema no tiene solución única (pivote nulo).")
            if max_row != k:
                Ab[[k, max_row]] = Ab[[max_row, k]]
                pasos_matriz.append({
                    "paso": f"Paso {paso_num}: Pivotaje Parcial",
                    "formula": f"Intercambio Fila L_{k+1} <-> Fila L_{max_row+1}",
                    "matriz": [[round(val, 4) for val in row] for row in Ab.tolist()],
                })
                paso_num += 1

            # Normalizar fila pivote
            pivote = Ab[k, k]
            Ab[k] = Ab[k] / pivote
            pasos_matriz.append({
                "paso": f"Paso {paso_num}: Normalizar Pivote L_{k+1}",
                "formula": f"L_{k+1} = L_{k+1} / ({pivote:.4f})",
                "matriz": [[round(val, 4) for val in row] for row in Ab.tolist()],
            })
            paso_num += 1

            # Eliminación en ambas direcciones (arriba y abajo)
            for i in range(n):
                if i != k:
                    factor = Ab[i, k]
                    if abs(factor) > 1e-12:
                        Ab[i] -= factor * Ab[k]
                        pasos_matriz.append({
                            "paso": f"Paso {paso_num}: Anular elemento en Fila L_{i+1}",
                            "formula": f"L_{i+1} = L_{i+1} - ({factor:.4f}) * L_{k+1}",
                            "matriz": [[round(val, 4) for val in row] for row in Ab.tolist()],
                        })
                        paso_num += 1

        return {
            "metodo": "Gauss-Jordan con Pivotaje Parcial",
            "solucion": [round(val, 4) for val in Ab[:, -1].tolist()],
            "pasos_matriz": pasos_matriz,
        }

    # -------------------------------------------------------------------------
    # 3.3 MÉTODO ITERATIVO DE JACOBI
    # -------------------------------------------------------------------------
    @staticmethod
    def jacobi(A, b, x0=None, tol=1e-4, max_iter=30, tipo_error="residuo"):
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        n = len(b)
        x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)

        D = np.diag(A)
        if np.any(D == 0):
            raise ZeroDivisionError("Cero detectado en la diagonal principal de A.")

        R = A - np.diagflat(D)
        historial = []
        convergio = False

        for k in range(1, max_iter + 1):
            x_new = (b - np.dot(R, x)) / D

            if tipo_error == "residuo":
                residuos = np.abs(b - np.dot(A, x_new))
                err_eval = float(np.max(residuos))
            else:
                err_eval = float(np.linalg.norm(x_new - x))

            row_iter = {"i": k}
            for idx in range(n):
                row_iter[f"x_{idx+1}"] = round(x_new[idx], 5)
            row_iter["Error"] = round(err_eval, 5)
            historial.append(row_iter)

            if err_eval <= tol:
                convergio = True
                break
            x = x_new

        return {
            "metodo": "Jacobi",
            "solucion": [round(val, 5) for val in x.tolist()],
            "convergio": convergio,
            "iteraciones": historial,
        }

    # -------------------------------------------------------------------------
    # 3.4 MÉTODO ITERATIVO DE GAUSS-SEIDEL
    # -------------------------------------------------------------------------
    @staticmethod
    def gauss_seidel(A, b, x0=None, tol=1e-4, max_iter=30, tipo_error="residuo"):
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        n = len(b)
        x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)

        historial = []
        convergio = False

        for k in range(1, max_iter + 1):
            x_old = x.copy()
            for i in range(n):
                if A[i, i] == 0:
                    raise ZeroDivisionError(f"Cero en la diagonal principal (fila {i+1}).")
                sum1 = sum(A[i, j] * x[j] for j in range(i))
                sum2 = sum(A[i, j] * x_old[j] for j in range(i + 1, n))
                x[i] = (b[i] - sum1 - sum2) / A[i, i]

            if tipo_error == "residuo":
                residuos = np.abs(b - np.dot(A, x))
                err_eval = float(np.max(residuos))
            else:
                err_eval = float(np.linalg.norm(x - x_old))

            row_iter = {"i": k}
            for idx in range(n):
                row_iter[f"x_{idx+1}"] = round(x[idx], 5)
            row_iter["Error"] = round(err_eval, 5)
            historial.append(row_iter)

            if err_eval <= tol:
                convergio = True
                break

        return {
            "metodo": "Gauss-Seidel",
            "solucion": [round(val, 5) for val in x.tolist()],
            "convergio": convergio,
            "iteraciones": historial,
        }