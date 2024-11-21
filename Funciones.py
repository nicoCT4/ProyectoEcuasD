import numpy as np

# Definición de métodos numéricos

def RungeKutta4(f, y0, t0, tf, h):
   """
   Método de Runge-Kutta de cuarto orden para resolver EDOs.

   Parámetros:
   - f: Función que describe la EDO (puede ser escalar o sistema).
   - y0: Valor inicial (puede ser escalar o vector).
   - t0: Tiempo inicial.
   - tf: Tiempo final.
   - h: Tamaño del paso.

   Retorna:
   - t: Array con los tiempos.
   - y: Array con las soluciones.
   """
   t = np.arange(t0, tf + h, h)
   n = len(t)
   y = np.zeros((n, len(y0))) if isinstance(y0, (list, np.ndarray)) else np.zeros(n)
   y[0] = y0

   for i in range(n - 1):
      k1 = h * f(t[i], y[i])
      k2 = h * f(t[i] + h / 2, y[i] + k1 / 2)
      k3 = h * f(t[i] + h / 2, y[i] + k2 / 2)
      k4 = h * f(t[i] + h, y[i] + k3)
      y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

   return t, y

def SelectMethod():
   """
   Permite al usuario seleccionar el método numérico a utilizar mostrando una descripción de cada uno.

   Retorna:
   - Un string indicando el método seleccionado ("RK4" o "ABM").
   """
   print("\nSeleccione el método numérico para resolver la ecuación diferencial:")
   print("1. Método de Runge-Kutta de cuarto orden (RK4)")
   print("2. Método de Adams-Bashforth-Moulton (ABM)")
   
   choice = input("\nSeleccione una opción (1 para RK4, 2 para ABM): ")
   if choice == "1":
      return "RK4"
   elif choice == "2":
      return "ABM"
   else:
      print("Opción inválida. Usando RK4 por defecto.")
      return "RK4"

def AdamsBashforthMoulton(f, y0, t0, tf, h):
   """
   Implementación del método predictor-corrector Adams-Bashforth-Moulton.

   Parámetros:
   - f: Función que describe la EDO (puede ser escalar o sistema).
   - y0: Valor inicial (puede ser escalar o vector).
   - t0: Tiempo inicial.
   - tf: Tiempo final.
   - h: Tamaño del paso.

   Retorna:
   - t: Array con los tiempos.
   - y: Array con las soluciones.
   """
   t = np.arange(t0, tf + h, h)
   n = len(t)

   # Ajustar la inicialización de y dependiendo si es un sistema o un escalar
   if isinstance(y0, (list, np.ndarray)):
      y = np.zeros((n, len(y0)))  # Matriz para sistemas de ecuaciones
   else:
      y = np.zeros(n)  # Vector para una ecuación diferencial escalar
   y[0] = y0

   # Usar RK4 para los primeros pasos (necesarios para ABM)
   for i in range(min(3, n - 1)):  # RK4 para los primeros 3 pasos
      k1 = h * f(t[i], y[i])
      k2 = h * f(t[i] + h / 2, y[i] + k1 / 2)
      k3 = h * f(t[i] + h / 2, y[i] + k2 / 2)
      k4 = h * f(t[i] + h, y[i] + k3)
      y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

   # Predictor-Corrector para los pasos restantes
   for i in range(3, n - 1):
      # Predictor (Adams-Bashforth de cuarto orden)
      y_pred = y[i] + h / 24 * (55 * f(t[i], y[i]) - 59 * f(t[i - 1], y[i - 1]) +
                                 37 * f(t[i - 2], y[i - 2]) - 9 * f(t[i - 3], y[i - 3]))
      # Corrector (Adams-Moulton de tercer orden)
      y[i + 1] = y[i] + h / 24 * (9 * f(t[i + 1], y_pred) + 19 * f(t[i], y[i]) -
                                 5 * f(t[i - 1], y[i - 1]) + f(t[i - 2], y[i - 2]))
   return t, y

# Funciones para las ecuaciones específicas

def LogisticEquation(r, K):
   """
   Retorna la función de la ecuación logística.

   Parámetros:
   - r: Tasa de crecimiento.
   - K: Capacidad de carga.

   Retorna:
   - Función que representa la ecuación diferencial.
   """
   return lambda t, y: r * y * (1 - y / K)

def LogisticAnalyticalSolution(r, K, y0, t):
   """
   Solución analítica de la ecuación logística.

   Parámetros:
   - r: Tasa de crecimiento.
   - K: Capacidad de carga.
   - y0: Valor inicial.
   - t: Array de tiempos.

   Retorna:
   - Array con las soluciones analíticas.
   """
   return (K * y0 * np.exp(r * t)) / (K + y0 * (np.exp(r * t) - 1))

# Resolver ecuación diferencial de primer orden
def SolveFirstOrder():
   """
   Resuelve una ecuación diferencial de primer orden usando el método seleccionado por el usuario.
   """
   print("\nResolviendo ecuación diferencial de primer orden (Ecuación Logística):")
   r = float(input("Ingrese la tasa de crecimiento (r): "))
   K = float(input("Ingrese la capacidad de carga (K): "))
   f = LogisticEquation(r, K)
   y0 = float(input("Ingrese el valor inicial de y (y0): "))
   t0 = float(input("Ingrese el tiempo inicial (t0): "))
   tf = float(input("Ingrese el tiempo final (tf): "))
   h = float(input("Ingrese el tamaño del paso (h): "))
   method = SelectMethod()

   if method == "RK4":
      t, y = RungeKutta4(f, y0, t0, tf, h)
   elif method == "ABM":
      t, y = AdamsBashforthMoulton(f, y0, t0, tf, h)

   # Solución analítica
   t_analytical = t
   y_analytical = LogisticAnalyticalSolution(r, K, y0, t_analytical)
   analytical_equation = f"y(x) = (K * y0 * exp(r * x)) / (K + y0 * (exp(r * x) - 1)), donde K={K}, y0={y0}, r={r}"

   print("\n--- Solución Numérica ---")
   print("{:<10} {:<15}".format("x", "y(x) numérica"))
   for i in range(len(t)):
      print(f"{t[i]:<10.4f} {y[i]:<15.8f}")

   print("\n--- Solución Analitica ---")
   print(f"{analytical_equation}")

# Resolver ecuaciones de segundo orden y sistemas

def HarmonicOscillatorEquation(g, L):
   """
   Retorna el sistema de ecuaciones para la ecuación diferencial de segundo orden homogénea.

   Parámetros:
   - g: Aceleración debido a la gravedad.
   - L: Longitud del péndulo.

   Retorna:
   - Sistema de ecuaciones representado por una función lambda.
   """
   omega_squared = g / L
   def system(t, Y):
      y, yp = Y
      return np.array([yp, -omega_squared * y])
   return system

def SolveHarmonicOscillator():
   """
   Resuelve la ecuación diferencial de un oscilador armónico simple usando el método seleccionado.
   """
   print("\nResolviendo la ecuación diferencial de segundo orden homogénea (movimiento armónico simple):")
   g = 9.81
   L = float(input("Ingrese la longitud del péndulo (L): "))
   y0 = float(input("Ingrese el valor inicial de y (y0): "))
   yp0 = float(input("Ingrese el valor inicial de y' (yp0): "))
   t0 = float(input("Ingrese el tiempo inicial (t0): "))
   tf = float(input("Ingrese el tiempo final (tf): "))
   h = float(input("Ingrese el tamaño del paso (h): "))
   method = SelectMethod()
   system = HarmonicOscillatorEquation(g, L)

   if method == "RK4":
      t, Y = RungeKutta4(system, [y0, yp0], t0, tf, h)
   elif method == "ABM":
      print("El método Adams-Bashforth-Moulton aún no está implementado para sistemas de ecuaciones.")
      return

   # Solución analítica
   omega = np.sqrt(g / L)
   A = y0
   B = yp0 / omega
   y_analytical = A * np.cos(omega * t) + B * np.sin(omega * t)
   analytical_equation = f"y(t) = {A} * cos({omega:.4f} * t) + {B} * sin({omega:.4f} * t)"

   print("\n--- Solución Numérica ---")
   print("{:<10} {:<15}".format("t", "y(t) numérica",))
   for i in range(len(t)):
      print(f"{t[i]:<10.4f} {Y[i, 0]:<15.8f}")

   print("\n--- Solución Analítica ---")
   print(f"{analytical_equation}")

def SpringDamperSystem(k, m, b):
   """
   Sistema de ecuaciones diferenciales para masa-resorte-amortiguador.

   Parámetros:
   - k: Constante del resorte.
   - m: Masa del objeto.
   - b: Coeficiente de amortiguamiento.

   Retorna:
   - Sistema de ecuaciones representado por una función lambda.
   """
   def system(t, Y):
      x, v = Y
      dxdt = v
      dvdt = -(k / m) * x - (b / m) * v
      return np.array([dxdt, dvdt])
   return system

def SolveSpringDamper():
   """
   Resuelve el sistema de ecuaciones masa-resorte-amortiguador usando el método seleccionado.
   """
   print("\nResolviendo el sistema masa-resorte-amortiguador:")
   k = float(input("Ingrese la constante del resorte (k): "))
   m = float(input("Ingrese la masa (m): "))
   b = float(input("Ingrese el coeficiente de amortiguación (b): "))
   x0 = float(input("Ingrese el valor inicial de x (x0): "))
   v0 = float(input("Ingrese el valor inicial de v (v0): "))
   t0 = float(input("Ingrese el tiempo inicial (t0): "))
   tf = float(input("Ingrese el tiempo final (tf): "))
   h = float(input("Ingrese el tamaño del paso (h): "))
   method = SelectMethod()
   system = SpringDamperSystem(k, m, b)

   if method == "RK4":
      t, Y = RungeKutta4(system, [x0, v0], t0, tf, h)
   elif method == "ABM":
      t, Y = AdamsBashforthMoulton(system, [x0, v0], t0, tf, h)

   print("\n--- Solución Numérica ---")
   print("{:<10} {:<15} ".format("t", "x(t) numérica", ))
   for i in range(len(t)):
      print(f"{t[i]:<10.4f} {Y[i, 0]:<15.8f} ")

   print("\n--- Sistema de Ecuaciones ---")
   print("dx/dt = v")
   print(f"dv/dt = -({k}/{m}) * x - ({b}/{m}) * v")
