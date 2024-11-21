import numpy as np

# Definición de métodos numéricos

def runge_kutta_4(f, y0, t0, tf, h):
   """Método RK4 para resolver EDOs."""
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

def select_method():
   """Permite al usuario seleccionar el método numérico a utilizar mostrando una descripción de cada uno."""
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

def adams_bashforth_moulton(f, y0, t0, tf, h):
   """Implementación del método predictor-corrector Adams-Bashforth-Moulton."""
   t = np.arange(t0, tf + h, h)
   n = len(t)
   y = np.zeros((n, len(y0)))  # Matriz para sistemas
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

def logistic_equation(r, K):
   """Retorna la función de la ecuación logística."""
   return lambda t, y: r * y * (1 - y / K)

def logistic_analytical_solution(r, K, y0, t):
   """Solución analítica de la ecuación logística."""
   return (K * y0 * np.exp(r * t)) / (K + y0 * (np.exp(r * t) - 1))

# Resolver ecuación diferencial de primer orden
def solve_first_order():
   print("\nResolviendo ecuación diferencial de primer orden (Ecuación Logística):")
   r = float(input("Ingrese la tasa de crecimiento (r): "))
   K = float(input("Ingrese la capacidad de carga (K): "))
   f = logistic_equation(r, K)
   y0 = float(input("Ingrese el valor inicial de y (y0): "))
   t0 = float(input("Ingrese el tiempo inicial (t0): "))
   tf = float(input("Ingrese el tiempo final (tf): "))
   h = float(input("Ingrese el tamaño del paso (h): "))
   method = select_method()

   if method == "RK4":
      t, y = runge_kutta_4(f, y0, t0, tf, h)
   elif method == "ABM":
      t, y = adams_bashforth_moulton(f, y0, t0, tf, h)

   # Solución analítica
   t_analytical = t
   y_analytical = logistic_analytical_solution(r, K, y0, t_analytical)
   analytical_equation = f"y(x) = (K * y0 * exp(r * x)) / (K + y0 * (exp(r * x) - 1)), donde K={K}, y0={y0}, r={r}"

   print("\n--- Solución Numérica ---")
   print("{:<10} {:<15}".format("x", "y(x) numérica"))
   for i in range(len(t)):
      print(f"{t[i]:<10.4f} {y[i]:<15.8f}")

   print("\n--- Solución Analítica ---")
   print(f"{analytical_equation}")
   print("{:<10} {:<15}".format("x", "y(x) analítica"))
   for i in range(len(t_analytical)):
      print(f"{t_analytical[i]:<10.4f} {y_analytical[i]:<15.8f}")

# Función para resolver ecuaciones de segundo orden homogéneas
def harmonic_oscillator_equation(g, L):
   """Retorna el sistema de ecuaciones para la ecuación diferencial de segundo orden homogénea."""
   omega_squared = g / L
   def system(t, Y):
      y, yp = Y
      return np.array([yp, -omega_squared * y])
   return system

def solve_harmonic_oscillator():
   print("\nResolviendo la ecuación diferencial de segundo orden homogénea (movimiento armónico simple):")
   g = 9.81
   L = float(input("Ingrese la longitud del péndulo (L): "))
   y0 = float(input("Ingrese el valor inicial de y (y0): "))
   yp0 = float(input("Ingrese el valor inicial de y' (yp0): "))
   t0 = float(input("Ingrese el tiempo inicial (t0): "))
   tf = float(input("Ingrese el tiempo final (tf): "))
   h = float(input("Ingrese el tamaño del paso (h): "))
   method = select_method()
   system = harmonic_oscillator_equation(g, L)

   if method == "RK4":
      t, Y = runge_kutta_4(system, [y0, yp0], t0, tf, h)
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
   print("{:<10} {:<15} {:<15}".format("t", "y(t) numérica", "y'(t) numérica"))
   for i in range(len(t)):
      print(f"{t[i]:<10.4f} {Y[i, 0]:<15.8f} {Y[i, 1]:<15.8f}")

   print("\n--- Solución Analítica ---")
   print(f"{analytical_equation}")
   print("{:<10} {:<15}".format("t", "y(t) analítica"))
   for i in range(len(t)):
      print(f"{t[i]:<10.4f} {y_analytical[i]:<15.8f}")

# Resolver sistemas de EDOs (2x2 Masa-Resorte-Amortiguador)
def spring_damper_system(k, m, b):
   """Sistema de ecuaciones diferenciales para masa-resorte-amortiguador."""
   def system(t, Y):
      x, v = Y
      dxdt = v
      dvdt = -(k / m) * x - (b / m) * v
      return np.array([dxdt, dvdt])
   return system

def solve_spring_damper():
   print("\nResolviendo el sistema masa-resorte-amortiguador:")
   k = float(input("Ingrese la constante del resorte (k): "))
   m = float(input("Ingrese la masa (m): "))
   b = float(input("Ingrese el coeficiente de amortiguación (b): "))
   x0 = float(input("Ingrese el valor inicial de x (x0): "))
   v0 = float(input("Ingrese el valor inicial de v (v0): "))
   t0 = float(input("Ingrese el tiempo inicial (t0): "))
   tf = float(input("Ingrese el tiempo final (tf): "))
   h = float(input("Ingrese el tamaño del paso (h): "))
   method = select_method()
   system = spring_damper_system(k, m, b)

   if method == "RK4":
      t, Y = runge_kutta_4(system, [x0, v0], t0, tf, h)
   elif method == "ABM":
      t, Y = adams_bashforth_moulton(system, [x0, v0], t0, tf, h)

   print("\n--- Solución Numérica ---")
   print("{:<10} {:<15} {:<15}".format("t", "x(t) numérica", "v(t) numérica"))
   for i in range(len(t)):
      print(f"{t[i]:<10.4f} {Y[i, 0]:<15.8f} {Y[i, 1]:<15.8f}")

   print("\n--- Sistema de Ecuaciones ---")
   print("dx/dt = v")
   print(f"dv/dt = -({k}/{m}) * x - ({b}/{m}) * v")

