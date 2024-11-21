import numpy as np

# Definición de métodos numéricos

def runge_kutta_4(f, y0, t0, tf, h):
   """Método RK4 para resolver EDOs."""
   t = np.arange(t0, tf + h, h)
   n = len(t)
   y = np.zeros(n)
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
   y = np.zeros(n)
   y[0] = y0

   for i in range(1, n - 1):
      # Predictor (Adams-Bashforth)
      y_pred = y[i] + h * f(t[i], y[i])
      
      # Corrector (Adams-Moulton)
      y[i + 1] = y[i] + (h / 2) * (f(t[i], y[i]) + f(t[i + 1], y_pred))
   return t, y

# Funciones para las ecuaciones específicas

def logistic_equation(r, K):
   """Retorna la función de la ecuación logística."""
   return lambda t, y: r * y * (1 - y / K)

def logistic_analytical_solution(r, K, y0, t):
   """Solución analítica de la ecuación logística."""
   return (K * y0 * np.exp(r * t)) / (K + y0 * (np.exp(r * t) - 1))

# Resolución de la ecuación diferencial de primer orden

def solve_first_order():
   print("\nResolviendo ecuación diferencial de primer orden (Ecuación Logística):")
   
   # Parámetros de la ecuación logística
   r = float(input("Ingrese la tasa de crecimiento (r): "))
   K = float(input("Ingrese la capacidad de carga (K): "))
   f = logistic_equation(r, K)

   # Parámetros iniciales con validación de entrada
   y0 = get_valid_float("Ingrese el valor inicial de y (y0): ")
   t0 = get_valid_float("Ingrese el tiempo inicial (t0): ")
   tf = get_valid_float("Ingrese el tiempo final (tf): ")
   h = get_valid_float("Ingrese el tamaño del paso (h): ")

   # Método numérico
   method = select_method()

   # Resolver con el método seleccionado
   if method == "RK4":
      t, y = runge_kutta_4(f, y0, t0, tf, h)
   elif method == "ABM":
      t, y = adams_bashforth_moulton(f, y0, t0, tf, h)

   # Solución analítica para comparación
   t_analytical = t  # Usaremos los mismos puntos que la solución numérica
   y_analytical = logistic_analytical_solution(r, K, y0, t_analytical)
   print()
   analytical_equation = f"y(x) = (K * y0 * exp(r * x)) / (K + y0 * (exp(r * x) - 1)), donde K={K}, y0={y0}, r={r}"

   # Mostrar los resultados
   print("\n--- Solución Numérica ---")
   print("{:<10} {:<15}".format("x", "y"))
   for i in range(len(t)):
      print(f"{t[i]:<10.4f} {y[i]:<15.8f}")

   print("\n--- Solución Analítica ---")
   print(f"Ecuación: {analytical_equation}")
   print("{:<10} {:<15}".format("x", "y(x)"))
   for i in range(len(t_analytical)):
      print(f"{t_analytical[i]:<10.4f} {y_analytical[i]:<15.8f}")

def get_valid_float(prompt):
   """Función para obtener una entrada válida de punto flotante del usuario."""
   while True:
      try:
         return float(input(prompt))
      except ValueError:
         print("Entrada inválida. Por favor, ingrese un número válido.")

# Definición de métodos numéricos para sistemas

def rk4_system(system, Y0, t0, tf, h):
   """Método RK4 para resolver sistemas de EDOs."""
   t = np.arange(t0, tf + h, h)
   n = len(t)
   Y = np.zeros((n, len(Y0)))
   Y[0] = Y0

   for i in range(n - 1):
      k1 = h * system(t[i], Y[i])
      k2 = h * system(t[i] + h / 2, Y[i] + k1 / 2)
      k3 = h * system(t[i] + h / 2, Y[i] + k2 / 2)
      k4 = h * system(t[i] + h, Y[i] + k3)
      Y[i + 1] = Y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

   return t, Y

def harmonic_oscillator_equation(g, L):
   """Retorna el sistema de ecuaciones para la ecuación diferencial de segundo orden homogénea."""
   omega_squared = g / L  # ω^2 = g/L
   def system(t, Y):
      y, yp = Y
      return np.array([yp, -omega_squared * y])
   return system

# Resolver ecuación diferencial de segundo orden homogénea en términos de "y"

def solve_harmonic_oscillator():
   print("\nResolviendo la ecuación diferencial de segundo orden homogénea (movimiento armónico simple):")
   print("Forma: y'' + ω^2 * y = 0")
   
   # Parámetros del sistema
   g = 9.81  # Gravedad (m/s^2)
   L = float(input("Ingrese la longitud del péndulo (L): "))

   # Condiciones iniciales
   y0 = float(input("Ingrese el valor inicial de y (y0): "))
   yp0 = float(input("Ingrese el valor inicial de y' (yp0): "))
   t0 = float(input("Ingrese el tiempo inicial (t0): "))
   tf = float(input("Ingrese el tiempo final (tf): "))
   h = float(input("Ingrese el tamaño del paso (h): "))

   # Frecuencia angular
   omega = np.sqrt(g / L)

   # Selección del método numérico
   method = select_method()

   # Resolver con el método seleccionado
   system = harmonic_oscillator_equation(g, L)
   if method == "RK4":
      t, Y = rk4_system(system, [y0, yp0], t0, tf, h)
   elif method == "ABM":
      print("El método Adams-Bashforth-Moulton aún no está implementado para sistemas de ecuaciones.")
      return

   # Mostrar los resultados numéricos
   print("\n--- Resultados Numéricos ---")
   print("{:<10} {:<15} {:<15}".format("t", "y(t)", "y'(t)"))
   for i in range(len(t)):
      print(f"{t[i]:<10.4f} {Y[i, 0]:<15.8f} {Y[i, 1]:<15.8f}")

   # Calcular la solución analítica
   A = y0
   B = yp0 / omega
   y_analytical = A * np.cos(omega * t) + B * np.sin(omega * t)

   # Mostrar la solución analítica
   print("\n--- Solución Analítica ---")
   print("Forma: y(t) = A * cos(ω * t) + B * sin(ω * t)")
   print(f"Con A = {A}, B = {B}, ω = {omega:.4f}")
   print("{:<10} {:<15}".format("t", "y(t) analítica"))
   for i in range(len(t)):
      print(f"{t[i]:<10.4f} {y_analytical[i]:<15.8f}")


