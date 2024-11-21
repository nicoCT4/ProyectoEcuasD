import numpy as np
import matplotlib.pyplot as plt

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


def DatosDelPrimerOrden():
   """
   Devuelve los datos analíticos de la ecuación diferencial de primer orden.
   Retorna:
   - t_analitico: Array con los tiempos.
   - y_analitico: Array con las soluciones analíticas.
   """
   t_analitico = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
   y_analitico = np.array([50.00, 51.44, 52.92, 54.45, 56.02, 57.63, 59.28, 60.97, 62.71, 64.49, 66.33])
   return t_analitico, y_analitico


# Resolver ecuación diferencial de primer orden
def SolveFirstOrder():
   """
   Resuelve una ecuación diferencial de primer orden usando el método seleccionado por el usuario y compara con datos analíticos predefinidos.
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
      t_num, y_num = RungeKutta4(f, y0, t0, tf, h)
   elif method == "ABM":
      t_num, y_num = AdamsBashforthMoulton(f, y0, t0, tf, h)

   # Obtener datos analíticos predefinidos
   t_analitico, y_analitico = DatosDelPrimerOrden()

   # Imprimir resultados numéricos y analíticos
   print("\n--- Comparación de Soluciones ---")
   print("{:<10} {:<20} {:<20}".format("t", "y(t) Numérica", "y(t) Analítica"))
   for i in range(len(t_analitico)):
      print(f"{t_analitico[i]:<10.4f} {y_num[i]:<20.8f} {y_analitico[i]:<20.8f}")

   # Graficar las soluciones
   plt.figure(figsize=(10, 6))
   plt.plot(t_num, y_num, 'b.-', label='Solución Numérica')
   plt.plot(t_analitico, y_analitico, 'r.--', label='Solución Analítica (Datos)')
   plt.xlabel('Tiempo t')
   plt.ylabel('y(t)')
   plt.title('Comparación entre la solución numérica y datos analíticos (Ecuación Logística)')
   plt.legend()
   plt.grid(True)
   plt.show()


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

def DatosDelSegundoOrden():
   """
   Devuelve los datos analíticos de la ecuación diferencial de segundo orden.
   Retorna:
   - t_analitico: Array con los tiempos.
   - y_analitico: Array con las soluciones analíticas.
   """
   t_analitico = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
   y_analitico = np.array([0.200, -0.193, 0.193, -0.190, 0.187, -0.184, 0.181, -0.178, 0.174, -0.171, 0.168])
   return t_analitico, y_analitico


def SolveHarmonicOscillator():
   """
   Resuelve la ecuación diferencial de un oscilador armónico simple y compara con datos analíticos predefinidos.
   """
   print("\nResolviendo la ecuación diferencial de segundo orden homogénea (movimiento armónico simple):")
   g = 9.81
   L = float(input("Ingrese la longitud del péndulo (L): "))
   y0 = float(input("Ingrese el valor inicial de y (y0 en radianes): "))
   yp0 = float(input("Ingrese el valor inicial de y' (yp0 en rad/s): "))
   t0 = float(input("Ingrese el tiempo inicial (t0): "))
   tf = float(input("Ingrese el tiempo final (tf): "))
   h = float(input("Ingrese el tamaño del paso (h): "))
   method = SelectMethod()
   system = HarmonicOscillatorEquation(g, L)

   if method == "RK4":
      t_num, Y = RungeKutta4(system, [y0, yp0], t0, tf, h)
   elif method == "ABM":
      t_num, Y = AdamsBashforthMoulton(system, [y0, yp0], t0, tf, h)

   # Obtener datos analíticos predefinidos
   t_analitico, y_analitico = DatosDelSegundoOrden()

   # Imprimir resultados numéricos y analíticos
   print("\n--- Comparación de Soluciones ---")
   print("{:<10} {:<20} {:<20} {:<20}".format("t", "y(t) Numérica", "y'(t) Numérica", "y(t) Analítica"))
   for i in range(len(t_analitico)):
      print(f"{t_analitico[i]:<10.4f} {Y[i, 0]:<20.8f} {Y[i, 1]:<20.8f} {y_analitico[i]:<20.8f}")

   # Graficar las soluciones
   plt.figure(figsize=(10, 6))
   plt.plot(t_num, Y[:, 0], 'b.-', label='Desplazamiento Numérico')
   plt.plot(t_analitico, y_analitico, 'r.--', label='Desplazamiento Analítico (Datos)')
   plt.xlabel('Tiempo t')
   plt.ylabel('Desplazamiento y(t)')
   plt.title('Comparación entre la solución numérica y datos analíticos (Oscilador Armónico)')
   plt.legend()
   plt.grid(True)
   plt.show()



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

def DatosDelSistema():
   """
   Devuelve los datos analíticos del sistema de ecuaciones diferenciales.
   Retorna:
   - t_analitico: Array con los tiempos.
   - x_analitico: Array con los valores de x(t) analíticos.
   - v_analitico: Array con los valores de v(t) analíticos.
   """
   t_analitico = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
   x_analitico = np.array([0.200e-01, -0.0410, -0.0230, 0.0078, 0.00238, -0.00128, -0.00020, 0.000192, 0.00000881, -0.0000269, 0.00000137])
   v_analitico = np.array([-0.1810e-01, 0.0560, 0.0191, -0.00945, -0.00168, 0.00144, 0.0000912, -0.000204, 0.00000704, 0.0000268, -0.00000351])
   return t_analitico, x_analitico, v_analitico


def SolveSpringDamper():
   """
   Resuelve el sistema de ecuaciones masa-resorte-amortiguador y compara con datos analíticos predefinidos.
   """
   print("\nResolviendo el sistema masa-resorte-amortiguador:")
   k = float(input("Ingrese la constante del resorte (k): "))
   m = float(input("Ingrese la masa (m): "))
   b = float(input("Ingrese el coeficiente de amortiguamiento (b): "))
   x0 = float(input("Ingrese el valor inicial de x (x0): "))
   v0 = float(input("Ingrese el valor inicial de v (v0): "))
   t0 = float(input("Ingrese el tiempo inicial (t0): "))
   tf = float(input("Ingrese el tiempo final (tf): "))
   h = float(input("Ingrese el tamaño del paso (h): "))
   method = SelectMethod()
   system = SpringDamperSystem(k, m, b)

   if method == "RK4":
      t_num, Y = RungeKutta4(system, [x0, v0], t0, tf, h)
   elif method == "ABM":
      t_num, Y = AdamsBashforthMoulton(system, [x0, v0], t0, tf, h)

   # Obtener datos analíticos predefinidos
   t_analitico, x_analitico, v_analitico = DatosDelSistema()

   # Imprimir resultados numéricos y analíticos
   print("\n--- Comparación de Soluciones ---")
   print("{:<10} {:<20} {:<20} {:<20} {:<20}".format("t", "x(t) Numérica", "v(t) Numérica", "x(t) Analítica", "v(t) Analítica"))
   for i in range(len(t_analitico)):
      print(f"{t_analitico[i]:<10.4f} {Y[i, 0]:<20.8f} {Y[i, 1]:<20.8f} {x_analitico[i]:<20.8f} {v_analitico[i]:<20.8f}")

   # Graficar las soluciones de x(t)
   plt.figure(figsize=(10, 6))
   plt.plot(t_num, Y[:, 0], 'b.-', label='x(t) Numérica')
   plt.plot(t_analitico, x_analitico, 'r.--', label='x(t) Analítica (Datos)')
   plt.xlabel('Tiempo t')
   plt.ylabel('Desplazamiento x(t)')
   plt.title('Comparación entre la solución numérica y datos analíticos (Desplazamiento)')
   plt.legend()
   plt.grid(True)
   plt.show()

   # Graficar las soluciones de v(t)
   plt.figure(figsize=(10, 6))
   plt.plot(t_num, Y[:, 1], 'b.-', label='v(t) Numérica')
   plt.plot(t_analitico, v_analitico, 'r.--', label='v(t) Analítica (Datos)')
   plt.xlabel('Tiempo t')
   plt.ylabel('Velocidad v(t)')
   plt.title('Comparación entre la solución numérica y datos analíticos (Velocidad)')
   plt.legend()
   plt.grid(True)
   plt.show()


