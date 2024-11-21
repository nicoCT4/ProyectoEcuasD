import numpy as np
import matplotlib.pyplot as plt


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
   print("1. Método de Runge-Kutta de cuarto orden (RK4):")
   print("   Descripción: Es un método de un solo paso que mejora la precisión al calcular varias estimaciones intermedias")
   print("   dentro de cada paso. El método RK4 es especialmente popular por su equilibrio entre precisión y complejidad.")
   print()
   print("2. Método de Adams-Bashforth-Moulton (Predictor-Corrector):")
   print("   Descripción: Es un método de varios pasos que utiliza valores anteriores para predecir la solución (Adams-Bashforth)")
   print("   y luego corregirla (Adams-Moulton). Combina métodos explícitos e implícitos para mejorar la precisión y estabilidad.")
   
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

def pendulum_equation(g, L):
   """Retorna el sistema de ecuaciones para el péndulo simple."""
   def system(t, Y):
      theta, omega = Y
      return np.array([omega, -g / L * np.sin(theta)])
   return system

def lotka_volterra(alpha, beta, delta, gamma):
   """Retorna el sistema de ecuaciones para Lotka-Volterra (depredador-presa)."""
   def system(t, Y):
      x, y = Y
      dxdt = alpha * x - beta * x * y
      dydt = delta * x * y - gamma * y
      return np.array([dxdt, dydt])
   return system

# Soluciones

def solve_first_order():
   print("\nResolviendo ecuaciones diferenciales de primer orden:")
   print("Seleccione la ecuación a resolver:")
   print("1. Ecuación logística")
   print("2. Ecuación de crecimiento exponencial")
   choice = input("Seleccione una opción: ")

   if choice == "1":
      r = float(input("Ingrese la tasa de crecimiento (r): "))
      K = float(input("Ingrese la capacidad de carga (K): "))
      f = logistic_equation(r, K)
   elif choice == "2":
      r = float(input("Ingrese la tasa de crecimiento exponencial (r): "))
      f = lambda t, y: r * y
   else:
      print("Opción inválida. Resolviendo la ecuación logística por defecto.")
      r = float(input("Ingrese la tasa de crecimiento (r): "))
      K = float(input("Ingrese la capacidad de carga (K): "))
      f = logistic_equation(r, K)

   # Parámetros iniciales
   y0 = float(input("Ingrese el valor inicial de y (y0): "))
   t0 = float(input("Ingrese el tiempo inicial (t0): "))
   tf = float(input("Ingrese el tiempo final (tf): "))
   h = float(input("Ingrese el tamaño del paso (h): "))

   # Método numérico
   method = select_method()

   # Resolver con el método seleccionado
   if method == "RK4":
      t, y = runge_kutta_4(f, y0, t0, tf, h)
   elif method == "ABM":
      t, y = adams_bashforth_moulton(f, y0, t0, tf, h)

   # Mostrar los resultados
   print("\nResultados (t, y):")
   print("{:<10} {:<15}".format("t", "y(t)"))
   for i in range(len(t)):
      print(f"{t[i]:<10.4f} {y[i]:<15.8f}")



def solve_pendulum():
   print("\nResolviendo la ecuación del péndulo simple:")
   
   # Parámetros del péndulo
   g = 9.81  # Gravedad (m/s^2)
   L = float(input("Ingrese la longitud del péndulo (L): "))

   # Condiciones iniciales
   theta0 = float(input("Ingrese el valor inicial del ángulo (θ0, en radianes): "))
   omega0 = float(input("Ingrese la velocidad angular inicial (θ'(0)): "))
   t0 = float(input("Ingrese el tiempo inicial (t0): "))
   tf = float(input("Ingrese el tiempo final (tf): "))
   h = float(input("Ingrese el tamaño del paso (h): "))

   # Selección del método numérico
   method = select_method()

   # Resolver con el método seleccionado
   system = pendulum_equation(g, L)
   if method == "RK4":
      t, Y = rk4_system(system, [theta0, omega0], t0, tf, h)
   
   # Graficar resultados
   plt.figure(figsize=(10, 6))
   plt.plot(t, Y[:, 0], label="θ(t) (RK4)", color="blue")
   plt.xlabel("Tiempo (t)")
   plt.ylabel("Ángulo (θ)")
   plt.title("Ecuación del Péndulo Simple Resuelta con RK4")
   plt.legend()
   plt.grid()
   plt.show()

def solve_lotka_volterra():
   print("\nResolviendo el modelo de Lotka-Volterra (depredador-presa):")
   
   # Parámetros del modelo
   alpha = float(input("Ingrese la tasa de crecimiento de la presa (α): "))
   beta = float(input("Ingrese la tasa de depredación (β): "))
   delta = float(input("Ingrese la tasa de crecimiento del depredador al consumir presas (δ): "))
   gamma = float(input("Ingrese la tasa de muerte del depredador (γ): "))

   # Condiciones iniciales
   x0 = float(input("Ingrese el tamaño inicial de la población de presas (x0): "))
   y0 = float(input("Ingrese el tamaño inicial de la población de depredadores (y0): "))
   t0 = float(input("Ingrese el tiempo inicial (t0): "))
   tf = float(input("Ingrese el tiempo final (tf): "))
   h = float(input("Ingrese el tamaño del paso (h): "))

   # Selección del método numérico
   method = select_method()

   # Resolver con el método seleccionado
   system = lotka_volterra(alpha, beta, delta, gamma)
   if method == "RK4":
      t, Y = rk4_system(system, [x0, y0], t0, tf, h)

   # Graficar resultados
   plt.figure(figsize=(10, 6))
   plt.plot(t, Y[:, 0], label="Población de presas (x)", color="green")
   plt.plot(t, Y[:, 1], label="Población de depredadores (y)", color="red")
   plt.xlabel("Tiempo (t)")
   plt.ylabel("Población")
   plt.title("Modelo de Lotka-Volterra Resuelto con RK4")
   plt.legend()
   plt.grid()
   plt.show()



