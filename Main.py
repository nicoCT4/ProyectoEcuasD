import numpy as np
import matplotlib.pyplot as plt
from Funciones import *


# Función principal modificada
def main():
   while True:
      print("\n=== Menú Principal ===")
      print("1. Resolver ecuaciones diferenciales de primer orden")
      print("2. Resolver ecuaciones diferenciales de orden superior")
      print("3. Resolver la ecuación del péndulo simple (no lineal)")
      print("4. Salir")
      
      choice = input("Seleccione una opción: ")
      
      if choice == "1":
         solve_first_order()
      elif choice == "2":
         solve_higher_order()
      elif choice == "3":
         solve_pendulum()
      elif choice == "4":
         print("Saliendo del programa.")
         break
      else:
         print("Opción inválida. Intente de nuevo.")

if __name__ == "__main__":
   main()