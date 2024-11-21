import numpy as np
import matplotlib.pyplot as plt
from Funciones import *

# Modificación del programa para resolver EDs con diferentes métodos

def main():
    while True:
        print("\n=== Menú Principal ===")
        print("1. Resolver ecuación diferencial de primer orden (Ecuación Logística)")
        print("2. Salir")
        
        choice = input("Seleccione una opción: ")
        
        if choice == "1":
            solve_first_order()
        elif choice == "2":
            print("Saliendo del programa.")
            break
        else:
            print("Opción inválida. Intente de nuevo.")

if __name__ == "__main__":
    main()