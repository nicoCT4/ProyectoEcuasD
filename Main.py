from Funciones import *

def main():
    while True:
        print("\n=== Menú Principal ===")
        print("1. Resolver ecuaciones diferenciales de primer orden (Ecuación Logística)")
        print("2. Resolver ecuaciones diferenciales de orden superior (Homogénea - Armónico Simple)")
        print("3. Resolver un sistema de ecuaciones diferenciales de 2x2 (Masa-Resorte-Amortiguador)")
        print("4. Salir")
        
        choice = input("Seleccione una opción: ")
        
        if choice == "1":
            solve_first_order()
        elif choice == "2":
            solve_harmonic_oscillator()
        elif choice == "3":
            solve_spring_damper()
        elif choice == "4":
            print("Saliendo del programa.")
            break
        else:
            print("Opción inválida. Intente de nuevo.")

if __name__ == "__main__":
    main()
