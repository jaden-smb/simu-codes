import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import tkinter as tk
from tkinter import ttk

# Constantes
TIME_SPAN = (0, 10)
TIME_EVAL_POINTS = 300

def mass_spring_damped_system(t, y, m, k, c):
    """
    Define la ecuación diferencial para un sistema masa-resorte amortiguado.
    
    Parámetros:
    t (float): Variable de tiempo.
    y (list): Lista que contiene desplazamiento y velocidad.
    m (float): Masa del objeto.
    k (float): Constante del resorte.
    c (float): Coeficiente de amortiguamiento.
    
    Retorna:
    list: Derivadas del desplazamiento y la velocidad.
    """
    x, v = y
    dxdt = v
    dvdt = -c/m * v - k/m * x
    return [dxdt, dvdt]

def calculate_damping_ratio(m, k, c):
    """
    Calcula el coeficiente de amortiguamiento (ζ).
    
    Parámetros:
    m (float): Masa del objeto.
    k (float): Constante del resorte.
    c (float): Coeficiente de amortiguamiento.
    
    Retorna:
    float: Coeficiente de amortiguamiento (ζ).
    """
    return c / (2 * np.sqrt(m * k))

def run_simulation():
    """
    Ejecuta la simulación del sistema masa-resorte amortiguado basado en las entradas del usuario.
    """
    try:
        # Leer valores de entrada
        m = float(mass_entry.get())
        k = float(spring_constant_entry.get())
        c = float(damping_coefficient_entry.get())
        x0 = float(initial_displacement_entry.get())
        v0 = float(initial_velocity_entry.get())
    except ValueError:
        print("Por favor, ingrese valores numéricos válidos.")
        return

    # Calcular el coeficiente de amortiguamiento
    damping_ratio = calculate_damping_ratio(m, k, c)
    
    if damping_ratio > 1:
        damping_type.set("Sistema Sobreamortiguado")
    elif damping_ratio == 1:
        damping_type.set("Sistema Críticamente Amortiguado")
    else:
        damping_type.set("Sistema Subamortiguado")

    # Puntos de evaluación de tiempo
    t_eval = np.linspace(TIME_SPAN[0], TIME_SPAN[1], TIME_EVAL_POINTS)

    # Resolver la ecuación diferencial
    sol = solve_ivp(mass_spring_damped_system, TIME_SPAN, [x0, v0], t_eval=t_eval, args=(m, k, c))

    # Graficar los resultados
    plot_results(sol)

def plot_results(sol):
    """
    Grafica los resultados de la simulación.
    
    Parámetros:
    sol (OdeResult): El objeto solución retornado por solve_ivp.
    """
    plt.plot(sol.t, sol.y[0])
    plt.title('Movimiento Amortiguado del Sistema Masa-Resorte')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Desplazamiento (m)')
    plt.grid(True)
    plt.show()

def create_gui():
    """
    Crea la interfaz gráfica para la simulación del sistema masa-resorte.
    """
    root = tk.Tk()
    root.title("Simulación del Sistema Masa-Resorte")

    # Crear y colocar los campos de entrada y etiquetas
    ttk.Label(root, text="Masa (kg):").grid(column=0, row=0, padx=10, pady=5)
    global mass_entry
    mass_entry = ttk.Entry(root)
    mass_entry.grid(column=1, row=0, padx=10, pady=5)

    ttk.Label(root, text="Constante del Resorte (N/m):").grid(column=0, row=1, padx=10, pady=5)
    global spring_constant_entry
    spring_constant_entry = ttk.Entry(root)
    spring_constant_entry.grid(column=1, row=1, padx=10, pady=5)

    ttk.Label(root, text="Coeficiente de Amortiguamiento (N·s/m):").grid(column=0, row=2, padx=10, pady=5)
    global damping_coefficient_entry
    damping_coefficient_entry = ttk.Entry(root)
    damping_coefficient_entry.grid(column=1, row=2, padx=10, pady=5)

    ttk.Label(root, text="Desplazamiento Inicial (m):").grid(column=0, row=3, padx=10, pady=5)
    global initial_displacement_entry
    initial_displacement_entry = ttk.Entry(root)
    initial_displacement_entry.grid(column=1, row=3, padx=10, pady=5)

    ttk.Label(root, text="Velocidad Inicial (m/s):").grid(column=0, row=4, padx=10, pady=5)
    global initial_velocity_entry
    initial_velocity_entry = ttk.Entry(root)
    initial_velocity_entry.grid(column=1, row=4, padx=10, pady=5)

    # Crear y colocar el botón de ejecución
    run_button = ttk.Button(root, text="Ejecutar Simulación", command=run_simulation)
    run_button.grid(column=0, row=5, columnspan=2, padx=10, pady=10)

    # Crear y colocar la etiqueta para el tipo de amortiguamiento
    global damping_type
    damping_type = tk.StringVar()
    ttk.Label(root, textvariable=damping_type).grid(column=0, row=6, columnspan=2, padx=10, pady=5)

    # Iniciar el bucle principal de eventos
    root.mainloop()

if __name__ == "__main__":
    create_gui()