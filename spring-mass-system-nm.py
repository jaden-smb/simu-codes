import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

TIEMPO_INTERVALO = (0, 10)
PUNTOS_EVALUACION_TIEMPO = 300

def mass_spring_damped_system(t, y, m, k, c):
    x = y[0]
    v = y[1]
    dxdt = v
    dvdt = -k/m * x - c/m * v
    return [dxdt, dvdt]

def calculate_damping_ratio(m, k, c):
    return c / (2 * np.sqrt(m * k))

def euler_method(f, y0, t, args):
    dt = t[1] - t[0]
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = y[i-1] + dt * np.array(f(t[i-1], y[i-1], *args))
    return y

def rk4_method(f, y0, t, args):
    dt = t[1] - t[0]
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        k1 = np.array(f(t[i-1], y[i-1], *args))
        k2 = np.array(f(t[i-1] + dt/2, y[i-1] + dt/2 * k1, *args))
        k3 = np.array(f(t[i-1] + dt/2, y[i-1] + dt/2 * k2, *args))
        k4 = np.array(f(t[i], y[i-1] + dt * k3, *args))
        y[i] = y[i-1] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y

def verlet_method(f, y0, t, args):
    dt = t[1] - t[0]
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    a = np.zeros((len(t), len(y0)))
    a[0] = f(t[0], y[0], *args)
    
    y[1] = y[0] + dt * y[0][1] + 0.5 * dt**2 * a[0][1]
    
    for i in range(1, len(t)-1):
        a[i] = f(t[i], y[i], *args)
        y[i+1] = 2*y[i] - y[i-1] + dt**2 * a[i][1]
    
    return y

def run_simulation(m, k, c, x0, v0):
    damping_ratio = calculate_damping_ratio(m, k, c)
    if damping_ratio > 1:
        damping_type = "Sistema Sobreamortiguado"
    elif damping_ratio == 1:
        damping_type = "Sistema Críticamente Amortiguado"
    else:
        damping_type = "Sistema Subamortiguado"
    print(f"Tipo de Amortiguamiento: {damping_type}")

    t_eval = np.linspace(TIEMPO_INTERVALO[0], TIEMPO_INTERVALO[1], PUNTOS_EVALUACION_TIEMPO)
    y0 = np.array([x0, v0])

    sol = solve_ivp(mass_spring_damped_system, TIEMPO_INTERVALO, y0, t_eval=t_eval, args=(m, k, c))
    euler_sol = euler_method(mass_spring_damped_system, y0, t_eval, (m, k, c))
    rk4_sol = rk4_method(mass_spring_damped_system, y0, t_eval, (m, k, c))
    verlet_sol = verlet_method(mass_spring_damped_system, y0, t_eval, (m, k, c))

    plot_results(t_eval, sol, euler_sol, rk4_sol, verlet_sol)
    plot_errors(t_eval, sol, euler_sol, rk4_sol, verlet_sol)

def plot_results(t, sol, euler_sol, rk4_sol, verlet_sol):
    plt.figure(figsize=(12, 6))
    plt.plot(t, sol.y[0], label='Solución Analítica')
    plt.plot(t, euler_sol[:, 0], label="Método de Euler")
    plt.plot(t, rk4_sol[:, 0], label='Método RK4')
    plt.plot(t, verlet_sol[:, 0], label='Método Verlet')
    plt.title('Sistema Masa-Resorte Amortiguado')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Desplazamiento (m)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_errors(t, sol, euler_sol, rk4_sol, verlet_sol):
    analytical = sol.y[0]
    euler_error = np.abs(analytical - euler_sol[:, 0])
    rk4_error = np.abs(analytical - rk4_sol[:, 0])
    verlet_error = np.abs(analytical - verlet_sol[:, 0])

    plt.figure(figsize=(12, 6))
    plt.plot(t, euler_error, label="Error del Método de Euler")
    plt.plot(t, rk4_error, label='Error del Método RK4')
    plt.plot(t, verlet_error, label='Error del Método Verlet')
    plt.title('Comparación de Errores')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Error (m)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    m = float(input("Ingrese la masa (kg): "))
    k = float(input("Ingrese la constante del resorte (N/m): "))
    c = float(input("Ingrese el coeficiente de amortiguamiento (N·s/m): "))
    x0 = float(input("Ingrese el desplazamiento inicial (m): "))
    v0 = float(input("Ingrese la velocidad inicial (m/s): "))
    run_simulation(m, k, c, x0, v0)