import math
import random
import numpy as np
import sys

# --- FUNCIONES AUXILIARES (Lectura y Cálculo Básico) ---

def read_instance_from_file(filename):
    print(f"--- Reading instance file: {filename} ---")
    try:
        with open(filename, 'r') as f:
            n = int(f.readline().strip())
            p = int(f.readline().strip())
            matrix_rows = []
            for _ in range(n):
                line = f.readline().strip()
                row = list(map(float, line.split()))
                matrix_rows.append(row)
            dist_matrix_np = np.array(matrix_rows)
            if dist_matrix_np.shape != (n, n):
                print(f"Error: Dimensions mismatch.")
                sys.exit()
            return n, p, dist_matrix_np
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit()

def calculate_total_cost(current_solution, n, dist_matrix):
    # Función clásica para validación y conteo inicial
    total_cost = 0
    if not current_solution: return float('inf')
    for i in range(n):
        min_dist = min(dist_matrix[i][j] for j in current_solution)
        total_cost += min_dist
    return total_cost

# --- HEURÍSTICA CONSTRUCTIVA (rgreedy) ---

def rgreedy_constructor(n, p, dist_matrix, k_rcl):
    print(f"\n--- Running rgreedy (k_rcl = {k_rcl}) ---")
    solution = []
    candidates = list(range(n))
    for i in range(p):
        candidate_costs = []
        for c in candidates:
            temp_sol = solution + [c]
            cost = calculate_total_cost(temp_sol, n, dist_matrix)
            candidate_costs.append((cost, c))
        candidate_costs.sort(key=lambda x: x[0])
        rcl = candidate_costs[:min(k_rcl, len(candidate_costs))]
        chosen_cost, chosen_node = random.choice(rcl)
        solution.append(chosen_node)
        candidates.remove(chosen_node)
    return solution, chosen_cost

# --- BÚSQUEDA LOCAL OPTIMIZADA (Fast Interchange + First Improvement) ---

def update_structures(solution, n, dist_matrix):
    """
    Precalcula para cada nodo cuál es su mediana más cercana
    y cuál es la SEGUNDA más cercana.
    Retorna:
    - closest: lista donde closest[i] es la mediana más cercana al nodo i
    - sec_closest: lista donde sec_closest[i] es la 2da más cercana
    - current_cost: el costo total actual
    """
    closest = [-1] * n
    sec_closest = [-1] * n
    current_cost = 0.0

    for i in range(n):
        # Encontrar la 1ra y 2da mejor distancia para el nodo i
        d1, d2 = float('inf'), float('inf')
        m1, m2 = -1, -1
        
        for med in solution:
            dist = dist_matrix[i][med]
            if dist < d1:
                d2, m2 = d1, m1
                d1, m1 = dist, med
            elif dist < d2:
                d2, m2 = dist, med
        
        closest[i] = m1
        sec_closest[i] = m2
        current_cost += d1
        
    return closest, sec_closest, current_cost

def local_search_fast(initial_solution, n, dist_matrix):
    current_solution = list(initial_solution)
    
    # Inicializar estructuras
    closest, sec_closest, current_cost = update_structures(current_solution, n, dist_matrix)
    
    while True:
        improvement_found = False
        nodes_out = [x for x in range(n) if x not in current_solution]
        
        # Probar todos los posibles intercambios
        for m_out in current_solution:
            for m_in in nodes_out:
                
                # --- CÁLCULO RÁPIDO DEL DELTA ---
                delta = 0.0
                for i in range(n):
                    dist_current = dist_matrix[i][closest[i]]
                    dist_in = dist_matrix[i][m_in]
                    
                    if closest[i] == m_out:
                        # CASO CRÍTICO: El nodo i pierde su mediana actual.
                        # Debe elegir entre su segunda opción o la nueva mediana entrante.
                        dist_sec = dist_matrix[i][sec_closest[i]]
                        new_dist = min(dist_sec, dist_in)
                    else:
                        # CASO NORMAL: El nodo i conserva su mediana actual.
                        # Solo cambia si la nueva mediana entrante es AÚN mejor.
                        new_dist = min(dist_current, dist_in)
                        
                    delta += (new_dist - dist_current)

                # --- ESTRATEGIA FIRST IMPROVEMENT ---
                if delta < -1e-9:
                    # Aplicar intercambio
                    current_solution.remove(m_out)
                    current_solution.append(m_in)
                    
                    # Actualizar costo rápidamente
                    current_cost += delta
                    improvement_found = True
                    
                    print(f"    -> Quick swap: {m_out} (out) <-> {m_in} (in) | New cost: {current_cost:.4f}")
                    
                    # Reconstruir estructuras para la siguiente iteración
                    # (Es necesario porque muchas 1ras y 2das opciones pueden haber cambiado)
                    closest, sec_closest, current_cost_check = update_structures(current_solution, n, dist_matrix)
                    
                    # Pequeña validación para asegurar que el delta funciona bien
                    if abs(current_cost - current_cost_check) > 1e-5:
                         print(f"Warning: Delta drift detected. Resetting cost to {current_cost_check:.4f}")
                         current_cost = current_cost_check
                         
                    break # Salir del bucle interno (m_in)
            
            if improvement_found:
                break # Salir del bucle externo (m_out) y reiniciar while

        if not improvement_found:
            print("  > Local optimum reached.")
            break

    return current_solution, current_cost

# --- MAIN ---
if __name__ == "__main__":
    filename = input("Enter instance file path: ")
    n, p, dist_matrix = read_instance_from_file(filename)
    
    print(f"\n--- Data: n={n}, p={p} ---")

    # 1. RGREEDY
    sol_constructive, cost_constructive = rgreedy_constructor(n, p, dist_matrix, k_rcl=3)
    print(f"Constructive Cost: {cost_constructive:.4f}")

    # 2. FAST LOCAL SEARCH
    print("\n--- Starting Fast Local Search ---")
    sol_final, cost_final = local_search_fast(sol_constructive, n, dist_matrix)
    
    print(f"\n=== FINAL RESULT ===")
    print(f"Initial Cost: {cost_constructive:.4f} -> Final Cost: {cost_final:.4f}")
    print(f"Improvement: {cost_constructive - cost_final:.4f}")
    print(f"Final Medians: {sorted(sol_final)}")