import math
import random
import numpy as np
import sys
import time 

def read_instance_from_file(filename):
    print(f"Reading instance {filename}")
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
        print(f"Error reading file {e}")
        sys.exit()

def calculate_total_cost(current_solution, n, dist_matrix):
    total_cost = 0
    if not current_solution: return float('inf')
    for i in range(n):
        min_dist = min(dist_matrix[i][j] for j in current_solution)
        total_cost += min_dist
    return total_cost


def rgreedy_constructor(n, p, dist_matrix, k_rcl):
    """
    Implementa rgreedy usando un cálculo de costo INCREMENTAL (delta)
    para ser mucho más rápido.
    """
    print(f"\n--- Running rgreedy (FAST implementation, k_rcl = {k_rcl}) ---")
    solution = []
    candidates = list(range(n))
    
    # Este array guardará la distancia más corta actual para CADA cliente
    # Inicia en infinito.
    min_dists = np.full(n, float('inf'))
    final_cost = float('inf')

    for i in range(p):
        print(f"Iteration {i+1}/{p} (selecting median)...")
        candidate_costs = []

        # --- 1. PASO DE EVALUACIÓN (OPTIMIZADO) ---
        for c in candidates:
            new_total_cost = 0.0
            
            # No llamamos a calculate_total_cost.
            # Calculamos el costo incrementalmente.
            for j in range(n):
                # ¿Está el candidato 'c' más cerca que la mejor mediana actual?
                dist_to_c = dist_matrix[j][c]
                new_total_cost += min(min_dists[j], dist_to_c)
            
            candidate_costs.append((new_total_cost, c))

        # --- 2. PASO DE RESTRICCIÓN (RCL) ---
        candidate_costs.sort(key=lambda x: x[0])
        rcl_size = min(k_rcl, len(candidate_costs))
        rcl = candidate_costs[:rcl_size]
        
        # --- 3. PASO DE ALEATORIZACIÓN ---
        # 'chosen_cost' es el costo total del sistema con este nodo añadido
        chosen_cost, chosen_node = random.choice(rcl)
        final_cost = chosen_cost # Guardamos el costo de la última iteración

        # --- 4. PASO DE SELECCIÓN ---
        solution.append(chosen_node)
        candidates.remove(chosen_node)
        
        # --- 5. ACTUALIZACIÓN CRUCIAL ---
        # Actualizamos permanentemente las distancias mínimas con el nodo elegido
        for j in range(n):
            dist_to_chosen = dist_matrix[j][chosen_node]
            min_dists[j] = min(min_dists[j], dist_to_chosen)
            
        print(f"    -> Node chosen: {chosen_node} (New Total Cost: {chosen_cost:.2f})")
        
    # El costo final es el último costo total calculado
    return solution, final_cost

# --- BÚSQUEDA LOCAL - (Sin cambios) ---

def update_structures(solution, n, dist_matrix):
    # Update closest and second closest medians for each node
    closest = [-1] * n
    sec_closest = [-1] * n
    current_cost = 0.0

    for i in range(n):
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
    
    #Initialize structures
    closest, sec_closest, current_cost = update_structures(current_solution, n, dist_matrix)
    
    while True:
        improvement_found = False
        nodes_out = [x for x in range(n) if x not in current_solution]
        
        for m_out in current_solution:
            for m_in in nodes_out:
                
                # Delta calculation
                delta = 0.0
                for i in range(n):
                    dist_current = dist_matrix[i][closest[i]]
                    dist_in = dist_matrix[i][m_in]
                    
                    if closest[i] == m_out:
                        dist_sec = dist_matrix[i][sec_closest[i]]
                        new_dist = min(dist_sec, dist_in)
                    else:
                        new_dist = min(dist_current, dist_in)
                        
                    delta += (new_dist - dist_current)

                # First found routine
                if delta < -1e-9:
                    current_solution.remove(m_out)
                    current_solution.append(m_in)
                    current_cost += delta
                    improvement_found = True
                    
                    print(f"    -> Quick swap: {m_out} (out) <-> {m_in} (in) | New cost: {current_cost:.4f}")
                    
                    closest, sec_closest, current_cost_check = update_structures(current_solution, n, dist_matrix)
                    
                    if abs(current_cost - current_cost_check) > 1e-5:
                         print(f"Warning: Delta drift detected. Resetting cost to {current_cost_check:.4f}")
                         current_cost = current_cost_check
                         
                    break 
            
            if improvement_found:
                break 

        if not improvement_found:
            print("  > Local optimum reached.")
            break

    return current_solution, current_cost

# --- Bloque Principal de Ejecución (AJUSTADO) ---
if __name__ == "__main__":
    filename = input("Enter instance file path: ")
    n, p, dist_matrix = read_instance_from_file(filename)
    
    print(f"\n--- Data: n={n}, p={p} ---")
    
    # Total Timer Start
    total_start_time = time.perf_counter()

    # 1. RGREEDY (Ahora usa la versión rápida)
    constructive_start_time = time.perf_counter()
    # AHORA CAPTURAMOS EL COSTO DEVUELTO, AHORRANDO UN CÁLCULO
    sol_constructive, cost_constructive = rgreedy_constructor(n, p, dist_matrix, k_rcl=3)
    constructive_end_time = time.perf_counter()
    
    # Ya no necesitamos recalcular el costo constructivo
    print(f"Constructive Cost: {cost_constructive:.4f}")

    # 2. FAST LOCAL SEARCH
    ls_start_time = time.perf_counter()
    print("\n--- Starting Fast Local Search ---")
    sol_final, cost_final = local_search_fast(sol_constructive, n, dist_matrix)
    ls_end_time = time.perf_counter()
    
    total_end_time = time.perf_counter()
    # --- FIN CRONÓMETRO TOTAL ---

    
    print(f"\n=== FINAL RESULT ===")
    print(f"Initial Cost: {cost_constructive:.4f} -> Final Cost: {cost_final:.4f}")
    print(f"Improvement: {cost_constructive - cost_final:.4f}")
    print(f"Final Medians: {sorted(sol_final)}")
    
    # --- IMPRESIÓN DE TIEMPOS ---
    print("\n--- EXECUTION TIME ---")
    print(f"Constructive (rgreedy): {constructive_end_time - constructive_start_time:.6f} seconds")
    print(f"Local Search (Fast):    {ls_end_time - ls_start_time:.6f} seconds")
    print(f"Total Time:             {total_end_time - total_start_time:.6f} seconds")