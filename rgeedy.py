import random
import numpy as np
import sys
import time 

#Function that reads the instance file
def read_instance_from_file(filename):
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
                print("Error: Dimensions mismatch.")
                sys.exit()
            return n, p, dist_matrix_np
    except Exception as e:
        print(f"Error reading file {e}")
        sys.exit()

#Function that calculates the total cost of a solution (used for verification)
def calculate_total_cost(current_solution, n, dist_matrix):
    total_cost = 0
    if not current_solution: return float('inf')
    for i in range(n):
        min_dist = float('inf')
        for j in current_solution:
            dist = dist_matrix[i][j]
            if dist < min_dist:
                min_dist = dist
        total_cost += min_dist
    return total_cost

#Function that implements the rgreedy construction heuristic, 
#tried to explain it with detail so we can show real understanding on the constructive, sorry for the long comments dr. Roger
#(its not exactly GRASP, but its similar because it uses RCL) at least what I understood from the paper
def rgreedy_constructor(n, p, dist_matrix, k_rcl):
    print(f"\nRunning rgreedy (RCL = {k_rcl})")
    solution = [] # Final solution facilities
    candidates = list(range(n)) # All nodes are initially candidates
    min_dists = np.full(n, float('inf'))  # Minimum distances to current facilities
    final_cost = float('inf') # Final cost of the solution

    for i in range(p):
        candidate_costs = []

        #Evaluate each candidate
        for c in candidates:
            new_total_cost = 0.0
            for j in range(n):
                dist_to_c = dist_matrix[j][c]
                new_total_cost += min(min_dists[j], dist_to_c)
            candidate_costs.append((new_total_cost, c))

        # Restricted Candidate List creation
        candidate_costs.sort(key=lambda x: x[0])
        rcl_size = min(k_rcl, len(candidate_costs))
        rcl = candidate_costs[:rcl_size]
        
        # Randomized selection from RCL
        chosen_cost, chosen_node = random.choice(rcl)
        final_cost = chosen_cost 

        # Append chosen node to solution and update structures
        solution.append(chosen_node)
        candidates.remove(chosen_node)
        for j in range(n): # Update min distances
            dist_to_chosen = dist_matrix[j][chosen_node]
            min_dists[j] = min(min_dists[j], dist_to_chosen)   
    return solution, final_cost

#Local search development (Swapping by first found routine)

def update_structures(solution, n, dist_matrix):
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
    
    
    closest, sec_closest, current_cost = update_structures(current_solution, n, dist_matrix)
    print(f"\nStarting Fast Local Search (Initial Cost: {current_cost:.4f})")
    
    while True:
        improvement_found = False
        nodes_out = [x for x in range(n) if x not in current_solution]
        
        for m_out in current_solution:
            for m_in in nodes_out:
                
                delta = 0.0
                for i in range(n):
                    dist_current = dist_matrix[i][closest[i]]
                    dist_in = dist_matrix[i][m_in]
                    
                    if closest[i] == m_out:
                        dist_sec = dist_matrix[i][sec_closest[i]] if sec_closest[i] != -1 else float('inf')
                        new_dist = min(dist_sec, dist_in)
                    else:
                        new_dist = min(dist_current, dist_in)
                        
                    delta += (new_dist - dist_current)

                #First found strategy
                if delta < -1e-9:
                    current_solution.remove(m_out)
                    current_solution.append(m_in)
                    current_cost += delta
                    improvement_found = True
                    closest, sec_closest, current_cost_check = update_structures(current_solution, n, dist_matrix)
                    assert abs(current_cost - current_cost_check) < 1e-6, "Cost mismatch after update!"
                         
                    break 
            
            if improvement_found:
                break 

        if not improvement_found:
            print("Local optimum reached.") 
            break 

    return current_solution, current_cost

#Main part
if __name__ == "__main__":
    filename = input("\nEnter instance file path: ")
    n, p, dist_matrix = read_instance_from_file(filename)
    
    print(f"Initial data: n={n}, p={p}")
    
    # Timer start
    total_start_time = time.perf_counter()

    #Constructive Phase
    constructive_start_time = time.perf_counter()
    sol_constructive, cost_constructive = rgreedy_constructor(n, p, dist_matrix, k_rcl=3)
    constructive_end_time = time.perf_counter()
    
    print(f"Constructive Cost: {cost_constructive:.4f}")

    #Local Search Phase
    ls_start_time = time.perf_counter()
    sol_final, cost_final = local_search_fast(sol_constructive, n, dist_matrix)
    ls_end_time = time.perf_counter()
    
    # Timer end
    total_end_time = time.perf_counter()

    
    print("\nSolution Summary:")
    print(f"Initial Cost: {cost_constructive:.2f}")
    print(f"Final Cost: {cost_final:.2f}")
    print(f"Improvement: {cost_constructive - cost_final:.2f}")
    
    print("\nTime Report:")
    print(f"Constructive (rgreedy):   {constructive_end_time - constructive_start_time:.6f} seconds")
    print(f"Local Search (swapping):  {ls_end_time - ls_start_time:.6f} seconds")
    print(f"Total execution time:     {total_end_time - total_start_time:.6f} seconds\n")