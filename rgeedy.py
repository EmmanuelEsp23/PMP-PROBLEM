import math
import random
import numpy as np
import sys # Used for exiting on error

def read_instance_from_file(filename):
    """
    Reads an instance file containing n, p, and the distance matrix.
    """
    print(f"--- Reading instance file: {filename} ---")
    try:
        with open(filename, 'r') as f:
            # Read n (first line)
            n = int(f.readline().strip())
            
            # Read p (second line)
            p = int(f.readline().strip())
            
            matrix_rows = []
            # Read the next 'n' lines (the matrix)
            for _ in range(n):
                line = f.readline().strip()
                row = list(map(float, line.split()))
                matrix_rows.append(row)
            
            # Convert to NumPy array
            dist_matrix_np = np.array(matrix_rows)
            
            # Validation
            if dist_matrix_np.shape != (n, n):
                print(f"Error: Matrix dimensions ({dist_matrix_np.shape}) do not match n ({n}, {n}).")
                sys.exit()
                
            return n, p, dist_matrix_np
            
    except FileNotFoundError:
        print(f"Error: File not found at '{filename}'")
        sys.exit() # Exit the program
    except Exception as e:
        print(f"Error: Could not parse file. {e}")
        sys.exit() # Exit the program

def calculate_total_cost(current_solution, n, dist_matrix):
    """
    Calculates the total cost of assigning all 'n' clients to the
    nearest median in the 'current_solution'.
    """
    total_cost = 0
    
    if not current_solution:
        return float('inf')
        
    for i in range(n):
        min_distance = float('inf')
        for j in current_solution:
            dist = dist_matrix[i][j]
            if dist < min_distance:
                min_distance = dist
        total_cost += min_distance
        
    return total_cost

def rgreedy_constructor(n, p, dist_matrix, k_rcl):
    """
    Implements the rgreedy (Greedy Randomized) constructive heuristic.
    
    k_rcl: The size of the Restricted Candidate List (e.g., 3 or 5).
    """
    print(f"\n--- Running rgreedy (k_rcl = {k_rcl}) ---")
    
    solution = []
    candidates = list(range(n))
    
    for i in range(p):
        print(f"Iteration {i+1}/{p} (selecting median)...")
        candidate_costs = []
        
        for c in candidates:
            temp_solution = solution + [c]
            cost = calculate_total_cost(temp_solution, n, dist_matrix)
            candidate_costs.append((cost, c)) # (cost, node_id)
            
        candidate_costs.sort(key=lambda x: x[0])
        
        rcl_size = min(k_rcl, len(candidate_costs))
        rcl = candidate_costs[:rcl_size]
        
        print(f"    RCL (Top {rcl_size}): {rcl}")
        
        chosen_cost, chosen_node = random.choice(rcl)
        
        solution.append(chosen_node)
        candidates.remove(chosen_node)
        
        print(f"    -> Node chosen: {chosen_node} (Cost: {chosen_cost:.2f})")
        print(f"    Current solution: {solution}\n")
        
    return solution, calculate_total_cost(solution, n, dist_matrix)

# --- Main Execution Block ---
if __name__ == "__main__":
    
    # 1. Ask user for the file to read
    filename_to_read = input("Enter the path to the instance .txt file: ")
    
    # 2. Get data from the file
    n_nodes, p_medians, distance_matrix = read_instance_from_file(filename_to_read)
    
    print("\n--- Input Data Summary ---")
    print(f"Nodes (n): {n_nodes}, Medians (p): {p_medians}")
    print("Distance Matrix (first 5x5 rows/cols):")
    print(distance_matrix[:5, :5])
    
    # 3. Define the RCL size (you can change this)
    K_RCL = 3
    
    # 4. Run the constructive heuristic
    final_solution, final_cost = rgreedy_constructor(n_nodes, p_medians, distance_matrix, K_RCL)
    
    print("--- rgreedy Heuristic Finished! ---")
    print(f"The solution (selected medians) is: {final_solution}")
    print(f"Total cost of the solution: {final_cost:.4f}")