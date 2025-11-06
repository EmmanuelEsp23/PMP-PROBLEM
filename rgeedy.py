import math
import random
import numpy as np
import sys 

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

def local_search_interchange(initial_solution, n, dist_matrix):
    """
    Improves a given solution using a 'best improvement'
    Interchange (Swap) local search.
    """
    
    # Start with the solution from the constructive heuristic
    # We use list() to create a copy, so we don't change the original
    current_solution = list(initial_solution)
    current_cost = calculate_total_cost(current_solution, n, dist_matrix)
    
    # Get p from the solution length
    p = len(current_solution)

    # Loop until no improvement is found
    while True:
        improvement_found = False
        best_swap = None
        best_cost_so_far = current_cost

        print(f"  > Starting search iteration. Current cost: {current_cost:.4f}")

        # 1. Create list of nodes NOT in the solution
        nodes_out = [node for node in range(n) if node not in current_solution]
        
        # 2. Iterate through all possible swaps
        # For every node 'node_out' (IN the solution)
        for i in range(p):
            node_out = current_solution[i]
            
            # For every node 'node_in' (OUT of the solution)
            for node_in in nodes_out:
                
                # 3. Create the temporary swapped solution
                # (copy list, remove node_out, add node_in)
                temp_solution = list(current_solution)
                temp_solution.pop(i) # Remove node_out by its index
                temp_solution.append(node_in)
                
                # 4. Calculate its cost
                temp_cost = calculate_total_cost(temp_solution, n, dist_matrix)
                
                # 5. Check for improvement (using a small tolerance for float comparison)
                if temp_cost < (best_cost_so_far - 1e-9):
                    best_cost_so_far = temp_cost
                    best_swap = (node_out, node_in)
                    improvement_found = True
                        
        # 6. Apply the best move found in this iteration
        if improvement_found:
            node_to_remove, node_to_add = best_swap
            current_solution.remove(node_to_remove)
            current_solution.append(node_to_add)
            current_cost = best_cost_so_far
            
            print(f"    -> Found improvement! Swapping {node_to_remove} (out) with {node_to_add} (in).")
            print(f"    New cost: {current_cost:.4f}")
        else:
            # No swaps in the entire neighborhood improved the solution
            print("  > No further improvements found. Local optimum reached.")
            break # Exit the 'while True' loop
            
    return current_solution, current_cost

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
    
    # --- STEP 1: CONSTRUCTIVE HEURISTIC ---
    print("\n--- STEP 1: Running rgreedy (Constructive Heuristic) ---")
    constructive_solution, constructive_cost = rgreedy_constructor(
        n_nodes, p_medians, distance_matrix, K_RCL
    )
    print("--- rgreedy Heuristic Finished! ---")
    print(f"Constructive solution: {constructive_solution}")
    print(f"Constructive cost: {constructive_cost:.4f}\n")

    # --- STEP 2: LOCAL SEARCH HEURISTIC ---
    print("--- STEP 2: Running Local Search (Interchange) ---")
    final_solution, final_cost = local_search_interchange(
        constructive_solution, n_nodes, distance_matrix
    )
    
    print("\n--- Algorithm Finished! ---")
    print(f"Initial Solution (from rgreedy): {constructive_solution} (Cost: {constructive_cost:.4f})")
    print(f"Final Solution (after Local Search): {final_solution} (Cost: {final_cost:.4f})")
    
    improvement = constructive_cost - final_cost
    if improvement > 1e-9: # Check for any real improvement
        print(f"Total Improvement: {improvement:.4f}")
    else:
        print("The initial solution was already a local optimum.")