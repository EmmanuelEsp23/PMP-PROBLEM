import math
import random
import numpy as np

def get_user_data_v2():
    """
    Prompts the user for the number of nodes (n), the number of medians (p),
    and the DISTANCE MATRIX row by row.
    """
    dist_matrix = []
    
    while True:
        try:
            n = int(input("Enter the TOTAL number of nodes (n): "))
            if n > 0:
                break
            print("Error: n must be greater than 0.")
        except ValueError:
            print("Error: Please enter a valid integer.")
            
    while True:
        try:
            p = int(input(f"Enter the number of medians to select (p, max {n}): "))
            if 0 < p <= n:
                break
            print(f"Error: p must be a number between 1 and {n}.")
        except ValueError:
            print("Error: Please enter a valid integer.")

    print("\n--- Enter the Distance Matrix (row by row) ---")
    print(f"You must enter {n} numbers per row, separated by spaces.")
    
    for i in range(n):
        while True:
            try:
                row_input = input(f"  Row {i} (nodes {i} to 0...{n-1}): ")
                # Convert the string input into a list of floats
                row = list(map(float, row_input.split()))
                
                if len(row) == n:
                    dist_matrix.append(row)
                    break
                else:
                    print(f"Error: The row must have exactly {n} numbers. (You entered {len(row)})")
            except ValueError:
                print("Error: Incorrect format. Please enter numbers separated by spaces.")
    
    # Convert the Python list of lists into a NumPy array
    dist_matrix_np = np.array(dist_matrix)
    return n, p, dist_matrix_np

def calculate_total_cost(current_solution, n, dist_matrix):
    """
    Calculates the total cost of assigning all 'n' clients to the
    nearest median in the 'current_solution'.
    (This function's logic is unchanged)
    """
    total_cost = 0
    
    if not current_solution:
        return float('inf')
        
    # For each client i (from 0 to n-1)
    for i in range(n):
        min_distance = float('inf')
        
        # Find the distance to the nearest median (j)
        for j in current_solution:
            dist = dist_matrix[i][j]
            if dist < min_distance:
                min_distance = dist
                
        # Add that minimum distance to the total cost
        total_cost += min_distance
        
    return total_cost

def rgreedy_constructor(n, p, dist_matrix, k_rcl):
    """
    Implements the rgreedy (Greedy Randomized) constructive heuristic.
    
    k_rcl: The size of the Restricted Candidate List (e.g., 3 or 5).
    (This function's logic is unchanged)
    """
    print(f"\n--- Running rgreedy (k_rcl = {k_rcl}) ---")
    
    # The solution (set of selected medians) starts empty
    solution = []
    
    # Candidates are all nodes (indices 0 to n-1)
    candidates = list(range(n))
    
    # Repeat p times (to select p medians)
    for i in range(p):
        print(f"Iteration {i+1}/{p} (selecting median)...")
        candidate_costs = []
        
        # --- 1. EVALUATION STEP ---
        # Evaluate the cost of adding each remaining candidate
        for c in candidates:
            temp_solution = solution + [c]
            cost = calculate_total_cost(temp_solution, n, dist_matrix)
            candidate_costs.append((cost, c)) # (cost, node_id)
            
        # --- 2. RESTRICTION STEP (RCL) ---
        # Sort candidates by cost (best to worst)
        candidate_costs.sort(key=lambda x: x[0])
        
        # Create the RCL with the 'k' best
        # Ensure we don't ask for more candidates than are left
        rcl_size = min(k_rcl, len(candidate_costs))
        rcl = candidate_costs[:rcl_size]
        
        print(f"    RCL (Top {rcl_size}): {rcl}")
        
        # --- 3. RANDOMIZATION STEP ---
        # Choose one at random from the RCL
        chosen_cost, chosen_node = random.choice(rcl)
        
        # --- 4. SELECTION STEP ---
        solution.append(chosen_node)
        candidates.remove(chosen_node)
        
        print(f"    -> Node chosen: {chosen_node} (Cost: {chosen_cost:.2f})")
        print(f"    Current solution: {solution}\n")
        
    return solution, calculate_total_cost(solution, n, dist_matrix)

# --- Main Execution Block ---
if __name__ == "__main__":
    
    # 1. Get user data (v2 - English)
    n_nodes, p_medians, distance_matrix = get_user_data_v2()
    
    # We no longer need calculate_matrix_distances()
    
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