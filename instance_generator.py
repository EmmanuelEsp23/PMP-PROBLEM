import random
import numpy as np

def generate_and_save_instance():
    """
    Asks user for n, p, and max distance, then generates a random
    symmetric distance matrix and saves it to a .txt file.
    """
    
    # --- 1. Get Parameters ---
    try:
        n = int(input("Enter number of nodes (n): "))
        p = int(input(f"Enter number of medians (p): "))
        max_dist = int(input("Enter max distance (e.g., 100): "))
        filename = input("Enter filename to save (e.g., 'instance_n50.txt'): ")
        if not filename.endswith(".txt"):
            filename += ".txt"
    except ValueError:
        print("Error: Invalid input. Please use integers.")
        return

    # --- 2. Generate Matrix ---
    # Create an n x n matrix of random integers
    # from 1 (inclusive) to max_dist (inclusive)
    dist_matrix = np.random.randint(1, max_dist + 1, size=(n, n))
    
    # Make the matrix symmetric (dist[i,j] = dist[j,i])
    for i in range(n):
        for j in range(i + 1, n):
            dist_matrix[j, i] = dist_matrix[i, j]
    
    # Set the diagonal to 0 (dist[i,i] = 0)
    np.fill_diagonal(dist_matrix, 0)

    # --- 3. Save to File ---
    try:
        with open(filename, 'w') as f:
            # Write n (number of nodes)
            f.write(f"{n}\n")
            
            # Write p (number of medians)
            f.write(f"{p}\n")
            
            # Write the matrix, row by row
            for row in dist_matrix:
                # Convert row of numbers to a string, separated by spaces
                row_str = ' '.join(map(str, row))
                f.write(f"{row_str}\n")
                
        print(f"\nSuccess! Instance generated and saved to '{filename}'")
        print("Matrix preview (first 5x5):")
        print(dist_matrix[:5, :5])

    except IOError as e:
        print(f"Error: Could not write to file. {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    generate_and_save_instance()