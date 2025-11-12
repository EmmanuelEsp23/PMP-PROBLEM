import random
import numpy as np

def generate_and_save_instance():
    # User inputs
    try:
        n = int(input("\nEnter number of nodes (n): "))
        p = int(input(f"Enter number of facilities (p): "))
        max_dist = int(input("Enter max distance (example, 100): "))
        filename = input("Enter filename to save instance (without extension): ")
        if not filename.endswith(".txt"):
            filename += ".txt"
    except ValueError:
        print("Invalid input. Please enter integer values.")
        return

    #Generate random distance matrix
    dist_matrix = np.random.randint(1, max_dist + 1, size=(n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist_matrix[j, i] = dist_matrix[i, j]
    np.fill_diagonal(dist_matrix, 0)

    #Save to file
    try:
        with open(filename, 'w') as f:
            f.write(f"{n}\n")
            f.write(f"{p}\n")
            
            for row in dist_matrix:
                row_str = ' '.join(map(str, row))
                f.write(f"{row_str}\n")
                
        print(f"Instance generated to '{filename}'\n")

    except IOError as e:
        print(f"Error: Could not write to file. {e}")

if __name__ == "__main__":
    generate_and_save_instance()