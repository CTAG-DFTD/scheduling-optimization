from itertools import product
import time
from src.maxcut_scheduling_utils import run_scheduling 

def maxcut_bruteforce(maxcut_matrix):
    """
    Solves the MaxCut problem by brute force.
    
    Parameters:
    -----------
    maxcut_matrix : numpy.ndarray
        Weight matrix of the edges (may be symmetric or upper-triangular)
    
    Returns:
    --------
    bitstring_maxcut, elapsed_time
    """
    start_time = time.time()
    
    n = len(maxcut_matrix)
    max_value = -float('inf')
    bitstring_maxcut = None
    
    # Try all possible partitions (2^n combinations)
    for bitstring in product([0, 1], repeat=n):
        cut_value = 0
        
        # Compute the cut value for this bitstring
        for i in range(n):
            for j in range(i+1, n):
                # If the vertices are in different sets, add the weight
                if bitstring[i] != bitstring[j]:
                    cut_value += maxcut_matrix[i][j]
        
        # Update the best cut if we find a better one
        if cut_value > max_value:
            max_value = cut_value
            bitstring_maxcut = list(bitstring)
    
    elapsed_time = time.time() - start_time
    
    return bitstring_maxcut, elapsed_time


if __name__ == "__main__":
    solver_name = "BruteForce"
    solver_function = maxcut_bruteforce
    days = 7
    num_refs = 3
    num_slots = 5

    print(f"""
    Running scheduling optimization. Parameters:
        - Solver: {solver_name}
        - Days: {days}
        - Number of references: {num_refs}
        - Number of slots per day: {num_slots}
          """)

    

    run_scheduling(days, num_refs, num_slots, solver_function, solver_name) 

    print("\nScheduling task complete.")