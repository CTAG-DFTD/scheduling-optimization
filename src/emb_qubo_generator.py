"""
QUBO and MaxCut Matrix Generator for Production Scheduling

This module converts production scheduling problems into Quadratic Unconstrained 
Binary Optimization (QUBO) formulations and MaxCut graph problems. The QUBO 
formulation balances multiple objectives:
- Production optimization (meeting demand and safety stock)
- Assignment constraints (one reference per slot)
- Storage capacity constraints
- Production continuity (minimizing changeovers)

The MaxCut transformation enables the use of quantum annealers and specialized
graph optimization algorithms.
"""

import pandas as pd
import numpy as np


class EmbQuboGenerator:
    """
    Generator for QUBO and MaxCut formulations of production scheduling problems.
    
    This class reads production data from CSV files and constructs optimization
    matrices that can be solved by quantum annealers, classical QUBO solvers,
    or MaxCut optimization algorithms.
    
    Attributes:
        csv_file (str): Path to CSV file containing production data
        A (float): Weight coefficient for production objective (H_production)
        B (float): Weight coefficient for assignment constraints (H_assignment)
        C (float): Weight coefficient for storage constraints (H_storage)
        D (float): Weight coefficient for changeover minimization (H_change)
        day (int): Current day number in the scheduling horizon
        num_slots (int): Number of production time slots available per day
        num_refs (int): Number of production references (products/SKUs) to schedule
    
    Expected CSV columns:
        - C_r: Production capacity/cadence (cycles per hour per reference)
        - S_r: Current stock level for each reference
        - A_r: Storage capacity limit for each reference
        - SS_d,r: Safety stock requirement for each reference
        - D_{day},r: Demand for each reference on the specified day
    """
    
    def __init__(self, csv_file, day=1, num_slots=8, num_refs=3):
        """
        Initialize the QUBO/MaxCut generator with problem parameters.
        
        Args:
            csv_file (str): Path to CSV file with production data
            day (int, optional): Day number to schedule. Defaults to 1.
            num_slots (int, optional): Number of production slots per day. Defaults to 8.
            num_refs (int, optional): Number of references to include. Defaults to 3.
        """
        self.csv_file = csv_file  # Data file path
        self.A = 0.5  # Weight for H_production (meeting demand/safety stock)
        self.B = 1.0  # Weight for H_assignment (unique slot assignments)
        self.C = 0.4  # Weight for H_storage (respecting capacity limits)
        self.D = 0.05 # Weight for H_change (production continuity)
        self.day = day
        self.num_slots = num_slots
        self.num_refs = num_refs

    def generate_qubo_matrix(self):
        """
        Generate the QUBO matrix formulation of the scheduling problem.
        
        The QUBO objective function combines four components:
        
        1. H_production: Minimizes deviation from required production
           (safety stock + demand). Encourages meeting production targets.
        
        2. H_assignment: Enforces constraint that each slot can only be
           assigned to one reference (one product per time slot).
        
        3. H_storage: Penalizes exceeding storage capacity limits.
           Ensures production doesn't create excess inventory.
        
        4. H_change: Encourages production continuity by penalizing
           changeovers between different references in consecutive slots.
        
        Binary variables x_{r,s} represent: "Is reference r assigned to slot s?"
        
        Returns:
            tuple: (Q, var_mapping) where:
                - Q (numpy.ndarray): QUBO matrix of shape (total_vars, total_vars)
                  where total_vars = num_refs × num_slots
                - var_mapping (dict): Maps (reference, slot) tuples to linear
                  indices in the Q matrix, e.g., {(0,0): 0, (0,1): 1, ...}
        
        Matrix Structure:
            Q is a symmetric matrix where:
            - Q[i,i] contains diagonal terms (individual variable costs)
            - Q[i,j] contains interaction terms between variables i and j
        """
        
        num_slots = self.num_slots
        num_refs = self.num_refs
        day = self.day

        # Read production data from CSV
        df = pd.read_csv(self.csv_file)

        # Limit to requested number of references or available data
        if num_refs < len(df):
            df = df.iloc[:num_refs]
        else:
            num_refs = len(df)

        # Weight parameters for objective function components
        A = self.A  # H_production weight
        B = self.B  # H_assignment weight
        C = self.C  # H_storage weight
        D = self.D  # H_change weight
        
        # Extract production data for all references
        C_r = df['C_r'].values      # Production capacity (cycles/hour)
        S_r = df['S_r'].values      # Current stock levels
        A_r = df['A_r'].values      # Storage capacity limits
        SS_dr = df['SS_d,r'].values # Safety stock requirements
        D_dr = df[f'D_{day},r'].values  # Demand for the current day
        
        # Total number of binary decision variables
        total_vars = num_refs * num_slots
        
        # Initialize QUBO matrix (symmetric, upper triangular will be filled)
        Q = np.zeros((total_vars, total_vars))
        
        # Create mapping from (reference, slot) pairs to linear indices
        # Example: (0,0)->0, (0,1)->1, ..., (1,0)->num_slots, etc.
        var_mapping = {}
        idx = 0
        for r in range(num_refs):
            for s in range(num_slots):
                var_mapping[(r, s)] = idx
                idx += 1
        
        # Helper function to retrieve linear index for variable x_{r,s}
        def get_idx(r, s):
            return var_mapping.get((r, s), None)
        
        # Build the Q matrix term by term
        for r in range(num_refs):
            # Get production parameters for reference r
            c_r = C_r[r]      # Production capacity
            s_r = S_r[r]      # Current stock
            a_r = A_r[r]      # Storage capacity
            ss_dr = SS_dr[r]  # Safety stock requirement
            d_dr = D_dr[r]    # Demand for this day
            
            for s in range(num_slots):
                idx_rs = get_idx(r, s)
                
                # ========== DIAGONAL TERMS (individual variable costs) ==========
                # These represent costs when x_{r,s} = 1 (reference r in slot s)
                
                # H_production contribution:
                # Derived from minimizing (sum_s x_{r,s} * c_r - (ss_dr + d_dr))^2
                # Expands to: A*c_r^2 * x_{r,s} - 2*A*c_r*(ss_dr + d_dr) * x_{r,s}
                production_diagonal = A*c_r**2 - 2*A*c_r*(ss_dr + d_dr)
                
                # H_assignment contribution:
                # Derived from (sum_r x_{r,s} - 1)^2 to enforce one assignment per slot
                # Each variable contributes: -B (constant term absorbed elsewhere)
                assignment_diagonal = -B
                
                # H_storage contribution:
                # Derived from max(0, s_r - d_dr + sum_s x_{r,s}*c_r - a_r)^2
                # Penalizes exceeding storage capacity
                # Expands to: C*c_r^2 * x_{r,s} + 2*C*(s_r - d_dr - a_r)*c_r * x_{r,s}
                storage_diagonal = C*c_r**2 + 2*C*(s_r - d_dr - a_r)*c_r
                
                Q[idx_rs, idx_rs] += (
                    production_diagonal +
                    assignment_diagonal +
                    storage_diagonal
                )
                
                # ========== OFF-DIAGONAL TERMS (variable interactions) ==========
                
                # 1. Same reference, different slots (production and storage interactions)
                # When x_{r,s} = x_{r,s'} = 1, reference r is assigned to both slots
                for s_prime in range(s + 1, num_slots):
                    idx_rs_prime = get_idx(r, s_prime)
                    
                    # H_production interaction: from (x_{r,s} + x_{r,s'})^2 expansion
                    # Contributes: 2*A*c_r^2
                    production_interaction = 2*A*c_r**2
                    
                    # H_storage interaction: from storage constraint expansion
                    # Contributes: 2*C*c_r^2
                    storage_interaction = 2*C*c_r**2
                    
                    base_term = production_interaction + storage_interaction
                    
                    # H_change: encourage production continuity
                    # If slots s and s' are consecutive (s' = s+1), reduce cost
                    # This encourages the same reference to continue in next slot
                    if s_prime == s + 1:
                        base_term += -D  # Negative = reward for continuity
                    
                    # QUBO matrix is symmetric, so fill both (i,j) and (j,i)
                    Q[idx_rs, idx_rs_prime] += base_term
                    Q[idx_rs_prime, idx_rs] += base_term
                
                # 2. Different references, same slot (assignment constraint)
                # When x_{r,s} = x_{r',s} = 1, two references try to use same slot
                # This should be heavily penalized by H_assignment
                for r_prime in range(r + 1, num_refs):
                    idx_r_prime_s = get_idx(r_prime, s)
                    
                    # H_assignment penalty: from (x_{r,s} + x_{r',s} + ... - 1)^2
                    # Cross terms contribute: 2*B (penalizes double-assignment)
                    assignment_penalty = 2*B
                    
                    # Symmetric matrix: fill both directions
                    Q[idx_rs, idx_r_prime_s] += assignment_penalty
                    Q[idx_r_prime_s, idx_rs] += assignment_penalty
    
        return Q, var_mapping

    def generate_maxcut_matrix(self):
        """
        Convert the QUBO matrix into a MaxCut graph formulation.
        
        MaxCut is a graph partitioning problem where nodes are divided into
        two sets to maximize the sum of edge weights between the sets.
        Any QUBO can be transformed into MaxCut by adding an auxiliary node.
        
        Transformation process:
        1. Add auxiliary node (node 0) connected to all QUBO variables
        2. Node 0's edges encode diagonal terms from Q
        3. Edges between other nodes encode off-diagonal terms from Q
        4. Solution: nodes in same partition as node 0 → x_i = 1
                    nodes in opposite partition → x_i = 0
        
        Returns:
            tuple: (maxcut_matrix, var_mapping) where:
                - maxcut_matrix (numpy.ndarray): Symmetric adjacency matrix
                  of shape (total_vars+1, total_vars+1) representing edge
                  weights in the MaxCut graph. Node 0 is auxiliary.
                - var_mapping (dict): Maps (reference, slot) tuples to
                  indices in the original QUBO (node i in MaxCut = index i-1 in QUBO)
        
        Matrix Structure:
            maxcut_matrix[0, i] = sum of row i and column i in Q (diagonal encoding)
            maxcut_matrix[i, j] = Q[i-1,j-1] + Q[j-1,i-1] (off-diagonal encoding)
            All diagonal entries = 0 (no self-loops in graph)
        """

        # Generate the QUBO formulation first
        Q, var_mapping = self.generate_qubo_matrix()
        total_vars = len(Q)

        # Create MaxCut matrix with one extra node (node 0 = auxiliary)
        maxcut_matrix = np.zeros((total_vars + 1, total_vars + 1))

        # Connect auxiliary node (0) to all QUBO variable nodes (1 to total_vars)
        # Edge weight from node 0 to node i encodes the sum of Q's row i and column i
        # This transformation ensures diagonal terms of Q are captured
        for i in range(1, total_vars + 1):
            # Sum all interactions involving variable i-1 in QUBO
            edge_weight = sum(Q[i-1, j] + Q[j, i-1] for j in range(total_vars))
            maxcut_matrix[0, i] = edge_weight
            maxcut_matrix[i, 0] = edge_weight  # Symmetric
        
        # Connect all pairs of QUBO variable nodes
        # Edge weight between nodes i and j encodes Q[i-1,j-1] + Q[j-1,i-1]
        # This captures the off-diagonal interaction terms from QUBO
        for i in range(1, total_vars + 1):
            for j in range(i + 1, total_vars + 1):
                # Sum both directions to handle symmetric QUBO matrix
                edge_weight = Q[i-1, j-1] + Q[j-1, i-1]
                maxcut_matrix[i, j] = edge_weight
                maxcut_matrix[j, i] = edge_weight  # Symmetric
        
        # Ensure no self-loops (diagonal must be zero in MaxCut)
        np.fill_diagonal(maxcut_matrix, 0.0)

        return maxcut_matrix, var_mapping
    
    def export_maxcut_to_mc(self, output_dir):
        """
        Export the MaxCut formulation to a .mc file format.
        
        The .mc format is a standard text format for MaxCut problems used by
        various graph optimization solvers and benchmarking tools.
        
        File format:
            Line 1: <num_nodes> <num_edges>
            Following lines: <node_i> <node_j> <edge_weight>
            (one line per edge, only edges with non-zero weights)
        
        Args:
            output_dir (str): Directory path where the .mc file will be saved.
                            Should end with '/' or '\\'
        
        Returns:
            tuple: (output_file, var_mapping) where:
                - output_file (str): Full path to the created .mc file
                - var_mapping (dict): Maps (reference, slot) tuples to node
                  indices (for interpreting solutions)
        
        Side Effects:
            Creates a text file at output_dir/MaxCut_{num_nodes}.mc
        
        Example file content:
            5 6
            0 1 2.5
            0 2 -1.0
            1 2 3.0
            1 3 0.5
            2 4 1.5
            3 4 -0.5
        """

        # Generate the MaxCut matrix formulation
        maxcut_matrix, var_mapping = self.generate_maxcut_matrix()

        # Number of nodes in the graph (QUBO vars + 1 auxiliary node)
        num_nodes = maxcut_matrix.shape[0]

        # Construct output filename based on problem size
        output_file = output_dir + f"MaxCut_{num_nodes}.mc"

        # Collect all edges (i, j, weight) where i < j and weight != 0
        # Only store upper triangular since matrix is symmetric
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                weight = maxcut_matrix[i, j]
                # Include all edges (even zero weights in standard format)
                # Some solvers may require all edges, others only non-zero
                edges.append((i, j, float(weight)))

        num_edges = len(edges)

        # Write to .mc file in standard format
        with open(output_file, 'w') as f:
            # Header line: number of nodes and edges
            f.write(f"{num_nodes} {num_edges}\n")
            
            # Edge list: one edge per line with format "node_i node_j weight"
            for i, j, weight in edges:
                f.write(f"{i+1} {j+1} {weight}\n")
        
        return output_file, var_mapping