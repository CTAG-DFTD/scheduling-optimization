"""
Production Scheduling System using QUBO/MaxCut Optimization

This module implements a multi-day production scheduling system that converts
scheduling problems into MaxCut graph problems, solves them using desire solver, 
and manages production data across multiple days.

The system handles:
- Converting MaxCut bitstring solutions to QUBO format
- Mapping solutions to production assignments (reference-slot pairs)
- Tracking inventory, security stocks, and production across days
- Saving scheduling results and production metrics
"""

import pandas as pd
import os
from src.emb_qubo_generator import EmbQuboGenerator
from collections import Counter
import joblib
import matplotlib.pyplot as plt
import networkx as nx

def maxcut_bitstring_to_qubo(bitstring_maxcut):
    """
    Convert a MaxCut bitstring solution to QUBO format.
    
    In MaxCut, nodes are partitioned into two sets. This function converts
    that partition into a QUBO representation where node 0 is the reference,
    and other nodes are encoded as 1 if they're in the same set as node 0,
    or 0 if they're in the opposite set.
    
    Args:
        bitstring_maxcut (list): Binary solution from MaxCut solver where
                                 bitstring_maxcut[i] indicates which partition
                                 node i belongs to
    
    Returns:
        list: QUBO bitstring where each bit indicates whether node i is in
              the same partition as node 0
    """
    node0 = bitstring_maxcut[0]
    bitstring_qubo = []

    for i in range(1, len(bitstring_maxcut)):
        bit_i = bitstring_maxcut[i]
        # Set bit to 1 if node i is in same partition as node 0
        bitstring_i = int(bit_i == node0)              
        bitstring_qubo.append(bitstring_i)
    
    return bitstring_qubo


def get_assigment_solution(var_mapping, bitstring_maxcut):
    """
    Map QUBO solution bitstring to production assignments.
    
    Converts the binary solution into a list of (reference, slot) tuples
    representing which production references are assigned to which time slots.
    
    Args:
        var_mapping (dict): Dictionary mapping (reference, slot) tuples to
                           their index positions in the bitstring
        bitstring_maxcut (list): Binary solution from MaxCut solver
    
    Returns:
        list: List of (reference, slot) tuples representing the production
              schedule assignments where bitstring value is 1
    """
    bitstring_qubo = maxcut_bitstring_to_qubo(bitstring_maxcut)

    assigment_solution = []
    for (r, s), idx in var_mapping.items():
        # Only include assignments where the bit is 1 (assigned)
        if bitstring_qubo[idx] == 1:
            assigment_solution.append((r, s))
    
    return assigment_solution


def get_production_ref_day(assigment_solution, day_data_path):
    """
    Calculate total production for each reference on a given day.
    
    Aggregates the number of slots assigned to each reference and multiplies
    by the production rate (CHR - Cycles per Hour Rate) to get total production.
    
    Args:
        assigment_solution (list): List of (reference, slot) assignment tuples
        day_data_path (str): Path to CSV file containing production data for the day
    
    Returns:
        Counter: Dictionary-like object mapping reference ID to total production
                 (CHR * number of slots assigned)
    """
    # Count how many slots each reference is assigned
    production_ref_day = Counter(x[0] for x in assigment_solution)
    df = pd.read_csv(day_data_path)

    for ref, slots in production_ref_day.items():
        # Calculate total production: cycles per hour × number of slots
        total_production = df.iloc[ref]["C_r"] * slots
        production_ref_day[ref] = total_production
        
    return production_ref_day


def update_entry_data(assigment_solution, num_refs, day, day_data_path, next_day_data_path):
    """
    Update inventory and security stock data for the next day based on current day's production.
    
    This function implements the inventory balance equation:
    New Stock (S_r) = Previous Stock + Production - Demand
    
    It also calculates security stock (SS_d,r) by checking if projected
    demand exceeds available stock.
    
    Args:
        assigment_solution (list): List of (reference, slot) assignment tuples
        num_refs (int): Number of production references to process
        day (int): Current day number
        num_slots (int): Number of production slots available
        day_data_path (str): Path to current day's data CSV
        next_day_data_path (str): Path where next day's updated data will be saved
    
    Side Effects:
        Creates/updates CSV file at next_day_data_path with updated inventory
        and security stock values
    """
    df = pd.read_csv(day_data_path)
    df = df[:num_refs]

    # Calculate total production for current day
    production_ref_day = get_production_ref_day(assigment_solution, day_data_path)

    for ref, ref_production in production_ref_day.items():
        # Update stock: S_r = previous_stock - demand_today + production_today
        df.loc[ref, "S_r"] = df.iloc[ref]["S_r"] - df.iloc[ref][f"D_{day},r"] + ref_production
        
        try:
            # Calculate security stock shortage for next 2 days: (D_tomorrow + D_day_after) - S_r
            ss = df.iloc[ref][f"D_{day + 1},r"] + df.iloc[ref][f"D_{day + 2},r"] - df.iloc[ref]["S_r"]
            # security stock shortage is 0 if we have enough stock (negative ss)
            df.loc[ref, "SS_d,r"] = ss if ss > 0.0 else 0.0
        except:
            # If future demand columns don't exist (end of planning horizon)
            df.loc[ref, "SS_d,r"] = 0.0
    
    # Save updated data for next day
    df.to_csv(next_day_data_path, index=False)

    
def save_production_per_day(day, assigment_solution, day_data_path, production_data_path):
    """
    Save daily production metrics to a cumulative production tracking file.
    
    Records security stock shortage, stock levels, and actual production (Golpes/hits)
    for each reference on the given day. Merges with existing production history.
    
    Args:
        day (int): Current day number
        num_slots (int): Number of production slots available
        assigment_solution (list): List of (reference, slot) assignment tuples
        day_data_path (str): Path to current day's data CSV
        production_data_path (str): Path to cumulative production tracking CSV
    
    Side Effects:
        Creates or updates CSV file at production_data_path with columns:
        - R: Reference ID
        - SS_{day}: security stock shortage for this day
        - S_{day}: Stock level for this day
        - Golpes_{day}: Production hits/cycles for this day
    """
    df_day_production = pd.read_csv(day_data_path)

    # Rename columns to include day number for tracking across days
    df_day_production = df_day_production.rename(columns={
        "SS_d,r": f"SS_{day}", 
        "S_r": f"S_{day}"
    })
    # Initialize production column for this day
    df_day_production[f"Golpes_{day}"] = 0.0

    # Get actual production for each reference
    production_ref_day = get_production_ref_day(assigment_solution, day_data_path)

    for ref, ref_production in production_ref_day.items():
        df_day_production.loc[ref, f"Golpes_{day}"] = ref_production
    
    try:
        # Try to load existing production history
        df_total_production = pd.read_csv(production_data_path)
    except:
        # If file doesn't exist, create new dataframe with reference IDs
        df_total_production = df_day_production[["R"]].copy()

    # Merge current day's data with historical data
    df_total_production = df_total_production.merge(df_day_production)

    df_total_production.to_csv(production_data_path, index=False)


def save_scheduling(day, num_slots, assigment_solution, scheduling_data_path):
    """
    Save the daily production schedule to a CSV file.
    
    Records which reference is assigned to each slot for the given day.
    Each row represents one day with columns for each time slot.
    
    Args:
        day (int): Current day number
        num_slots (int): Number of production slots available
        assigment_solution (list): List of (reference, slot) assignment tuples
        scheduling_data_path (str): Path to scheduling CSV file
    
    Side Effects:
        Creates or updates CSV file with columns:
        - day: Day number
        - ref_slot_1, ref_slot_2, ..., ref_slot_N: Reference ID assigned to each slot
    """
    try:
        df = pd.read_csv(scheduling_data_path)
    except:
        # Create new dataframe if file doesn't exist
        columns = ["day"] + [f"ref_slot_{s}" for s in range(1, num_slots + 1)]
        df = pd.DataFrame(columns=columns)
    
    # Sort assignments by slot number to maintain order
    assigment_solution = sorted(assigment_solution, key=lambda x: x[1])
    # Extract just the reference IDs in slot order
    assigment_solution = [slot_assigment[0] for slot_assigment in assigment_solution]
    
    # Create row: [day, ref_in_slot_1, ref_in_slot_2, ...]
    row = [day] + assigment_solution
    
    df.loc[len(df)] = row
    df.to_csv(scheduling_data_path, index=False)


def denormalize_final_data(scaler_pkl_path, data_csv_path):
    """
    Reverse normalization of production data to restore original values.
    
    During preprocessing, data may have been normalized for optimization.
    This function applies the inverse transformation to convert normalized
    values back to their original scale using a saved scaler.
    
    Args:
        scaler_pkl_path (str): Path to pickled sklearn scaler object
        data_csv_path (str): Path to CSV file with normalized data
    
    Side Effects:
        Overwrites the CSV file at data_csv_path with denormalized values
    """
    # Load the scaler that was used during normalization
    scaler = joblib.load(scaler_pkl_path)

    # Read the normalized data
    df_normalized = pd.read_csv(data_csv_path)

    # Identify all numeric columns (exclude text columns like reference IDs)
    numeric_cols = [col for col in df_normalized.select_dtypes(include='number').columns]

    # Flatten all numeric values into a single column for inverse transform
    values_normalized = df_normalized[numeric_cols].values
    values_normalized_reshaped = values_normalized.reshape(-1, 1)

    # Apply inverse transformation to get original scale
    values_original_reshaped = scaler.inverse_transform(values_normalized_reshaped)
    # Reshape back to original dataframe structure and convert to integers
    values_original = values_original_reshaped.reshape(values_normalized.shape).astype(int)

    # Create copy and update numeric columns with denormalized values
    df_original = df_normalized.copy()
    df_original[numeric_cols] = values_original

    # Save denormalized data
    df_original.to_csv(data_csv_path, index=False)

def plot_graph(maxcut_matrix, bitstring, elapsed_time, title, file):
            bitstring = [int(bit) for bit in bitstring]
            n = len(bitstring)

            G = nx.Graph()
            for i in range(n):
                for j in range(i + 1, n):
                    if maxcut_matrix[i][j] != 0:
                        G.add_edge(i, j, weight=maxcut_matrix[i][j])

            group1 = [i for i in range(n) if bitstring[i] == 0]
            group2 = [i for i in range(n) if bitstring[i] == 1]
            cut_edges = [(i, j) for i, j in G.edges() if bitstring[i] != bitstring[j]]
            uncut_edges = [(i, j) for i, j in G.edges() if bitstring[i] == bitstring[j]]
            total_cut_weight = sum(G[i][j]['weight'] for i, j in cut_edges)

            pos = nx.spring_layout(G, seed=42)
            plt.figure(figsize=(9, 7))
            nx.draw_networkx_nodes(G, pos, nodelist=group1, node_color="skyblue", label="Grupo 1", node_size=300)
            nx.draw_networkx_nodes(G, pos, nodelist=group2, node_color="salmon", label="Grupo 2", node_size=300)
            nx.draw_networkx_edges(G, pos, edgelist=cut_edges, edge_color="red", width=2, label="Corte")
            nx.draw_networkx_edges(G, pos, edgelist=uncut_edges, edge_color="gray", alpha=0.3)
            nx.draw_networkx_labels(G, pos, font_size=8)
            plt.title(f"{title}\nPesos del corte: {total_cut_weight} | Tiempo: {elapsed_time:.2f}s", fontsize=12)
            plt.legend()
            plt.axis("off")

            # Guardar imagen en results
            results_dir = os.path.join(os.getcwd(), "results")
            filename = os.path.join(results_dir, f"{file}.jpg")
            plt.savefig(filename, format="jpg")
            plt.close()

def run_scheduling(days, num_refs, num_slots, solver_function, solver_name):
    """
    Execute the complete multi-day production scheduling optimization process.
    
    This is the main orchestration function that:
    1. For each day, generates a MaxCut optimization problem from production data
    2. Solves the optimization problem using the provided solver
    3. Converts the solution to production assignments
    4. Updates inventory and security stock data for the next day
    5. Records production metrics and scheduling decisions
    6. Finally, denormalizes all recorded data to original scale
    
    Args:
        days (int): Number of days to schedule
        num_refs (int): Number of production references (products/SKUs) to schedule
        num_slots (int): Number of production time slots available per day
        solver_function (callable): Function that takes a MaxCut matrix and returns
                                   a bitstring solution. Should have signature:
                                   solver_function(maxcut_matrix) -> list[int]
        solver_name (str): Name identifier for the solver (used in output filenames)
    
    Side Effects:
        Creates/updates multiple CSV files:
        - data/data_day_{day+1}.csv: Updated data for each subsequent day
        - results/{solver_name}_production_{num_slots}_slots_{num_refs}_refs.csv:
          Cumulative production metrics across all days
        - results/{solver_name}_scheduling_{num_slots}_slots_{num_refs}_refs.csv:
          Daily scheduling assignments
    
    Example:
        >>> def my_solver(matrix):
        ...     # Custom solver implementation
        ...     return bitstring_solution, elapsed_time
        >>> 
        >>> run_scheduling(
        ...     days=30,
        ...     num_refs=50,
        ...     num_slots=10,
        ...     solver_function=my_solver,
        ...     solver_name="quantum_annealer"
        ... )
    """
    data_dir = os.path.join(os.getcwd(), "data")
    results_dir = os.path.join(os.getcwd(), "results")
    production_data_file = f'{solver_name}_production_{num_slots}_slots_{num_refs}_refs.csv'
    scheduling_data_file = f'{solver_name}_scheduling_{num_slots}_slots_{num_refs}_refs.csv'
    production_data_path = os.path.join(results_dir, production_data_file)
    scheduling_data_path = os.path.join(results_dir, scheduling_data_file)
    scaler_pkl_path = os.path.join(data_dir, "data_scaler.pkl")
    
    if production_data_file in os.listdir(results_dir):
        os.remove(production_data_path)

    if scheduling_data_file in os.listdir(results_dir):
        os.remove(scheduling_data_path)

    nodes = num_refs * num_slots + 1

    for day in range(1, days + 1):    
        if days == 1:
            title = f"Max-cut {nodes} nodos ({solver_name})"
            file = f"{solver_name}_{nodes}"
        else:
            title = f"Max-cut día {day} ({solver_name})"
            file = f"{solver_name}_dia_{day}"
        
        # Set up file paths
        day_data_path = os.path.join(data_dir, f'data_day_{day}.csv')
        next_day_data_path = os.path.join(data_dir, f'data_day_{day + 1}.csv')

        # Generate MaxCut optimization problem from current day's production data
        maxcut_matrix, var_mapping = EmbQuboGenerator(
            day_data_path, 
            day=day, 
            num_slots=num_slots, 
            num_refs=num_refs
        ).generate_maxcut_matrix()
        
        # Solve the MaxCut problem using the provided solver
        bitstring_maxcut, elapsed_time = solver_function(maxcut_matrix)

        # Plot graph
        plot_graph(maxcut_matrix, bitstring_maxcut, elapsed_time, title, file)
        
        # Convert solution to production assignments (reference, slot) pairs
        assigment_solution = get_assigment_solution(var_mapping, bitstring_maxcut)
        
        # Update inventory and security stock data for next day
        update_entry_data(assigment_solution, num_refs, day, day_data_path, next_day_data_path)
        
        # Record production metrics for this day
        save_production_per_day(day, assigment_solution, day_data_path, production_data_path)
        
        # Record scheduling assignments for this day
        save_scheduling(day, num_slots, assigment_solution, scheduling_data_path)

        # Print control message
        print(f"Day {day} of {days} scheduled")
    
    # After all days are processed, convert normalized data back to original scale
    denormalize_final_data(scaler_pkl_path, production_data_path)