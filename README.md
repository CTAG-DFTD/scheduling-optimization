# Production Scheduling Optimization via QUBO/MaxCut

A production scheduling system that transforms multi-objective scheduling problems into Quadratic Unconstrained Binary Optimization (QUBO) and MaxCut graph formulations, enabling the use of quantum annealers, classical solvers, and specialized graph optimization algorithms.

## Overview

This project solves complex production scheduling problems by:
1. Converting scheduling constraints into QUBO mathematical formulations
2. Transforming QUBO into MaxCut graph partitioning problems
3. Solving using various optimization methods (brute force, PuLP, QAOA)
4. Managing multi-day production planning with inventory tracking

The system balances multiple objectives:
- **Production optimization**: Meeting demand and safety stock requirements
- **Assignment constraints**: One reference per time slot
- **Storage capacity**: Respecting warehouse limits
- **Production continuity**: Minimizing changeovers between products

## Project Structure

```
.
├── example.py                          # Main execution script
├── data/
│   ├── data_day_1.csv                 # Initial production data
│   └── data_scaler.pkl                # Normalization scaler
├── src/
│   ├── emb_qubo_generator.py          # QUBO/MaxCut matrix generation
│   ├── maxcut_scheduling_utils.py     # Scheduling orchestration
│   ├── data_preprocessing_functions.py # Data preparation utilities
│   ├── pulp_scheduler.py              # Linear programming solver
│   └── qaoa_scheduler.py              # Quantum-inspired solver
└── results/
    ├── *_production_*.csv             # Production metrics
    ├── *_scheduling_*.csv             # Daily schedules
    └── *.jpg                          # Graph visualizations
```

## Mathematical Formulation

### QUBO Objective Function

The system minimizes a weighted combination of four components:

**H_production**: Penalizes deviation from required production
```
A × Σ_r (Σ_s x_{r,s} × c_r - (ss_{d,r} + d_{d,r}))²
```

**H_assignment**: Enforces one reference per slot
```
B × Σ_s (Σ_r x_{r,s} - 1)²
```

**H_storage**: Penalizes exceeding storage capacity
```
C × Σ_r max(0, s_r - d_{d,r} + Σ_s x_{r,s} × c_r - a_r)²
```

**H_change**: Encourages production continuity
```
D × Σ_r Σ_s,s' |x_{r,s} - x_{r,s+1}|
```

Where:
- `x_{r,s}`: Binary variable (1 if reference r assigned to slot s)
- `c_r`: Production capacity (cycles per hour)
- `s_r`: Current stock level
- `a_r`: Storage capacity limit
- `ss_{d,r}`: Safety stock requirement
- `d_{d,r}`: Demand for day d

### MaxCut Transformation

The QUBO is converted to a MaxCut graph problem using an auxiliary node:
- Node 0 (auxiliary) encodes diagonal terms via edge weights
- Nodes 1 to N represent QUBO variables
- Edge weights between variable nodes encode QUBO interactions
- Solution: nodes in same partition as node 0 → x_i = 1

## Installation

### Requirements

```bash
pip install pandas numpy scikit-learn joblib matplotlib networkx pulp openpyxl
```

### Optional (for QAOA solver)
- Set environment variable: `MAXCUT_QAOA_SCRIPT=/path/to/qaoa_script.py`

## Usage

### Basic Example

```python
from src.maxcut_scheduling_utils import run_scheduling

def maxcut_bruteforce(maxcut_matrix):
    """Your solver implementation"""
    # ... solve the MaxCut problem
    return bitstring_solution, elapsed_time

# Run scheduling
run_scheduling(
    days=7,              # Number of days to schedule
    num_refs=3,          # Number of product references
    num_slots=5,         # Time slots per day
    solver_function=maxcut_bruteforce,
    solver_name="BruteForce"
)
```

### Running the Example

```bash
python example.py
```

This executes a 7-day scheduling optimization with:
- 3 production references
- 5 time slots per day
- Brute force solver

### Data Preprocessing

If you have raw production data:

```python
from src.data_preprocessing_functions import (
    get_raw_data,
    simulate_data,
    normalize_data
)

# Process raw Excel files
get_raw_data(data_dir)

# Simulate inventory dynamics
simulate_data("real_production_data.csv")

# Normalize for optimization
normalize_data("simulated_data.csv")
```

## Input Data Format

The system expects CSV files with the following columns:

| Column | Description |
|--------|-------------|
| `R` | Reference ID |
| `C_r` | Production capacity (cycles/hour) |
| `S_r` | Current stock level |
| `A_r` | Storage capacity limit |
| `SS_d,r` | Safety stock requirement |
| `D_1,r` to `D_N,r` | Demand for days 1 to N |

Example (`data/data_day_1.csv`):

```csv
R,C_r,S_r,A_r,SS_d,r,D_1,r,D_2,r,D_3,r
0,0.45,0.62,0.58,0.32,0.41,0.39,0.43
1,0.38,0.51,0.49,0.28,0.35,0.37,0.36
2,0.52,0.48,0.61,0.35,0.44,0.42,0.45
```

## Available Solvers

### 1. Brute Force (`example.py`)
- Exhaustive search of all possible assignments
- Guaranteed optimal solution
- **Complexity**: O(2^n) - suitable for small problems (n ≤ 20)

### 2. PuLP Linear Programming (`src/pulp_scheduler.py`)
```python
from src.pulp_scheduler import pulp_solver

run_scheduling(
    days=7,
    num_refs=10,
    num_slots=8,
    solver_function=pulp_solver,
    solver_name="PuLP"
)
```
- Uses CBC (COIN-OR Branch and Cut) solver
- Includes partition balance penalty
- Fast for medium-sized problems (n ≤ 100)

### 3. QAOA Quantum-Inspired (`src/qaoa_scheduler.py`)
```python
from src.qaoa_scheduler import maxcut_qaoa

run_scheduling(
    days=7,
    num_refs=20,
    num_slots=10,
    solver_function=maxcut_qaoa,
    solver_name="QAOA"
)
```
- Quantum Approximate Optimization Algorithm
- Requires external QAOA script (set via `MAXCUT_QAOA_SCRIPT` env variable)
- Suitable for large-scale problems

## Output Files

### Production Metrics
`results/{solver_name}_production_{num_slots}_slots_{num_refs}_refs.csv`

Contains daily production data:
- `SS_{day}`: Safety stock shortage
- `S_{day}`: Stock level
- `Golpes_{day}`: Production cycles

### Scheduling Assignments
`results/{solver_name}_scheduling_{num_slots}_slots_{num_refs}_refs.csv`

Contains slot assignments:
```csv
day,ref_slot_1,ref_slot_2,ref_slot_3,ref_slot_4,ref_slot_5
1,0,1,0,2,1
2,1,0,2,0,1
```

### Visualizations
`results/{solver_name}_{nodes}.jpg` or `results/{solver_name}_dia_{day}.jpg`

Graph visualizations showing:
- Node partitions (blue vs. red)
- Cut edges (red) and uncut edges (gray)
- Total cut weight and computation time

## Customization

### Adjusting Objective Weights

Modify weights in `src/emb_qubo_generator.py`:

```python
class EmbQuboGenerator:
    def __init__(self, csv_file, day=1, num_slots=8, num_refs=3):
        self.A = 0.5  # Production objective weight
        self.B = 1.0  # Assignment constraint weight
        self.C = 0.4  # Storage constraint weight
        self.D = 0.05 # Changeover penalty weight
```

### Implementing Custom Solvers

Your solver must accept a MaxCut matrix and return a bitstring:

```python
def my_custom_solver(maxcut_matrix):
    """
    Args:
        maxcut_matrix (numpy.ndarray): Symmetric adjacency matrix
    
    Returns:
        tuple: (bitstring, elapsed_time)
            - bitstring (list[int]): Binary partition [0 or 1] for each node
            - elapsed_time (float): Computation time in seconds
    """
    start_time = time.time()
    
    # Your optimization logic here
    bitstring = your_optimization_algorithm(maxcut_matrix)
    
    elapsed_time = time.time() - start_time
    return bitstring, elapsed_time
```

## Key Features

- **Multi-day planning**: Automatically updates inventory across planning horizon
- **Constraint handling**: Hard constraints via penalty functions in QUBO
- **Solver flexibility**: Easy integration of custom optimization algorithms
- **Data normalization**: Automatic scaling for numerical stability
- **Visualization**: Graph plots for solution analysis
- **Denormalization**: Results converted back to original scale

## Performance Considerations

| Solver | Best for | Time Complexity | Problem Size |
|--------|----------|----------------|--------------|
| Brute Force | Small problems, verification | O(2^n) | n ≤ 20 |
| PuLP | Medium problems, speed | O(n³) typical | n ≤ 100 |
| QAOA | Large problems, approximation | Problem-dependent | n > 100 |

Where n = num_refs × num_slots

## Troubleshooting

### Issue: "Environment variable MAXCUT_QAOA_SCRIPT is not set"
**Solution**: Set the environment variable before using QAOA solver:
```bash
export MAXCUT_QAOA_SCRIPT=/path/to/your/qaoa_script.py
```

### Issue: Solver produces invalid assignments (multiple refs per slot)
**Solution**: Increase weight `B` in `EmbQuboGenerator` to strengthen assignment constraints

### Issue: Poor solution quality
**Solution**: 
- Adjust objective weights (A, B, C, D)
- Increase solver iterations/time
- Try different solvers

### Issue: Memory error with large problems
**Solution**: 
- Reduce `num_refs` or `num_slots`
- Use sparse matrix representations
- Switch to iterative solvers (PuLP, QAOA)

## Acknowledgements

![](images/Cartel-PCCC-800x571.png)