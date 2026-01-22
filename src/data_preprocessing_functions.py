import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def clean_raw_data(path):
    df = pd.read_excel(f"{path}")
    df = df.drop(index=[0,1,2])
    df.columns = df.iloc[0]
    df = df[1:-1]
    df = df.reset_index(drop=True)

    return df

def get_raw_data(data_dir):
    # Read dataset directory paths
    data_file = os.listdir(data_dir)

    # --- 2. Processing and consolidation ---
    processed_list = []

    # Iterate over the list using enumerate to get an index (starting at 1)
    for i, file in enumerate(data_file, 1):

        df = clean_raw_data(f"{data_dir}\\{file}")

        # Select and rename columns and set 'Referencia' as index
        processed_df = df[['Referencia', 'Golpes']] \
            .rename(columns={'Golpes': f'Golpes_{i-2}'}) \
            .set_index('Referencia')
        
        # Add the processed DataFrame to the list
        processed_list.append(processed_df)

    # --- 3. Final merge ---
    # Concatenate all DataFrames horizontally (axis=1)
    final_df = pd.concat(processed_list, axis=1)

    # Convert 'Referencia' index back to a column
    final_df = final_df.reset_index()

    # --- 4. Target processing ---
    targets_df = clean_raw_data("Grupo de Produccion_ Linea 1.xlsx")

    targets_df = targets_df.rename(columns={
        'Flan  o Referencias': 'Referencia_Grupo',
        'Objetivo BD (enero)': '%OK'
    })

    targets_df = targets_df[["Referencia_Grupo", "CHR", "%OK"]]

    # Store the original group before exploding
    targets_df['Grupo_Original'] = targets_df['Referencia_Grupo']

    # Split grouped references by " - " into a list
    targets_df['Referencia_Grupo'] = targets_df['Referencia_Grupo'].str.split(' - ')

    # Explode the df to have one row per reference in the lists
    # 'CHR', '%OK', and 'Grupo_Original' will be duplicated for each member
    reference_map = targets_df.explode('Referencia_Grupo')

    # Clean whitespace
    reference_map['Referencia_Grupo'] = reference_map['Referencia_Grupo'].str.strip()

    # --- 5. Merge and deduplication ---
    # Perform an inner join to keep only the references in final_df that DO have data in reference_map.
    # Use 'Referencia' (from final_df) and 'Referencia_Grupo' (from reference_map) as keys
    merged_df = pd.merge(
        final_df,
        reference_map,
        left_on='Referencia',
        right_on='Referencia_Grupo',
        how='inner'
    )

    # --- 6. Final filtering ---
    # If there are multiple references from the same group, keep only the first one.
    # Use 'Grupo_Original' to identify group members.

    final_filtered_df = merged_df.drop_duplicates(
        subset=['Grupo_Original'],
        keep='first'
    )

    # --- 7. Cleanup ---
    # Drop auxiliary columns no longer needed
    final_filtered_df = final_filtered_df.drop(
        columns=['Referencia_Grupo', 'Grupo_Original']
    )

    final_filtered_df = final_filtered_df.fillna(0).reset_index(drop=True)

    # Save CSV with real production data
    final_filtered_df.to_csv("real_production_data.csv", index=False)

def simulate_data(data_path):

    # Load the original CSV
    df = pd.read_csv(data_path)
    df["Referencia"] = [f"{i}" for i in range(len(df))]

    # Identify production columns (Golpes)
    golpes_cols = [col for col in df.columns if col.startswith("Golpes_")]
    n_dias = len(golpes_cols) - 2

    # Function to simulate stock, demand, and safety stock
    def simulate_stock_and_demand(row):
        produccion = row[golpes_cols].values.astype(int)

        # Initial stock equals the production of the first two days
        stock_inicial = produccion[:2].sum()
        
        S = [stock_inicial]  # Initial stock at the start of day 1
        D = []
        SS = []

        for i in range(n_dias):
            demanda_min = int(0.9 * produccion[i])
            demanda_max = min(S[i] + produccion[i + 2], int(1.1 * produccion[i]))

            # Random demand between 90% and 110% of forecast based on previous two days, avoiding stockout
            np.random.seed(42)
            demanda = np.random.randint(demanda_min, demanda_max + 1) if demanda_max > demanda_min else 0
            D.append(demanda)

            # Calculate stock at the start of the next day (non-negative)
            nuevo_stock = S[i] + produccion[i + 2] - demanda
            if i < n_dias - 1:
                S.append(nuevo_stock)

        # Calculate safety stock (SS_i = D_{i+1} + D_{i+2} - S_{i})
        for i in range(n_dias):
            if i <= n_dias - 3:
                ss = D[i + 1] + D[i + 2] - S[i]
                ss = ss if ss > 0 else 0
            else:
                ss = 0  # last days
            SS.append(ss)

        # Return as a dictionary to add to the dataframe
        resultados = {}
        for i in range(n_dias):
            resultados[f"S_{i+1}"] = S[i]
            resultados[f"D_{i+1}"] = D[i]
            resultados[f"SS_{i+1}"] = SS[i]

        return pd.Series(resultados)

    # Apply simulation per reference
    sim_df = df.apply(simulate_stock_and_demand, axis=1)

    # ---- Calculate total storage capacity ----
    stock_cols = [col for col in sim_df.columns if col.startswith("S_")]

    # Capacity = maximum stock reached per reference + 10%
    df["Capacidad_almacenamiento"] = (sim_df[stock_cols].max(axis=1) * 1.05).astype(int)

    # Merge results with the original dataframe
    df_final = pd.concat([df, sim_df], axis=1)

    # Save CSV with simulated data
    drop_cols = golpes_cols + ["%OK"]
    df_final = df_final.drop(axis=1, columns=drop_cols)
    df_final.to_csv("simulated_data.csv", index=False)

def normalize_data(data_path): 
    df = pd.read_csv(data_path)

    # Normalize all values together
    numeric_cols = [col for col in df.select_dtypes(include='number').columns]
    scaler = MinMaxScaler()
    values = df[numeric_cols].values
    values_reshaped = values.reshape(-1, 1)
    values_normalized = scaler.fit_transform(values_reshaped) 
    df[numeric_cols] = values_normalized.reshape(values.shape)

    # Save scaler
    joblib.dump(scaler, 'data_scaler.pkl')

    # Save CSV with simulated data normalized
    df.to_csv("simulated_data_normalized.csv", index=False)

    # Rename columns
    df = df[['Referencia', 'CHR', 'S_1', 'Capacidad_almacenamiento', 'SS_1', 'D_1', 'D_2', 'D_3', 'D_4', 'D_5', 'D_6', 'D_7']]
    column_replace = {f"D_{i}": f"D_{i},r" for i in range(1,16)} | {'Referencia': 'R', 'Capacidad_almacenamiento': 'A_r', 'CHR': 'C_r', 'S_1': 'S_r', 'SS_1': 'SS_d,r'}
    df = df.rename(columns=column_replace)

    # Save CSV with the data for the model
    df.to_csv("data_day_1.csv", index=False)