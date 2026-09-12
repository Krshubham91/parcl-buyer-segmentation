import os
import pandas as pd
import numpy as np

def run_phase_1():
    # -------------------------------------------------------------
    # 1. Load Raw Datasets
    # -------------------------------------------------------------
    print("--> Step 1: Loading raw datasets...")
    
    # Path configuration (adjust if files are in current working directory)
    raw_clients_path = "Clients.csv" if os.path.exists("Clients.csv") else "data/raw/Clients.csv"
    raw_props_path = "Properties.csv" if os.path.exists("Properties.csv") else "data/raw/Properties.csv"

    df_clients = pd.read_csv(raw_clients_path)
    df_props = pd.read_csv(raw_props_path)

    print(f"    Clients raw shape: {df_clients.shape}")
    print(f"    Properties raw shape: {df_props.shape}")

    # -------------------------------------------------------------
    # 2. Data Cleaning & Standardization: Clients
    # -------------------------------------------------------------
    print("\n--> Step 2: Cleaning and engineering client demographic features...")
    
    # Check and handle duplicates
    df_clients = df_clients.drop_duplicates(subset=["client_id"]).copy()

    # Standardize string columns (strip whitespace, normalize casing)
    str_cols = ["client_type", "gender", "country", "region", "acquisition_purpose", "loan_applied", "referral_channel"]
    for col in str_cols:
        df_clients[col] = df_clients[col].astype(str).str.strip()

    # Parse date_of_birth with mixed format handling
    df_clients["date_of_birth"] = pd.to_datetime(df_clients["date_of_birth"], format="mixed")

    # Feature Engineering: Derive 'age' (using base year 2024 from transaction timeline)
    df_clients["age"] = 2024 - df_clients["date_of_birth"].dt.year

    # Feature Engineering: International Buyer Flag (1 if outside USA, 0 if USA)
    df_clients["is_international"] = (df_clients["country"] != "USA").astype(int)

    # Validate client demographic results
    print(f"    Age summary: min={df_clients['age'].min()}, max={df_clients['age'].max()}, mean={df_clients['age'].mean():.1f}")
    print(f"    International buyers: {df_clients['is_international'].sum()} out of {len(df_clients)}")

    # -------------------------------------------------------------
    # 3. Data Cleaning & Aggregation: Properties
    # -------------------------------------------------------------
    print("\n--> Step 3: Cleaning property transactions and aggregating investor metrics...")

    # Filter for sold units associated with clients
    df_sold = df_props[df_props["listing_status"].str.strip() == "Sold"].copy()

    # Clean sale_price: remove '$' and ',' symbols and convert to float
    df_sold["sale_price_clean"] = (
        df_sold["sale_price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    # Aggregating property transactions per investor
    investor_metrics = (
        df_sold.groupby("client_ref")
        .agg(
            total_invested=("sale_price_clean", "sum"),
            avg_unit_price=("sale_price_clean", "mean"),
            properties_bought=("listing_id", "count"),
            total_sqft=("floor_area_sqft", "sum"),
            avg_sqft=("floor_area_sqft", "mean")
        )
        .reset_index()
    )

    print(f"    Unique investors found in sold properties: {len(investor_metrics)}")

    # -------------------------------------------------------------
    # 4. Merge Demographic & Financial Records
    # -------------------------------------------------------------
    print("\n--> Step 4: Merging client profiles with investment aggregates...")

    final_df = pd.merge(
        df_clients,
        investor_metrics,
        left_on="client_id",
        right_on="client_ref",
        how="inner"
    ).drop(columns=["client_ref"])

    # Feature Engineering: Average price per square foot across bought units
    final_df["avg_price_per_sqft"] = (final_df["total_invested"] / final_df["total_sqft"]).round(2)

    # -------------------------------------------------------------
    # 5. Export Master Analytical Dataset
    # -------------------------------------------------------------
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "master_buyer_profiles.csv")
    final_df.to_csv(output_path, index=False)

    print(f"\n[SUCCESS] Phase 1 Complete!")
    print(f"Master dataset saved to: '{output_path}'")
    print(f"Total Records: {final_df.shape[0]} rows, {final_df.shape[1]} columns")
    print("\nSample Preview:")
    preview_cols = [
        "client_id", "client_type", "country", "age",
        "acquisition_purpose", "loan_applied", "properties_bought",
        "total_invested", "avg_unit_price"
    ]
    print(final_df[preview_cols].head(5))

if __name__ == "__main__":
    run_phase_1()