import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_phase_2():
    print("--> Step 1: Master dataset load ho raha hai...")
    input_path = os.path.join("data", "processed", "master_buyer_profiles.csv")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"'{input_path}' file nahi mili. Phase 1 pehle run karein.")
        
    df = pd.read_csv(input_path)
    print(f"    Loaded rows: {df.shape[0]}, columns: {df.shape[1]}")

    # ---------------------------------------------------------
    # 2. Feature Encoding
    # ---------------------------------------------------------
    print("\n--> Step 2: Categorical features encoding...")
    
    # Binary mapping
    df["is_company"] = (df["client_type"] == "Company").astype(int)
    df["is_investment"] = (df["acquisition_purpose"] == "Investment").astype(int)
    df["loan_flag"] = (df["loan_applied"] == "Yes").astype(int)
    
    # One-Hot Encoding for referral channels
    channel_dummies = pd.get_dummies(df["referral_channel"], prefix="channel", drop_first=True, dtype=int)

    # Core numeric & engineered features for clustering
    feature_cols = [
        "age",
        "satisfaction_score",
        "total_invested",
        "avg_unit_price",
        "properties_bought",
        "is_international",
        "is_company",
        "is_investment",
        "loan_flag"
    ]
    
    X = pd.concat([df[feature_cols], channel_dummies], axis=1)
    print(f"    Clustering features ({X.shape[1]}): {list(X.columns)}")

    # ---------------------------------------------------------
    # 3. Feature Scaling (StandardScaler)
    # ---------------------------------------------------------
    print("\n--> Step 3: Normalizing features with StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------------------------------------------------
    # 4. K-Means Clustering & Silhouette Evaluation
    # ---------------------------------------------------------
    print("\n--> Step 4: Optimal cluster training (k=4)...")
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, df["cluster"])
    print(f"    Silhouette Score for k=4: {score:.4f}")

    # ---------------------------------------------------------
    # 5. Segment Label Assignment
    # ---------------------------------------------------------
    print("\n--> Step 5: Assigning business archetypes to clusters...")
    
    # Rule-based business segment profiling
    def label_segment(row):
        if row["is_company"] == 1 or row["properties_bought"] >= 7:
            return "C3: Corporate Buyers"
        elif row["is_international"] == 1:
            return "C1: Global Investors"
        elif row["loan_flag"] == 1 and row["age"] < 45:
            return "C2: First-Time Buyers"
        else:
            return "C4: Luxury / Domestic Investors"

    df["segment_name"] = df.apply(label_segment, axis=1)

    print("\nSegment Distribution:")
    print(df["segment_name"].value_counts())

    # ---------------------------------------------------------
    # 6. Model & Final Data Export
    # ---------------------------------------------------------
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Save artifacts for Streamlit app
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(kmeans, os.path.join(models_dir, "kmeans_model.pkl"))
    joblib.dump(list(X.columns), os.path.join(models_dir, "feature_names.pkl"))
    
    final_output_path = os.path.join("data", "processed", "segmented_buyer_profiles.csv")
    df.to_csv(final_output_path, index=False)

    print(f"\n[SUCCESS] Phase 2 & 3 Complete!")
    print(f"Segmented data saved: '{final_output_path}'")
    print(f"Model saved: '{models_dir}/kmeans_model.pkl'")

if __name__ == "__main__":
    run_phase_2()