#%%
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
from glestimation.core import PartitioningEstimate
#%%

from folktexts.acs import ACSDataset

from folktexts.acs import ACSTaskMetadata
import pandas as pd
import glest 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.tree import DecisionTreeRegressor

from glest.plot import grouping_diagram_residuals
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from scipy.interpolate import griddata
import glob
import re
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier
import argparse

def load_single_dataset(name):
    datasets = {}
    feature_path = f"merged_datasets_Llama70B_full/{name}_features.csv"
    label_path = f"merged_datasets_Llama70B_full/{name}_labels.csv"
    risk_path = f"merged_datasets_Llama70B_full/{name}_risk_scores.csv"

    if os.path.exists(feature_path) and os.path.exists(label_path) and os.path.exists(risk_path):
        features = pd.read_csv(feature_path)
        labels_df = pd.read_csv(label_path)
        risk_scores = pd.read_csv(risk_path).values[:, 0]

        # Apply specific label encoding if needed, matching the manual loading above
        if name == "completions":
            labels_df = labels_df.replace({"Completed": 1, "Not Completed": 0})
        elif name == "passengers":
            labels_df = labels_df.replace({"satisfied": 1, "neutral or dissatisfied": 0})
        elif name == "rain_in_australia":
            labels_df = labels_df.replace({"Yes": 1, "No": 0})

        datasets[name] = {
            "features": features,
            "labels": labels_df.values[:, 0],
            "risk_scores": risk_scores
        }
        print(f"Loaded dataset: {name}")
    else:
        print(f"Warning: Files not found for dataset {name}")

    return datasets


def load_all_datasets():
    datasets = {}
    feature_files = glob.glob("merged_datasets_Llama70B_full/*_features.csv")
    
    for file_path in feature_files:
        base_name = os.path.basename(file_path).replace("_features.csv", "")
        ds = load_single_dataset(base_name)
        datasets.update(ds)
            
    return datasets


all_datasets = load_all_datasets()
def train_partitioning_estimates(datasets):
    estimators = {}
    for name, data in datasets.items():
        print(f"Processing dataset: {name}")
        features = data['features']
        labels = data['labels']
        risk_scores = data['risk_scores']

        # One-hot encode features if necessary
        # pd.get_dummies is safe to call even if no categorical columns exist (it returns original if all numeric)
        features_encoded = pd.get_dummies(features, drop_first=False)
        
        if features.shape != features_encoded.shape:
            print(f"  Original features shape: {features.shape}")
            print(f"  Encoded features shape: {features_encoded.shape}")

        est = PartitioningEstimate(estimator=DecisionTreeRegressor(min_samples_leaf=30))
        est.fit(
            features_encoded.values,
            risk_scores,
            labels,
        )
        estimators[name] = est
        print(f"  Finished fitting {name} partitioning estimate.")
    
    return estimators




def calculate_f_star_all(datasets, estimators):
    for name, data in datasets.items():
        if name not in estimators:
            continue
            
        est = estimators[name]
        
        # Predict c_hat
        risk_scores = data['risk_scores']
        c_hat = est.calibrator.predict_proba(risk_scores.reshape(-1, 1))[:,1]
        
        # Predict r_hat (ensure encoding matches training)
        features = data['features']
        features_encoded = pd.get_dummies(features, drop_first=False)
        r_hat = est.predict(features_encoded.values)
        
        # Store in dataset
        data['c_hat'] = c_hat
        data['r_hat'] = r_hat
        data['f_star'] = c_hat + r_hat
        print(f"Calculated f_star for {name}")



#%%



def train_and_evaluate_single_dataset(dataset, layer_idx, seed=42):
    X = dataset['hidden_states'].values
    y_f_star = dataset['f_star']
    y_labels = dataset['labels']
    y_risk = dataset['risk_scores']
    
    # Shuffle data
    rng = np.random.default_rng(seed)
    shuffle_idx = rng.permutation(len(X))
    X = X[shuffle_idx]
    y_f_star = y_f_star[shuffle_idx]
    y_labels = y_labels[shuffle_idx]
    y_risk = y_risk[shuffle_idx]
    
    # Split into train/test
    split_idx = int(0.7 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_test_labels = y_labels[split_idx:]
    y_train_labels = y_labels[:split_idx]
    y_test_risk = y_risk[split_idx:]
    

    partest = PartitioningEstimate(estimator=DecisionTreeRegressor(min_samples_leaf=30, random_state=seed))
    partest.fit(
        X_train,
        y_risk[:split_idx],
        y_train_labels,
    )
    c_hat_test = partest.calibrator.predict_proba(y_test_risk.reshape(-1, 1))[:,1]
    r_hat_test = partest.predict(X_test)
    f_star_hat_test = c_hat_test + r_hat_test


    print(f"Layer {layer_idx}: Training on {len(X_train)} samples, Testing on {len(X_test)} samples...")
    
    # Train Random Forest
    nn = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    nn.fit(X_train, y_train_labels)
    
    # Evaluate on test data
    f_star_hat_nn = nn.predict_proba(X_test)[:, 1]
    
    mse_nn = np.mean((f_star_hat_nn - f_star_hat_test)**2)
    brier_nn = np.mean((f_star_hat_nn - y_test_labels)**2)
    brier_risk_score = np.mean((y_test_risk - y_test_labels)**2)
    brier_f_star_hat = np.mean((f_star_hat_test - y_test_labels)**2)
    
    print(f"Layer {layer_idx}: Brier Score (NN) = {brier_nn:.6f}")
    print(f"Layer {layer_idx}: Brier Score (Risk Scores) = {brier_risk_score:.6f}")
    print(f"Layer {layer_idx}: Brier Score (f_star) = {brier_f_star_hat:.6f}")
    print(f"Layer {layer_idx}: MSE = {mse_nn:.6f}")
    
    
    return {
        "layer": layer_idx,
        "mse": mse_nn,
        "brier_nn": brier_nn,
        "brier_risk": brier_risk_score,
        "brier_f_star": brier_f_star_hat
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run layer analysis for a specific test dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="The key of the dataset to use as the test set.")
    parser.add_argument("--output_dir", type=str, default="results_metamodel_single_dataset_Llama70B_full", help="Folder to save the output CSV.")

    args = parser.parse_args()
    
    dataset = args.dataset
    
    if dataset not in all_datasets:
        raise ValueError(f"Dataset '{dataset}' not found")

    dataset_dict = load_single_dataset(dataset)
    estimators = train_partitioning_estimates(dataset_dict)
    calculate_f_star_all(dataset_dict, estimators)

    print(f"--- Starting run for test dataset: {dataset} ---")

    path_dict = {
        "ACSIncome": "folktexts-results-metamodel-Llama70B/model-Llama-3.3-70B-Instruct_task-ACSIncome/Llama-3.3-70B-Instruct_bench-1206599377/ACSIncome_subsampled-0.4_seed-42_hash-1363604979.test_predictions_hidden_states",
        "ACSEmployment": "folktexts-results-metamodel-Llama70B/model-Llama-3.3-70B-Instruct_task-ACSEmployment/Llama-3.3-70B-Instruct_bench-2525682897/ACSEmployment_subsampled-0.2_seed-42_hash-1041950717.test_predictions_hidden_states",
        "satisfaction": "folktexts-results-metamodel-Llama-70B_full/Llama-3.3-70B-Instruct_bench-4169613192/airline passenger satisfaction_subsampled-0.5_seed-42_hash-2223126573.test_predictions_hidden_states",
        "loan_default": "folktexts-results-metamodel-Llama-70B_full/Llama-3.3-70B-Instruct_bench-2130411282/LoanDefault_full_seed-42_hash-4198601886.test_predictions_hidden_states",
        "course_completion": "folktexts-results-metamodel-Llama-70B_full/Llama-3.3-70B-Instruct_bench-1051901534/course completion prediction_subsampled-0.5_seed-42_hash-1848123039.test_predictions_hidden_states",
        "meps": "folktexts-results-metamodel-Llama-70B_full/Llama-3.3-70B-Instruct_bench-2962498681/health-care utilization_full_seed-42_hash-180138447.test_predictions_hidden_states",
        "ACSTravelTime": "folktexts-results-metamodel-Llama70B/model-Llama-3.3-70B-Instruct_task-ACSTravelTime/Llama-3.3-70B-Instruct_bench-3547806008/ACSTravelTime_subsampled-0.4_seed-42_hash-3244950302.test_predictions_hidden_states",
        "ACSMobility": "folktexts-results-metamodel-Llama70B/model-Llama-3.3-70B-Instruct_task-ACSMobility/Llama-3.3-70B-Instruct_bench-3442368737/ACSMobility_subsampled-0.4_seed-42_hash-2590312649.test_predictions_hidden_states",
        # "rain_in_australia" : 'folktexts-results-metamodel-Llama-70B_full/Llama-3.3-70B-Instruct_bench-3147198227/rain prediction in australia_subsampled-0.5_seed-42_hash-685251864.test_predictions_hidden_states',
        "ACSPublicCoverage": 'folktexts-results-metamodel-Llama70B/model-Llama-3.3-70B-Instruct_task-ACSPublicCoverage/Llama-3.3-70B-Instruct_bench-988939989/ACSPublicCoverage_subsampled-0.4_seed-42_hash-1833904006.test_predictions_hidden_states',
        "smoking": "folktexts-results-metamodel-Llama-70B_full/Llama-3.3-70B-Instruct_bench-1085260167/SmokingPrediction_subsampled-0.99_seed-42_hash-1596432086.test_predictions_hidden_states",
        "hotel_booking_cancellations": "folktexts-results-metamodel-Llama-70B_full/Llama-3.3-70B-Instruct_bench-1249158703/HotelBookingCancellation_full_seed-42_hash-1846594368.test_predictions_hidden_states",
        "heart_disease": "folktexts-results-metamodel-Llama-70B_full/Llama-3.3-70B-Instruct_bench-290020217/HeartDiseasePrediction_subsampled-0.15_seed-42_hash-408096573.test_predictions_hidden_states"
    }

    for i in range(0, 81, 5):
        print(f"Processing layer {i}...")
        ds_curr = dataset_dict[dataset].copy()
        file_path = f"{path_dict[dataset]}/layer_{i}.parquet"
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        hidden_state = pq.read_table(file_path).to_pandas()
        if "Unnamed: 0" in hidden_state.columns:
            hidden_state = hidden_state.drop(columns=["Unnamed: 0"])
        ds_curr['hidden_states'] = hidden_state
        
        all_results = []
        for seed in range(5):
            results = train_and_evaluate_single_dataset(ds_curr, i, seed=seed)
            results["seed"] = seed
            all_results.append(results)
        
        os.makedirs(args.output_dir, exist_ok=True)
        pd.DataFrame(all_results).to_csv(f"{args.output_dir}/results_{dataset}_layer{i}.csv", index=False)
    
    print(f"--- Finished {dataset}. Results saved to {args.output_dir} ---")