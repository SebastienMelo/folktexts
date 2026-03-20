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

def load_all_datasets():
    datasets = {}
    feature_files = glob.glob("merged_datasets_Llama70B_full/*_features.csv")
    
    for file_path in feature_files:
        base_name = os.path.basename(file_path).replace("_features.csv", "")
        
        feature_path = f"merged_datasets_Llama70B_full/{base_name}_features.csv"
        label_path = f"merged_datasets_Llama70B_full/{base_name}_labels.csv"
        risk_path = f"merged_datasets_Llama70B_full/{base_name}_risk_scores.csv"
        
        if os.path.exists(feature_path) and os.path.exists(label_path) and os.path.exists(risk_path):
            features = pd.read_csv(feature_path)
            labels_df = pd.read_csv(label_path)
            risk_scores = pd.read_csv(risk_path).values[:, 0]
            
            # Apply specific label encoding if needed, matching the manual loading above
            if base_name == "completions":
                labels_df = labels_df.replace({"Completed": 1, "Not Completed": 0})
            elif base_name == "passengers":
                labels_df = labels_df.replace({"satisfied": 1, "neutral or dissatisfied": 0})
            elif base_name == "rain_in_australia":
                labels_df = labels_df.replace({"Yes": 1, "No": 0})
            elif base_name == "heart":
                labels_df = labels_df.replace({"Yes": 1, "No": 0})
            elif base_name == "booking":
                labels_df = labels_df.replace({"Canceled": 1, "Not_Canceled": 0})

            
            datasets[base_name] = {
                "features": features,
                "labels": labels_df.values[:, 0],
                "risk_scores": risk_scores
            }
            print(f"Loaded dataset: {base_name}")
            
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

partitioning_estimates = train_partitioning_estimates(all_datasets)



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

calculate_f_star_all(all_datasets, partitioning_estimates)


#%%



#%%
results = []

def train_and_evaluate_loop(datasets, test_name, layer_idx):
        # Select training datasets (all except the test one)
        train_names = [name for name in datasets if name != test_name]
        
        # Concatenate training data
        X_train = np.concatenate([datasets[name]['hidden_states'].values for name in train_names], axis=0)
        print(X_train.shape)
        
        risk_train = np.concatenate([datasets[name]['risk_scores'] for name in train_names], axis=0)
        X_train = np.concatenate([X_train, risk_train.reshape(-1, 1)], axis=1)

        # Update test hidden_states to include risk scores so dimensions match during prediction
        datasets[test_name]['hidden_states']['risk_score'] = datasets[test_name]['risk_scores']

        y_train = np.concatenate([datasets[name]['labels'] for name in train_names], axis=0)
        

        # Shuffle training data
        shuffle_idx = np.random.permutation(len(X_train))
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]
        
        print(f"Layer {layer_idx}: Training on {train_names}, Testing on {test_name}...")
        
        # Train Random Forest
        nn = HistGradientBoostingClassifier(max_iter=500)
        print(f"Layer {layer_idx}: Training HBG Classifier.")

        nn.fit(X_train, y_train)

        # Evaluate on test data
        X_test = datasets[test_name]['hidden_states'].values
        
        # y_test_f_star = datasets[test_name]['f_star']
        y_test_labels = datasets[test_name]['labels']
        y_test_risk = datasets[test_name]['risk_scores']
        
        X_test_train, X_test_eval, y_test_train, y_test_test, y_test_risk_train, y_test_risk_test = train_test_split(
            X_test,
            y_test_labels,
            y_test_risk,
            test_size=0.7,
            random_state=42,
            shuffle=True
        )

        partest = PartitioningEstimate(estimator=DecisionTreeRegressor(min_samples_leaf=30))
        partest.fit(
            X_test_train,
            y_test_risk_train,
            y_test_train,
        )
        print(f"Layer {layer_idx}: Fitted Partitioning Estimate on test training split.")
        y_test_f_star = partest.calibrator.predict_proba(y_test_risk_test.reshape(-1,1))[:,1] + partest.predict(X_test_eval)


        f_star_hat_nn = nn.predict_proba(X_test_eval)[:,1]
        print(f"Layer {layer_idx}: Made predictions on test evaluation split.")
        mse_nn = np.mean((f_star_hat_nn - y_test_f_star)**2)
        # print(1)
        brier_nn = np.mean((f_star_hat_nn - y_test_test)**2)
        # print(2)
        brier_risk_score = np.mean((y_test_risk_test - y_test_test)**2)
        # print(3)
        brier_f_star_hat = np.mean((y_test_f_star - y_test_test)**2)
        
        print(f"Layer {layer_idx}: Test Set: {test_name}")
        print(f"Layer {layer_idx}: Brier Score (NN) = {brier_nn:.6f}")
        print(f"Layer {layer_idx}: Brier Score (Risk Scores) = {brier_risk_score:.6f}")
        print(f"Layer {layer_idx}: Brier Score (f_star) = {brier_f_star_hat:.6f}")
        print(f"Layer {layer_idx}: MSE = {mse_nn:.6f}")
        
        return {
            "layer": layer_idx,
            "test_dataset": test_name,
            "mse": mse_nn,
            "brier_nn": brier_nn,
            "brier_risk": brier_risk_score,
            "brier_f_star": brier_f_star_hat
        }



def run_layer_analysis(test_dataset_name):
    # Define mapping from dataset keys (from filenames) to task substring in directory names
    # key_to_task_map = {
    #     'ACSIncome': 'ACSIncome',
    #     'ACSEmployment': 'ACSEmployment',
    #     'ACSMobility': 'ACSMobility',
    #     'ACSPublicCoverage': 'ACSPublicCoverage',
    #     'ACSTravelTime': 'ACSTravelTime',
    #     'meps': 'health-care utilization',
    #     'completions': 'course completion prediction',
    #     'passengers': 'airline passenger satisfaction',
    #     'rain_in_australia': 'rain prediction in australia',
    #     "loan_default": "LoanDefault",

    # }

    parquet_map = {}
    base_results_dir = "folktexts-results-metamodel-Llama70B_full/"
    
    # print("Locating hidden states parquet files...")
    # for ds_key, task_pattern in key_to_task_map.items():
    #     if ds_key not in all_datasets:
    #         continue
            
    #     # Search for directory matching the task pattern
    #     # Directory structure: .../bench_id/task_name_full_.../layer_{}.parquet
    #     search_pattern = os.path.join(base_results_dir, "*", f"{task_pattern}*.test_predictions_hidden_states")
    #     candidates = glob.glob(search_pattern)
        
    #     if candidates:
    #         # Use the first match found
    #         parquet_map[ds_key] = os.path.join(candidates[0], "layer_{}.parquet")
    #         print(f"  Mapped {ds_key} -> {candidates[0]}")
    #     else:
    #         print(f"  Warning: Could not find hidden states for {ds_key} (pattern: {task_pattern})")
    
    parquet_map = {
        "ACSIncome": "folktexts-results-metamodel-Llama70B/model-Llama-3.3-70B-Instruct_task-ACSIncome/Llama-3.3-70B-Instruct_bench-1206599377/ACSIncome_subsampled-0.4_seed-42_hash-1363604979.test_predictions_hidden_states/layer_{}.parquet",
        "ACSEmployment": "folktexts-results-metamodel-Llama70B/model-Llama-3.3-70B-Instruct_task-ACSEmployment/Llama-3.3-70B-Instruct_bench-2525682897/ACSEmployment_subsampled-0.2_seed-42_hash-1041950717.test_predictions_hidden_states/layer_{}.parquet",
        "satisfaction": "folktexts-results-metamodel-Llama-70B_full/Llama-3.3-70B-Instruct_bench-4169613192/airline passenger satisfaction_subsampled-0.5_seed-42_hash-2223126573.test_predictions_hidden_states/layer_{}.parquet",
        "loan_default": "folktexts-results-metamodel-Llama-70B_full/Llama-3.3-70B-Instruct_bench-2130411282/LoanDefault_full_seed-42_hash-4198601886.test_predictions_hidden_states/layer_{}.parquet",
        "course_completion": "folktexts-results-metamodel-Llama-70B_full/Llama-3.3-70B-Instruct_bench-1051901534/course completion prediction_subsampled-0.5_seed-42_hash-1848123039.test_predictions_hidden_states/layer_{}.parquet",
        "meps": "folktexts-results-metamodel-Llama-70B_full/Llama-3.3-70B-Instruct_bench-2962498681/health-care utilization_full_seed-42_hash-180138447.test_predictions_hidden_states/layer_{}.parquet",
        "ACSTravelTime": "folktexts-results-metamodel-Llama70B/model-Llama-3.3-70B-Instruct_task-ACSTravelTime/Llama-3.3-70B-Instruct_bench-3547806008/ACSTravelTime_subsampled-0.4_seed-42_hash-3244950302.test_predictions_hidden_states/layer_{}.parquet",
        "ACSMobility": "folktexts-results-metamodel-Llama70B/model-Llama-3.3-70B-Instruct_task-ACSMobility/Llama-3.3-70B-Instruct_bench-3442368737/ACSMobility_subsampled-0.4_seed-42_hash-2590312649.test_predictions_hidden_states/layer_{}.parquet",
        "rain_in_australia" : 'folktexts-results-metamodel-Llama-70B_full/Llama-3.3-70B-Instruct_bench-3147198227/rain prediction in australia_subsampled-0.5_seed-42_hash-685251864.test_predictions_hidden_states/layer_{}.parquet',
        "ACSPublicCoverage": 'folktexts-results-metamodel-Llama70B/model-Llama-3.3-70B-Instruct_task-ACSPublicCoverage/Llama-3.3-70B-Instruct_bench-988939989/ACSPublicCoverage_subsampled-0.4_seed-42_hash-1833904006.test_predictions_hidden_states.layer_{}.parquet',
        "smoking": "folktexts-results-metamodel-Llama-70B_full/Llama-3.3-70B-Instruct_bench-1085260167/SmokingPrediction_subsampled-0.99_seed-42_hash-1596432086.test_predictions_hidden_states/layer_{}.parquet",
        "hotel_booking_cancellations": "folktexts-results-metamodel-Llama-70B_full/Llama-3.3-70B-Instruct_bench-1249158703/HotelBookingCancellation_full_seed-42_hash-1846594368.test_predictions_hidden_states/layer_{}.parquet",
        "heart_disease": "folktexts-results-metamodel-Llama-70B_full/Llama-3.3-70B-Instruct_bench-290020217/HeartDiseasePrediction_subsampled-0.15_seed-42_hash-408096573.test_predictions_hidden_states/layer_{}.parquet"
    }
    layer_results = []
    
    for layer in range(0, 81, 5): 
        layer_datasets = {}
        
        # Load hidden states for this layer and attach to dataset objects
        for ds_name, path_template in parquet_map.items():
            parquet_path = path_template.format(layer)
            if not os.path.exists(parquet_path):
                # Only print warning once per dataset (e.g. at layer 0) to reduce spam
                if layer == 0:
                    print(f"Layer {layer}: File not found for {ds_name} at {parquet_path}")
                continue
            try:
                hidden_states = pq.read_table(parquet_path).to_pandas()
                if "Unnamed: 0" in hidden_states.columns:
                    hidden_states = hidden_states.drop(columns=["Unnamed: 0"])
                # Combine with the pre-calculated statistics
                ds_entry = all_datasets[ds_name].copy()
                ds_entry['hidden_states'] = hidden_states
                layer_datasets[ds_name] = ds_entry
                
            except Exception as e:
                print(f"Error loading {ds_name} for layer {layer}: {e}")

        # Ensure our test dataset is present for this layer
        if test_dataset_name not in layer_datasets:
            print(f"Layer {layer}: Test dataset {test_dataset_name} missing. Skipping.")
            continue
            
        # Run training and evaluation
        try:
            res = train_and_evaluate_loop(layer_datasets, test_dataset_name, layer)
            layer_results.append(res)
            # Save intermediate results
            os.makedirs("results_metamodel", exist_ok=True)
            pd.DataFrame(res).to_csv(f"results_metamodel/results_{test_dataset_name}_layer{layer}.csv", index=False)
            print(f"Completed analysis for layer {layer} on test dataset {test_dataset_name}.")
        except Exception as e:
            print(f"Error during training/eval for layer {layer}: {e}")
            
    return pd.DataFrame(layer_results)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run layer analysis for a specific test dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="The key of the dataset to use as the test set.")
    parser.add_argument("--output_dir", type=str, default="results_metamodel_all_datasets_llama70b", help="Folder to save the output CSV.")

    args = parser.parse_args()
    
    test_dataset = args.dataset
    
    if test_dataset not in all_datasets:
        raise ValueError(f"Dataset '{test_dataset}' not found. Available keys: {list(all_datasets.keys())}")

    print(f"--- Starting run for test dataset: {test_dataset} ---")
    df_results = run_layer_analysis(test_dataset)
    
    # Save results to a CSV file specific to this dataset
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = os.path.join(args.output_dir, f"results_{test_dataset}.csv")
    df_results.to_csv(output_filename, index=False)
    print(f"--- Finished {test_dataset}. Results saved to {output_filename} ---")