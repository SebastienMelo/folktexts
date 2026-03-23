#%%
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
# from glestimation.core import PartitioningEstimate
#%% 

from folktexts.acs import ACSDataset

from folktexts.acs import ACSTaskMetadata
import pandas as pd
# import glest 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.tree import DecisionTreeRegressor

# from glest.plot import grouping_diagram_residuals
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



class PartitioningEstimate:
    """A class for partitioning-based estimation with honest splitting.

    This class fits a partitioning estimator to predict residuals from calibrated
    scores, enabling grouping loss estimation and risk analysis.

    Parameters
    ----------
    estimator : object or str
        The estimator to use for partitioning (e.g., DecisionTreeRegressor, KMeans).
        Can also be a string name like "decision_tree", "decision_stump", or "kmeans".
    predict_method : str, optional
        The method to call on the estimator to get partition assignments
        (e.g., "apply" for trees, "predict" for KMeans). Default is None.
    verbose : int, default=1
        Controls verbosity of output during fitting and evaluation.

    Attributes
    ----------
    calibrator : LogisticRegression
        The fitted calibrator for probability scores.
    tree : callable
        Function mapping features to residual predictions.
    r_j : ndarray
        Mean residuals for each partition.
    v_j : ndarray
        Variance of residuals for each partition.
    n_j : ndarray
        Number of samples in each partition.
    group_definitions : dict
        Human-readable definitions of each partition/group.

    """

    def __init__(
        self,
        estimator,
        predict_method: str = None,
        verbose: int = 0,
    ) -> None:
        self.estimator = estimator
        self.predict_method = predict_method
        self.verbose = verbose

    @classmethod
    def from_name(
        cls,
        name: str,
        verbose: int = 0,
        random_state: int = None,
    ):
        """Load a predefined partitioning estimator from a name.

        Parameters
        ----------
        name : {"decision_tree", "decision_stump", "kmeans", None}
            The predefined estimator to use for partitioning.
        verbose : int, default=0
            Controls verbosity of output.
        random_state : int, default=None
            Controls the randomness of the estimator.

        Returns
        -------
        estimator : object or None
            The configured estimator instance.

        """
        available_names = [
            "decision_tree",
            "decision_stump",
            "kmeans",
            None,
        ]

        if name not in available_names:
            raise ValueError(
                f'Unknown name "{name}". Available names are: {available_names}.'
            )

        if name == "decision_tree":
            estimator = DecisionTreeRegressor(
                max_depth=10,
                random_state=random_state,
                min_samples_leaf=15,
            )
            predict_method = "apply"

        elif name == "decision_stump":
            estimator = DecisionTreeRegressor(max_depth=1, random_state=random_state)
            predict_method = "apply"

        # elif name == "kmeans":
        #     estimator = KMeans(
        #         n_clusters=2,
        #         random_state=random_state,
        #         n_init="auto",
        #     )
        #     predict_method = "predict"

        elif name is None:
            estimator = None
            predict_method = None

        return estimator, predict_method

    def fit(self, X, y_scores, y_true, seed: int = 42):
        """
        Fit the partitioning estimator with honest splitting.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input features.
        y_scores : array-like of shape (n_samples,)
            The predicted probability scores from a classifier.
        y_true : array-like of shape (n_samples,)
            The true binary labels.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        if self.verbose > 0:
            print("Starting fit process...")

        y_scores = y_scores.reshape(-1, 1)
        X_train, X_test, y_scores_train, y_scores_test, y_true_train, y_true_test = (
            train_test_split(X, y_scores, y_true, test_size=0.5, random_state=seed)
        )

        X_train, X_cal, y_scores_train, y_scores_cal, y_true_train, y_true_cal = (
            train_test_split(
                X_train, y_scores_train, y_true_train, test_size=0.2, random_state=seed
            )
        )

        if self.verbose > 0:
            print(f"Calibration set size: {len(X_cal)}")
            print(f"Train set size: {len(X_train)}")
            print(f"Test set size: {len(X_test)}")

        self.calibrate(y_scores_cal, y_true_cal)
        self.train(X_train, y_scores_train, y_true_train)
        self.evaluate(X_test, y_scores_test, y_true_test)

        if hasattr(X_test, "columns"):
            feature_names = X_test.columns.tolist()
        else:
            feature_names = None
        self.get_group_definitions(X_test, feature_names=feature_names)

        if self.verbose > 0:
            print("Fit process completed.")

        return self

    def calibrate(self, y_scores, y_true):
        """
        Calibrate the predicted scores using logistic regression.
        Parameters
        ----------
        y_scores : array-like of shape (n_samples,)
            The predicted probability scores from a classifier.
        y_true : array-like of shape (n_samples,)
            The true binary labels.
        Returns
        -------
        self : object
            Fitted calibrator.
        """
        if self.verbose > 1:
            print("Calibrating scores...")

        calibrator = LogisticRegression()
        calibrator.fit(y_scores, y_true)
        self.calibrator = calibrator

        if self.verbose > 1:
            print("Calibration completed.")

        return self

    def train(self, X, y_scores, y_true):
        """
        Train the partitioning estimator on residuals.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input features.
        y_scores : array-like of shape (n_samples,)
            The predicted probability scores from a classifier.
        y_true : array-like of shape (n_samples,)
            The true binary labels.
        Returns
        -------
        self : object
            Fitted partitioning estimator.
        """
        if self.verbose > 1:
            print("Training partitioning estimator...")

        if isinstance(self.estimator, str):
            self.estimator, self.predict_method = PartitioningEstimate.from_name(
                self.estimator
            )

        residuals_train = y_true - self.calibrator.predict(y_scores)
        self.estimator.fit(X, residuals_train)

        if self.verbose > 1:
            print("Training completed.")

        return self

    def evaluate(self, X, y_scores, y_true):
        """
        Evaluate the partitioning estimator on a test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input features.
        y_scores : array-like of shape (n_samples,)
            The predicted probability scores from a classifier.
        y_true : array-like of shape (n_samples,)
            The true binary labels.
        Returns
        -------
        self : object
            Evaluated partitioning estimator with computed statistics.
        """
        if self.verbose > 1:
            print("Evaluating on test set...")

        self.y_scores = y_scores
        self.y_true = y_true
        self.X = X
        leaf_indices = self.estimator.apply(X)

        c_hat = self.calibrator.predict_proba(y_scores)[:, 1]

        v_j = np.zeros(max(leaf_indices) + 1)
        r_j = np.zeros(max(leaf_indices) + 1)
        n_j = np.zeros(max(leaf_indices) + 1)
        # Vectorized computation using bincount for better performance
        n_j = np.bincount(leaf_indices, minlength=max(leaf_indices) + 1)
        # Compute residuals once
        residuals = y_true - c_hat

        # Vectorized computation of means and variances
        r_j = np.divide(
            np.bincount(leaf_indices, weights=residuals),
            n_j,
            out=np.zeros_like(n_j, dtype=float),
            where=n_j > 0,
        )
        # Compute variance using E[X^2] - E[X]^2 formula
        residuals_sq = residuals**2
        mean_sq = np.divide(
            np.bincount(leaf_indices, weights=residuals_sq),
            n_j,
            out=np.zeros_like(n_j, dtype=float),
            where=n_j > 0,
        )
        v_j = mean_sq - r_j**2

        # Apply Bessel's correction (ddof=1)
        v_j *= n_j / (n_j - 1)
        v_j = np.where(n_j > 1, v_j, 0)

        def r(X):
            leaf_indices = self.estimator.apply(X)
            return r_j[leaf_indices]

        self.cal_err = np.mean(np.square(y_scores.flatten() - c_hat))
        self.tree = r
        self.r_j = r_j
        self.v_j = v_j
        self.n_j = n_j

        if self.verbose > 0:
            print(f"Evaluation completed. Found {len(np.unique(leaf_indices))} groups.")
            print(f"Calibration error: {self.cal_err:.4f}")

        return self

    def predict(self, X):
        """
        Predict honest residuals for new data points.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input features.
        Returns
        -------
        r_hat : array-like of shape (n_samples,)
            The predicted residuals.
        """
        return self.tree(X)

    def apply(self, X):
        return self.estimator.apply(X)

    # def plot(self, groups="all"):
    #     # check_is_fitted(self)
    #     leaf_ids = self.apply(self.X)
    #     n_in_leaf = self.n_j[leaf_ids]
    #     grouping_diagram(
    #         c_hat=self.calibrator.predict_proba(self.y_scores)[:, 1],
    #         r_hat=self.predict(self.X),
    #         n_in_leaf=n_in_leaf,
    #         f=self.y_scores.flatten(),
    #         leaf_ids=leaf_ids,
    #         groups=groups,
    #     )

    def get_group_definitions(self, X, feature_names=None):
        """
        Extract human-readable decision rules for each partition/group.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input features used to traverse the tree.
        feature_names : list of str, optional
            Names of features for readable output. If None, uses X_0, X_1, etc.

        Returns
        -------
        group_definitions : dict
            Dictionary mapping leaf IDs to group information including rules,
            sample counts, and heterogeneity measures.
        """
        tree = self.estimator
        # Convert to numpy array if pandas DataFrame
        X_array = X.values if hasattr(X, "values") else np.asarray(X)

        # Get unique leaf IDs
        leaf_ids = tree.apply(X_array)
        unique_leaves = np.unique(leaf_ids)

        group_definitions = {}
        if feature_names is None:
            feature_names = [f"X_{i}" for i in range(X_array.shape[1])]
        elif all(isinstance(f, int) for f in feature_names):
            feature_names = [f"X_{i}" for i in feature_names]

        for leaf_id in unique_leaves:
            # Get samples in this leaf
            samples_in_leaf = X_array[leaf_ids == leaf_id]

            # Get the path to this leaf
            path = tree.decision_path(samples_in_leaf[:1]).toarray()[0]

            # Extract the rules
            raw_rules = []

            # Get the path from root to leaf
            feature = tree.tree_.feature
            threshold = tree.tree_.threshold

            for node_id in range(len(path)):
                if path[node_id] == 1:  # This node is in the path
                    if feature[node_id] != -2:  # Not a leaf node
                        # Determine if we went left or right
                        sample_feature_value = samples_in_leaf[0, feature[node_id]]
                        feat_name = feature_names[feature[node_id]]
                        if sample_feature_value <= threshold[node_id]:
                            raw_rules.append((feat_name, "<=", threshold[node_id]))
                        else:
                            raw_rules.append((feat_name, ">", threshold[node_id]))

            # Combine rules for the same feature
            feature_bounds = {}
            for feat_name, operator, value in raw_rules:
                if feat_name not in feature_bounds:
                    feature_bounds[feat_name] = {"min": None, "max": None}

                if operator == "<=":
                    if (
                        feature_bounds[feat_name]["max"] is None
                        or value < feature_bounds[feat_name]["max"]
                    ):
                        feature_bounds[feat_name]["max"] = value
                else:  # operator == ">"
                    if (
                        feature_bounds[feat_name]["min"] is None
                        or value > feature_bounds[feat_name]["min"]
                    ):
                        feature_bounds[feat_name]["min"] = value

            # Convert bounds to readable rules
            combined_rules = []
            for feat_name, bounds in feature_bounds.items():
                if bounds["min"] is not None and bounds["max"] is not None:
                    combined_rules.append(
                        f"{bounds['min']:.1f} < {feat_name} <= {bounds['max']:.1f}"
                    )
                elif bounds["min"] is not None:
                    combined_rules.append(f"{feat_name} > {bounds['min']:.1f}")
                elif bounds["max"] is not None:
                    combined_rules.append(f"{feat_name} <= {bounds['max']:.1f}")

            group_definitions[leaf_id] = {
                "rules": combined_rules,
                "n_samples": len(samples_in_leaf),
                "sample_indices": np.where(leaf_ids == leaf_id)[0],
                "heterogeneity": self.r_j[leaf_id],
            }
        self.group_definitions = group_definitions
        return group_definitions

    def groups(self):
        """
        Convert group definitions to a human-readable string format.

        Parameters
        ----------
        group_definitions : dict
            Dictionary with leaf IDs as keys and group information as values

        Returns
        -------
        str
            A formatted string with group definitions
        """
        group_definitions = self.group_definitions
        lines = []
        lines.append("=" * 80)
        lines.append("GROUP DEFINITIONS")
        lines.append("=" * 80)

        for leaf_id in sorted(group_definitions.keys()):
            info = group_definitions[leaf_id]
            lines.append(f"\nGroup {leaf_id}:")
            lines.append(f"  Heterogeneity detected: {info['heterogeneity']:.4f}")
            lines.append(f"  Number of samples: {info['n_samples']}")
            lines.append("  Rules:")
            if info["rules"]:
                for rule in info["rules"]:
                    lines.append(f"    • {rule}")
            else:
                # lines.append(f"    • No splitting rules (root/single leaf)")
                lines.append("-" * 80)

        result = "\n".join(lines)
        print(result)
        return self.group_definitions


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
            if base_name == "course_completion":
                labels_df = labels_df.replace({"Completed": 1, "Not Completed": 0})
            elif base_name == "satisfaction":
                labels_df = labels_df.replace({"satisfied": 1, "neutral or dissatisfied": 0})
            elif base_name == "rain_in_australia":
                labels_df = labels_df.replace({"Yes": 1, "No": 0})
            elif base_name == "heart_disease":
                labels_df = labels_df.replace({"Yes": 1, "No": 0})
            elif base_name == "hotel_booking_cancellations":
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
        for name in train_names:
            print(f"  {name} hidden states shape: {datasets[name]['hidden_states'].shape}")
        
        risk_train = np.concatenate([datasets[name]['risk_scores'] for name in train_names], axis=0)
        print(risk_train.shape)
        for name in train_names:
            print(f"  {name} risk scores shape: {datasets[name]['risk_scores'].shape}")

        # X_train = np.concatenate([X_train, risk_train.reshape(-1, 1)], axis=1)

        # Update test hidden_states to include risk scores so dimensions match during prediction
        # datasets[test_name]['hidden_states']['risk_score'] = datasets[test_name]['risk_scores']

        y_train = np.concatenate([datasets[name]['labels'] for name in train_names], axis=0)
        print(y_train.shape)

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
        print(f"Layer {layer_idx}: Test hidden states shape: {X_test.shape}")
        
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
        # "rain_in_australia" : 'folktexts-results-metamodel-Llama-70B_full/Llama-3.3-70B-Instruct_bench-3147198227/rain prediction in australia_subsampled-0.5_seed-42_hash-685251864.test_predictions_hidden_states/layer_{}.parquet',
        "ACSPublicCoverage": 'folktexts-results-metamodel-Llama70B/model-Llama-3.3-70B-Instruct_task-ACSPublicCoverage/Llama-3.3-70B-Instruct_bench-988939989/ACSPublicCoverage_subsampled-0.4_seed-42_hash-1833904006.test_predictions_hidden_states/layer_{}.parquet',
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
            pd.DataFrame([res]).to_csv(f"results_metamodel/results_{test_dataset_name}_layer{layer}.csv", index=False)
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