#%%
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.tree import DecisionTreeRegressor

import numpy as np

import os
import glob
import re
import pandas as pd
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