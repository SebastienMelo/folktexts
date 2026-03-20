#%%
import numpy as np
import pandas as pd
from pathlib import Path
#%%

from folktexts.acs import ACSDataset
from folktexts.acs import ACSTaskMetadata
import pandas as pd
import numpy as np
import pandas as pd
import folktables

#%%
def preprocess_dataset(feature_path, prediction_path, columns_to_keep, output_folder, dataset_name):
    df_features = pd.read_csv(feature_path)
    df_preds = pd.read_csv(prediction_path)

    # Create a mapping between features and df indexes
    matched_features = df_features.loc[df_preds["Unnamed: 0"]].copy()

    test = df_preds.set_index(df_preds["Unnamed: 0"].values)
    test.drop(columns=["Unnamed: 0"], inplace=True)

    merged_df = pd.concat([matched_features, test], axis=1)

    f = merged_df["risk_score"]
    y = merged_df["label"]
    mask = y.notna()
    merged_df = merged_df[mask]
    f = f[mask]
    y = y[mask]
    merged_df.drop(columns=["risk_score", "label"], inplace=True)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    mask.to_csv(output_folder / f"{dataset_name}_mask.csv", index=True)
    
    merged_df = merged_df[columns_to_keep]

    # Save datasets
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    merged_df.to_csv(output_folder / f"{dataset_name}_features.csv", index=False)
    f.to_csv(output_folder / f"{dataset_name}_risk_scores.csv", index=False)
    y.to_csv(output_folder / f"{dataset_name}_labels.csv", index=False)

    return merged_df, f, y


output_folder = "merged_datasets_Llama70B_full"

list_rain = ["Location", "MinTemp", "MaxTemp", "Sunshine", "WindGustSpeed", "WindGustDir", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm"]
features_path_rain = "notebooks/data/weatherAUS.csv"
predictions_path_rain = "folktexts-results-metamodel-Llama-70B_full/Llama-3.3-70B-Instruct_bench-3147198227/rain prediction in australia_subsampled-0.5_seed-42_hash-685251864.test_predictions.csv"
name_rain = "rain_in_australia"

list_loan = ["person_age", "person_gender", "person_education", "person_income", "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_int_rate", "loan_intent", "loan_percent_income", "cb_person_cred_hist_length"]
features_path_loan = "notebooks/data/loan_data.csv"
predictions_path_loan = "folktexts-results-metamodel-Llama-70B_full/Llama-3.3-70B-Instruct_bench-2130411282/LoanDefault_full_seed-42_hash-4198601886.test_predictions.csv"
name_loan = "loan_default"


list_meps = ["AGE", "REGION", "SEX", "MARRY", "HONRDC", "RTHLTH", "MNHLTH", "POVCAT", "INSCOV", "DIABDX", "HIBPDX"]
features_path_meps = "notebooks/data/meps.csv"
meps_prediction_path = "folktexts-results-metamodel-Llama-70B_full/Llama-3.3-70B-Instruct_bench-2962498681/health-care utilization_full_seed-42_hash-180138447.test_predictions.csv"
name_meps = "meps"


list_airline_satisfaction = ["Age", "Customer Type", "Gender", "Type of Travel", "Class", "Flight Distance", "Inflight wifi service", "Food and drink", "Seat comfort", "Inflight wifi service", "Cleanliness"]
feature_path_airline_satisfaction = "notebooks/data/train.csv"
airline_satisfaction_prediction_path = "folktexts-results-metamodel-Llama-70B_full/Llama-3.3-70B-Instruct_bench-4169613192/airline passenger satisfaction_subsampled-0.5_seed-42_hash-2223126573.test_predictions.csv"
name_airline_satisfaction = "satisfaction"


list_completion = ["Age", "Education_Level", "Gender", "Employment_Status", "Login_Frequency", "Course_Level", "Category", "Course_Duration_Days", "Time_Spent_Hours", "Internet_Connection_Quality"]
feature_path_completion = "notebooks/data/Course_Completion_Prediction.csv"
prediction_path_completion = "folktexts-results-metamodel-Llama-70B_full/Llama-3.3-70B-Instruct_bench-1051901534/course completion prediction_subsampled-0.5_seed-42_hash-1848123039.test_predictions.csv"
name_completion = "course_completion"


list_smoking = ["gender", "age", "height(cm)", "weight(kg)", "systolic", "relaxation", "dental carries", "Cholesterol", "tartar"]
feature_path_smoking = "notebooks/data/smoking.csv"
smoking_prediction_path = "folktexts-results-metamodel-Llama-70B_full/Llama-3.3-70B-Instruct_bench-1085260167/SmokingPrediction_subsampled-0.99_seed-42_hash-1596432086.test_predictions.csv"
name_smoking = "smoking"


list_heart = ["Sex", "PhysicalHealth", "MentalHealth", "SleepTime", "PhysicalActivity", "AgeCategory", "BMI", "Race", "AlcoholDrinking", "GenHealth"]
feature_path_heart = "notebooks/data/heart_2020_cleaned.csv"
heart_prediction_path = "folktexts-results-metamodel-Llama-70B_full/Llama-3.3-70B-Instruct_bench-290020217/HeartDiseasePrediction_subsampled-0.15_seed-42_hash-408096573.test_predictions.csv"
name_heart = "heart_disease"


list_booking = ["number of adults", "number of children", "car parking space", "lead time", "number of weekend nights", "number of week nights", "average price", "special requests"]
feature_path_booking = "notebooks/data/bookings.csv"
prediction_path_booking = "folktexts-results-metamodel-Llama-70B_full/Llama-3.3-70B-Instruct_bench-1249158703/HotelBookingCancellation_full_seed-42_hash-1846594368.test_predictions.csv"
name_booking = "hotel_booking_cancellations"

def preprocess_acs_dataset(task_name, prediction_path, output_folder):
    task = ACSTaskMetadata.get_task(task_name, use_numeric_qa=False)
    dataset = ACSDataset.make_from_task(task=task, cache_dir="notebooks/data")

    df_preds = pd.read_csv(prediction_path)
    features = dataset.get_features_data()
    
    # Create a mapping between features and df indexes
    matched_features = features.loc[df_preds["Unnamed: 0"]].copy()

    test = df_preds.set_index(df_preds["Unnamed: 0"].values)
    test.drop(columns=["Unnamed: 0"], inplace=True)

    merged_df = pd.concat([matched_features, test], axis=1)

    f = merged_df["risk_score"]
    y = merged_df["label"]

    merged_df.drop(columns=["risk_score", "label"], inplace=True)
    columns_to_keep = folktables_task = getattr(folktables, task_name).features

    merged_df = merged_df[columns_to_keep]

    # Save datasets
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    merged_df.to_csv(output_folder / f"{task_name}_features.csv", index=False)
    f.to_csv(output_folder / f"{task_name}_risk_scores.csv", index=False)
    y.to_csv(output_folder / f"{task_name}_labels.csv", index=False)

    return merged_df, f, y


if __name__ == "__main__":
    tasks = [name_rain, name_loan, name_meps, name_airline_satisfaction, name_completion, name_smoking, name_heart, name_booking]
    prediction_paths = [
        # predictions_path_rain,
        # predictions_path_loan,
        # meps_prediction_path,
        # airline_satisfaction_prediction_path,
        # prediction_path_completion,
        smoking_prediction_path,
        heart_prediction_path,
        prediction_path_booking
    ]

    features = [
        # list_rain,
        # list_loan,
        # list_meps,
        # list_airline_satisfaction,
        # list_completion,
        list_smoking,
        list_heart,
        list_booking
    ]

    feature_paths = [
        # features_path_rain,
        # features_path_loan,
        # features_path_meps,
        # feature_path_airline_satisfaction,
        # feature_path_completion,
        feature_path_smoking,
        feature_path_heart,
        feature_path_booking
    ]


    for task_name, prediction_path, columns_to_keep, feature_path in zip(tasks, prediction_paths, features, feature_paths):
        preprocess_dataset(feature_path, prediction_path, columns_to_keep, output_folder, task_name)

# %%
