#%%
import folktexts
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import logging
logging.getLogger().setLevel(logging.INFO)

ROOT_DIR = Path(".")
ROOT_DIR

MODELS_DIR = ROOT_DIR / "models"

DATA_DIR = ROOT_DIR / "notebooks/data"

MODEL_NAME = "meta-llama/Llama-3.2-3B"
# MODEL_NAME = "google/gemma-2b"    # Smaller model that is faster to run

TASK_NAME = "ACSIncome"

RESULTS_ROOT_DIR = ROOT_DIR / "folktexts-results-metamodel-Llama70B_full"

from folktexts.llm_utils import load_model_tokenizer, get_model_folder_path
model_folder_path = get_model_folder_path(model_name=MODEL_NAME, root_dir=MODELS_DIR)

results_dir = RESULTS_ROOT_DIR / Path(model_folder_path).name
results_dir.mkdir(exist_ok=True, parents=True)
results_dir



from folktexts.col_to_text import ColumnToText
from folktexts.qa_interface import MultipleChoiceQA, Choice, DirectNumericQA
from folktexts.task import TaskMetadata
from folktexts.dataset import Dataset
from folktexts.classifier import TransformersLLMClassifier
from folktexts.llm_utils import load_model_tokenizer
from folktexts.benchmark import Benchmark
import argparse


def run_meps_task(data_path: Path, model_name: str, results_dir: Path):
    """Run healthcare utilization prediction task on MEPS data."""
    meps_age_col = ColumnToText(
        "AGE",
        short_description="age",
        value_map=lambda x: f"{int(x)} years old",
    )
    meps_region_col = ColumnToText(
        "REGION",
        short_description="US region",
        value_map={1: "Northeast", 2: "Midwest", 3: "South", 4: "West"},
    )
    meps_sex_col = ColumnToText(
        "SEX",
        short_description="sex",
        value_map={1: "Male", 2: "Female"},
    )
    meps_marital_status = ColumnToText(
        "MARRY",
        short_description="marital status",
        value_map={
            1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated",
            5: "Never married", 6: "Inapplicable - Under 16 years old",
            7: "Married during current survey round", 8: "Widowed during current survey round",
            9: "Divorced during current survey round", 10: "Separated during current survey round",
        },
    )
    meps_education_col = ColumnToText(
        "HONRDC",
        short_description="honorably discharged status",
        value_map={
            1: "Yes, honorably discharged from military",
            2: "Not part of military or not honorably discharged",
            3: "Inapplicable - Under 16 years old",
            4: "Inapplicable - Now on active duty",
        },
    )
    meps_health_status_col = ColumnToText(
        "RTHLTH",
        short_description="self-rated health status",
        value_map={1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor",
                   -1: "Inapplicable - Missing data", -7: "Inapplicable - Refused to answer",
                   -8: "Inapplicable - Don't know"},
    )
    meps_mental_health_status_col = ColumnToText(
        "MNHLTH",
        short_description="self-rated mental health status",
        value_map={1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor",
                   -1: "Inapplicable - Missing data", -7: "Inapplicable - Refused to answer",
                   -8: "Inapplicable - Don't know"},
    )
    meps_poverty_category_col = ColumnToText(
        "POVCAT",
        short_description="poverty category",
        value_map={1: "Poor", 2: "Near poor", 3: "Low income", 4: "Middle income", 5: "High income"},
    )
    meps_insurance_coverage_col = ColumnToText(
        "INSCOV",
        short_description="insurance coverage",
        value_map={1: "Private insurance", 2: "Public insurance", 3: "Uninsured"},
    )
    meps_diabetes_col = ColumnToText(
        "DIABDX",
        short_description="diabetes diagnosis",
        value_map={1: "Yes, diagnosed with diabetes", 2: "No, not diagnosed with diabetes",
                   -1: "Inapplicable - Under 17 years old"},
        missing_value_fill="Inapplicable - Under 17 years old",
    )
    meps_high_blood_pressure_col = ColumnToText(
        "HIBPDX",
        short_description="high blood pressure diagnosis",
        value_map={1: "Yes, diagnosed with high blood pressure",
                   2: "No, not diagnosed with high blood pressure",
                   -1: "Inapplicable - Under 17 years old"},
        missing_value_fill="Inapplicable - Under 17 years old",
    )

    TARGET_COL = "UTILIZATION"
    utilization_qa = MultipleChoiceQA(
        column=TARGET_COL,
        text="What is this person's estimated number of doctor visits in the past year?",
        choices=(
            Choice("More than 10 doctor visits (high health-care utilization)", 1),
            Choice("10 or fewer doctor visits (low health-care utilization)", 0),
        ),
    )
    meps_utilization_col = ColumnToText(TARGET_COL, short_description="doctor visits", question=utilization_qa)
    utilization_numeric_qa = DirectNumericQA(
        column=TARGET_COL,
        text="What is the probability that this person has high health-care utilization? (i.e., more than 10 doctor visits per year)",
    )

    cols = [meps_age_col, meps_region_col, meps_sex_col, meps_marital_status, meps_education_col,
            meps_health_status_col, meps_mental_health_status_col, meps_poverty_category_col,
            meps_insurance_coverage_col, meps_diabetes_col, meps_high_blood_pressure_col, meps_utilization_col]
    meps_columns_map = {col.name: col for col in cols}

    meps_task = TaskMetadata(
        name="health-care utilization",
        description="predict whether an individual had low or high healthcare utilization in the past year by their number of doctor visits",
        features=[col.name for col in cols if col.name != TARGET_COL],
        target=TARGET_COL,
        cols_to_text=meps_columns_map,
        sensitive_attribute="SEX",
        multiple_choice_qa=utilization_qa,
        direct_numeric_qa=utilization_numeric_qa,
    )
    meps_task.use_numeric_qa = False

    meps_df = pd.read_csv(data_path)
    dataset = Dataset(data=meps_df, task=meps_task, test_size=0.99, val_size=0)

    model, tokenizer = load_model_tokenizer(model_name)
    llm_clf = TransformersLLMClassifier(model=model, tokenizer=tokenizer, task=meps_task, batch_size=10, context_length=1000)

    bench = Benchmark(llm_clf=llm_clf, dataset=dataset)
    bench.run(results_root_dir=results_dir)


def run_airline_satisfaction_task(data_path: Path, model_name: str, results_dir: Path):
    """Run airline passenger satisfaction prediction task."""
    airline_age_col = ColumnToText("Age", short_description="age", value_map=lambda x: f"{int(x)} years old")
    
    airline_customer_type_col = ColumnToText(
        "Customer Type", short_description="customer type",
        value_map={"Loyal Customer": "Loyal Customer", "disloyal Customer": "disloyal Customer"},
    )
    airline_sex_col = ColumnToText("Gender", short_description="gender", value_map={"Male": "Male", "Female": "Female"})
    airline_travel_type_col = ColumnToText(
        "Type of Travel", short_description="type of travel",
        value_map={"Business travel": "Business travel", "Personal Travel": "Personal travel"},
    )
    airline_class_col = ColumnToText(
        "Class", short_description="flight class",
        value_map={"Eco": "Economy", "Business": "Business", "Eco Plus": "Economy Plus"},
    )
    airline_distance_col = ColumnToText("Flight Distance", short_description="flight distance", value_map=lambda x: f"{int(x)} kilometers")
    airline_wifi_col = ColumnToText(
        "Inflight wifi service", short_description="satisfaction level of the inflight wifi service",
        value_map={1: "Very bad", 2: "Bad", 3: "Neutral", 4: "Good", 5: "Very good", 0: "Not applicable"},
    )
    airline_seat_col = ColumnToText(
        "Seat comfort", short_description="comfort level of the seat",
        value_map={1: "Very bad", 2: "Bad", 3: "Neutral", 4: "Good", 5: "Very good"},
    )
    airline_cleanliness_col = ColumnToText(
        "Cleanliness", short_description="cleanliness of the airplane",
        value_map={1: "Very bad", 2: "Bad", 3: "Neutral", 4: "Good", 5: "Very good", 0: "Not applicable"},
    )
    airline_food_col = ColumnToText(
        "Food and drink", short_description="satisfaction level of the food and drinks",
        value_map={1: "Very bad", 2: "Bad", 3: "Neutral", 4: "Good", 5: "Very good", 0: "Not applicable"},
    )

    TARGET_COL = "satisfaction"
    satisfaction_qa = MultipleChoiceQA(
        column=TARGET_COL,
        text="What is this person's airline satisfaction level?",
        choices=(Choice("Satisfied", 1), Choice("Neutral or dissatisfied", 0)),
    )
    airline_satisfaction_col = ColumnToText(TARGET_COL, short_description="satisfaction", question=satisfaction_qa)
    satisfaction_numeric_qa = DirectNumericQA(
        column=TARGET_COL,
        text="What is the probability that this person is satisfied with their airline experience?",
    )

    cols = [airline_age_col, airline_customer_type_col, airline_sex_col, airline_travel_type_col,
            airline_class_col, airline_distance_col, airline_wifi_col, airline_seat_col,
            airline_cleanliness_col, airline_food_col, airline_satisfaction_col]
    airline_columns_map = {col.name: col for col in cols}

    airline_task = TaskMetadata(
        name="airline passenger satisfaction",
        description="predict whether an individual is satisfied or not with their airline experience",
        features=[col.name for col in cols if col.name != TARGET_COL],
        target=TARGET_COL,
        cols_to_text=airline_columns_map,
        sensitive_attribute="Gender",
        multiple_choice_qa=satisfaction_qa,
        direct_numeric_qa=satisfaction_numeric_qa,
    )
    airline_task.use_numeric_qa = False

    airline_df = pd.read_csv(data_path)
    dataset = Dataset(data=airline_df, task=airline_task, test_size=0.99, val_size=0, subsampling=0.5)

    model, tokenizer = load_model_tokenizer(model_name)
    llm_clf = TransformersLLMClassifier(model=model, tokenizer=tokenizer, task=airline_task, batch_size=10, context_length=1000)

    bench = Benchmark(llm_clf=llm_clf, dataset=dataset)
    bench.run(results_root_dir=results_dir)


def run_course_completion_task(data_path: Path, model_name: str, results_dir: Path):
    """Run course completion prediction task."""
    course_age_col = ColumnToText("Age", short_description="age", value_map=lambda x: f"{int(x)} years old")
    course_education_col = ColumnToText(
        "Education_Level", short_description="education level attained by the student",
        value_map={"Diploma": "Diploma", "Bachelor": "Bachelor", "Master": "Master", "PhD": "PhD", "HighSchool": "High School"},
    )
    course_gender_col = ColumnToText("Gender", short_description="gender", value_map={"Male": "Male", "Female": "Female", "Other": "Other"})
    course_employment_col = ColumnToText(
        "Employment_Status", short_description="current employment status of the student",
        value_map={"Employed": "Employed", "Student": "Student", "Unemployed": "Unemployed", "Self-Employed": "Self-employed"},
    )
    course_login_col = ColumnToText("Login_Frequency", short_description="number of times the student logs in per week", value_map=lambda x: f"{int(x)} times per week")
    course_level_col = ColumnToText(
        "Course_Level", short_description="course level",
        value_map={"Beginner": "Beginner", "Intermediate": "Intermediate", "Advanced": "Advanced"},
    )
    course_category_col = ColumnToText(
        "Category", short_description="subject category of the course",
        value_map={"Programming": "Programming", "Marketing": "Marketing", "Business": "Business", "Math": "Math", "Design": "Design"},
    )
    course_duration_col = ColumnToText("Course_Duration_Days", short_description="total intended duration of the course in days.", value_map=lambda x: f"{int(x)} days")
    course_time_col = ColumnToText("Time_Spent_Hours", short_description="Total number of hours spent actively engaging with the course content.", value_map=lambda x: f"{x} hours")
    course_internet_col = ColumnToText(
        "Internet_Connection_Quality", short_description="quality of the student's internet connection",
        value_map={"Medium": "Medium", "High": "High", "Low": "Low"},
    )

    TARGET_COL = "Completed"
    completion_qa = MultipleChoiceQA(
        column=TARGET_COL,
        text="Did the student complete the course?",
        choices=(Choice("Completed", 1), Choice("Not Completed", 0)),
    )
    course_completion_col = ColumnToText(TARGET_COL, short_description="course completion", question=completion_qa)
    completion_numeric_qa = DirectNumericQA(
        column=TARGET_COL,
        text="What is the probability that this student completes the course?",
    )

    cols = [course_age_col, course_education_col, course_gender_col, course_employment_col,
            course_login_col, course_level_col, course_category_col, course_duration_col,
            course_time_col, course_internet_col, course_completion_col]
    course_columns_map = {col.name: col for col in cols}

    course_task = TaskMetadata(
        name="course completion prediction",
        description="predict whether an individual completes an online course",
        features=[col.name for col in cols if col.name != TARGET_COL],
        target=TARGET_COL,
        cols_to_text=course_columns_map,
        sensitive_attribute="Gender",
        multiple_choice_qa=completion_qa,
        direct_numeric_qa=completion_numeric_qa,
    )
    course_task.use_numeric_qa = False

    course_df = pd.read_csv(data_path)
    dataset = Dataset(data=course_df, task=course_task, test_size=0.99, val_size=0, subsampling=0.5)

    model, tokenizer = load_model_tokenizer(model_name)
    llm_clf = TransformersLLMClassifier(model=model, tokenizer=tokenizer, task=course_task, batch_size=10, context_length=1000)

    bench = Benchmark(llm_clf=llm_clf, dataset=dataset)
    bench.run(results_root_dir=results_dir)



def run_rain_prediction_task(data_path: Path, model_name: str, results_dir: Path):
    """Run rain prediction task on Australian weather data."""
    location_col = ColumnToText(
        "Location",
        short_description="Location",
        value_map=lambda x: x,
    )
    min_temp_col = ColumnToText(
        "MinTemp",
        short_description="minimum temperature in degrees Celsius",
        value_map=lambda x: f"{x} °C",
    )
    max_temp_col = ColumnToText(
        "MaxTemp",
        short_description="maximum temperature in degrees Celsius",
        value_map=lambda x: f"{x} °C",
    )
    sunshine_col = ColumnToText(
        "Sunshine",
        short_description="number of hours of sunshine",
        value_map=lambda x: f"{x} hours",
    )
    wind_gust_speed_col = ColumnToText(
        "WindGustSpeed",
        short_description="wind gust speed",
        value_map=lambda x: f"{x} km/h",
    )
    wind_gust_dir_col = ColumnToText(
        "WindGustDir",
        short_description="wind gust direction",
        value_map=lambda x: x,
    )
    pressure9am_col = ColumnToText(
        "Pressure9am",
        short_description="air pressure at 9am",
        value_map=lambda x: f"{x} hPa",
    )
    pressure3pm_col = ColumnToText(
        "Pressure3pm",
        short_description="air pressure at 3pm",
        value_map=lambda x: f"{x} hPa",
    )
    cloud9am_col = ColumnToText(
        "Cloud9am",
        short_description="cloud cover at 9am",
        value_map=lambda x: f"{x} oktas",
    )
    cloud3pm_col = ColumnToText(
        "Cloud3pm",
        short_description="cloud cover at 3pm",
        value_map=lambda x: f"{x} oktas",
    )
    humidity9am_col = ColumnToText(
        "Humidity9am",
        short_description="humidity percent at 9am",
        value_map=lambda x: f"{x} %",
    )
    humidity3pm_col = ColumnToText(
        "Humidity3pm",
        short_description="humidity percent at 3pm",
        value_map=lambda x: f"{x} %",
    )

    TARGET_COL = "RainTomorrow"
    rain_qa = MultipleChoiceQA(
        column=TARGET_COL,
        text="Is it going to rain more than 1mm tomorrow?",
        choices=(
            Choice("Yes", 1),
            Choice("No", 0),
        ),
    )
    rain_col = ColumnToText(TARGET_COL, short_description="rain tomorrow", question=rain_qa)
    rain_numeric_qa = DirectNumericQA(
        column=TARGET_COL,
        text="What is the probability that it will rain more than 1mm tomorrow?",
    )

    cols = [location_col, min_temp_col, max_temp_col, sunshine_col, wind_gust_speed_col,
            wind_gust_dir_col, pressure9am_col, pressure3pm_col, cloud9am_col, cloud3pm_col,
            humidity9am_col, humidity3pm_col, rain_col]
    rain_columns_map = {col.name: col for col in cols}

    rain_task = TaskMetadata(
        name="rain prediction in australia",
        description="predict whether it will rain more than 1mm tomorrow based on weather data",
        features=[col.name for col in cols if col.name != TARGET_COL],
        target=TARGET_COL,
        cols_to_text=rain_columns_map,
        sensitive_attribute=None,
        multiple_choice_qa=rain_qa,
        direct_numeric_qa=rain_numeric_qa,
    )
    rain_task.use_numeric_qa = False

    rain_df = pd.read_csv(data_path)
    dataset = Dataset(data=rain_df, task=rain_task, test_size=0.99, val_size=0, subsampling=0.5)

    model, tokenizer = load_model_tokenizer(model_name)
    llm_clf = TransformersLLMClassifier(model=model, tokenizer=tokenizer, task=rain_task, batch_size=10, context_length=1000)

    bench = Benchmark(llm_clf=llm_clf, dataset=dataset)
    bench.run(results_root_dir=results_dir)


def run_loan_default_task(data_path: Path, model_name: str, results_dir: Path):
    """Run loan default prediction task."""
    loan_age_col = ColumnToText(
        "person_age",
        short_description="age",
        value_map=lambda x: f"{x} years old",
    )
    loan_gender_col = ColumnToText(
        "person_gender",
        short_description="gender",
        value_map=lambda x: x,
    )
    loan_education_col = ColumnToText(
        "person_education",
        short_description="highest education level",
        value_map=lambda x: x,
    )
    loan_income_col = ColumnToText(
        "person_income",
        short_description="annual income",
        value_map=lambda x: f"${x}"
    )
    loan_emp_exp_col = ColumnToText(
        "person_emp_exp",
        short_description="Years of employment experience",
        value_map=lambda x: f"{x} years",
    )
    loan_home_ownership_col = ColumnToText(
        "person_home_ownership",
        short_description="home ownership status",
        value_map=lambda x: x,
    )
    loan_amnt_col = ColumnToText(
        "loan_amnt",
        short_description="loan amount requested",
        value_map=lambda x: f"${x}"
    )
    loan_int_rate_col = ColumnToText(
        "loan_int_rate",
        short_description="loan interest rate",
        value_map=lambda x: f"{x} %",
    )
    loan_intent_col = ColumnToText(
        "loan_intent",
        short_description="loan intent",
        value_map=lambda x: x,
    )
    loan_percent_income_col = ColumnToText(
        "loan_percent_income",
        short_description="loan amount as a percentage of annual income",
        value_map=lambda x: f"{x} %",
    )
    loan_cred_hist_length_col = ColumnToText(
        "cb_person_cred_hist_length",
        short_description="credit history length",
        value_map=lambda x: f"{x} years",
    )

    TARGET_COL = "loan_status"
    loan_qa = MultipleChoiceQA(
        column=TARGET_COL,
        text="Is the loan likely to be accepted?",
        choices=(
            Choice("Yes", 1),
            Choice("No", 0),
        ),
    )
    loan_status_col = ColumnToText(TARGET_COL, short_description="loan status", question=loan_qa)
    loan_numeric_qa = DirectNumericQA(
        column=TARGET_COL,
        text="What is the probability that the loan will be accepted?",
    )

    cols = [loan_age_col, loan_gender_col, loan_education_col, loan_income_col,
            loan_emp_exp_col, loan_home_ownership_col, loan_amnt_col, loan_int_rate_col,
            loan_intent_col, loan_percent_income_col, loan_cred_hist_length_col, loan_status_col]
    loan_columns_map = {col.name: col for col in cols}

    loan_task = TaskMetadata(
        name="LoanDefault",
        description="Predict whether a loan application will be accepted or not based on applicant information.",
        features=[col.name for col in cols if col.name != TARGET_COL],
        target=TARGET_COL,
        cols_to_text=loan_columns_map,
        sensitive_attribute="person_gender",
        multiple_choice_qa=loan_qa,
        direct_numeric_qa=loan_numeric_qa,
    )
    loan_task.use_numeric_qa = False

    loan_df = pd.read_csv(data_path)
    dataset = Dataset(data=loan_df, task=loan_task, test_size=0.99, val_size=0, subsampling=None)

    model, tokenizer = load_model_tokenizer(model_name)
    llm_clf = TransformersLLMClassifier(model=model, tokenizer=tokenizer, task=loan_task, batch_size=10, context_length=1000)

    bench = Benchmark(llm_clf=llm_clf, dataset=dataset)
    bench.run(results_root_dir=results_dir)


def run_smoking_prediction_task(data_path: Path, model_name: str, results_dir: Path):
    """Run smoking prediction task."""
    smoking_gender_col = ColumnToText(
        "gender",
        short_description="gender",
        value_map={"F": "Female", "M": "Male"}
    )
    smoking_age_col = ColumnToText(
        "age",
        short_description="age",
        value_map=lambda x: f"{x} years old",
    )
    height_col = ColumnToText(
        "height(cm)",
        short_description="height",
        value_map=lambda x: f"{x} cm",
    )
    weight_col = ColumnToText(
        "weight(kg)",
        short_description="weight",
        value_map=lambda x: f"{x} kg",
    )
    smoking_systolic_col = ColumnToText(
        "systolic",
        short_description="systolic blood pressure",
        value_map=lambda x: f"{x} mmHg",
    )
    smoking_relaxation_col = ColumnToText(
        "relaxation",
        short_description="diastolic blood pressure",
        value_map=lambda x: f"{x} mmHg",
    )
    smoking_caries_col = ColumnToText(
        "dental caries",
        short_description="presence of dental caries",
        value_map=lambda x: f"{x}",
    )
    smoking_cholesterol_col = ColumnToText(
        "Cholesterol",
        short_description="cholesterol level",
        value_map=lambda x: f"{x} mg/dL",
    )
    smoking_tartar_col = ColumnToText(
        "tartar",
        short_description="presence of dental tartar",
        value_map=lambda x: f"{x}",
    )

    TARGET_COL = "smoking"
    smoking_qa = MultipleChoiceQA(
        column=TARGET_COL,
        text="Is this person a smoker?",
        choices=(
            Choice("Yes", 1),
            Choice("No", 0),
        ),
    )
    smoking_col = ColumnToText(TARGET_COL, short_description="smoking status", question=smoking_qa)
    smoking_numeric_qa = DirectNumericQA(
        column=TARGET_COL,
        text="What is the probability that this person is a smoker?",
    )

    cols = [smoking_cholesterol_col, smoking_systolic_col, smoking_relaxation_col, smoking_caries_col, smoking_tartar_col,
            smoking_gender_col, smoking_age_col, height_col, weight_col, smoking_col]
    
    smoking_columns_map = {col.name: col for col in cols}

    smoking_task = TaskMetadata(
        name="SmokingPrediction",
        description="Predict whether a person is a smoker based on health and dental information.",
        features=[col.name for col in cols if col.name != TARGET_COL],
        target=TARGET_COL,
        cols_to_text=smoking_columns_map,
        sensitive_attribute="gender",
        multiple_choice_qa=smoking_qa,
        direct_numeric_qa=smoking_numeric_qa,
    )

    smoking_task.use_numeric_qa = False
    smoking_df = pd.read_csv(data_path)
    dataset = Dataset(data=smoking_df, task=smoking_task, test_size=0.99, val_size=0, subsampling=0.99)
    model, tokenizer = load_model_tokenizer(model_name)
    llm_clf = TransformersLLMClassifier(model=model, tokenizer=tokenizer, task=smoking_task, batch_size=10, context_length=1000)
    bench = Benchmark(llm_clf=llm_clf, dataset=dataset)
    bench.run(results_root_dir=results_dir)


def run_heart_disease_prediction_task(data_path: Path, model_name: str, results_dir: Path):
    """Run heart disease prediction task."""
    heart_sex_col = ColumnToText(
        "Sex",
        short_description="gender",
        value_map=lambda x: f"{x}",
    )
    heart_physical_health_col = ColumnToText(
        "PhysicalHealth",
        short_description="number of days in the past month with poor physical health",
        value_map=lambda x: f"{x} days",
    )

    heart_mental_health_col = ColumnToText(
        "MentalHealth",
        short_description="number of days in the past month with poor mental health",
        value_map=lambda x: f"{x} days",
    )

    heart_physical_activity_col = ColumnToText(
        "PhysicalActivity",
        short_description="whether the person engages in physical activity",
        value_map=lambda x: f'{x}',
    )

    heart_sleep_hours_col = ColumnToText(
        "SleepTime",
        short_description="average number of hours of sleep per day",
        value_map=lambda x: f"{x} hours",
    )

    heart_age_col = ColumnToText(
        "AgeCategory",
        short_description="age category",
        value_map=lambda x: f"{x}",
    )

    heart_bmi_col = ColumnToText(
        "BMI",
        short_description="body mass index",
        value_map=lambda x: f"{x}",
    )

    heart_race_col = ColumnToText(
        "Race",
        short_description="race of the person",
        value_map=lambda x: f"{x}",
    )

    heart_alcohol_col = ColumnToText(
        "AlcoholDrinking",
        short_description="whether the person is an alcohol drinker",
        value_map=lambda x: f"{x}",
    )

    heart_general_health_col = ColumnToText(
        "GenHealth",
        short_description="self-rated general health status",
        value_map=lambda x: f"{x}",
    )

    TARGET_COL = "HeartDisease"
    heart_disease_qa = MultipleChoiceQA(
        column=TARGET_COL,
        text="Does this person have heart disease?",
        choices=(
            Choice("Yes", 1),
            Choice("No", 0),
        ),
    )
    heart_disease_col = ColumnToText(TARGET_COL, short_description="heart disease status", question=heart_disease_qa)
    heart_disease_numeric_qa = DirectNumericQA(
        column=TARGET_COL,
        text="What is the probability that this person has heart disease?",
    )
    
    cols = [heart_age_col, heart_bmi_col, heart_general_health_col, heart_mental_health_col, heart_physical_activity_col,
            heart_race_col, heart_sleep_hours_col, heart_sex_col, heart_physical_health_col, heart_alcohol_col, heart_disease_col]

    heart_columns_map = {col.name: col for col in cols}
    heart_disease_task = TaskMetadata(
        name="HeartDiseasePrediction",
        description="Predict whether a person has heart disease based on health and demographic information.",
        features=[col.name for col in cols if col.name != TARGET_COL],
        target=TARGET_COL,
        cols_to_text=heart_columns_map,
        sensitive_attribute="Sex",
        multiple_choice_qa=heart_disease_qa,
        direct_numeric_qa=heart_disease_numeric_qa,
    )
    heart_disease_task.use_numeric_qa = False
    heart_df = pd.read_csv(data_path)
    dataset = Dataset(data=heart_df, task=heart_disease_task, test_size=0.99, val_size=0, subsampling=0.15)
    model, tokenizer = load_model_tokenizer(model_name)
    llm_clf = TransformersLLMClassifier(model=model, tokenizer=tokenizer, task=heart_disease_task, batch_size=10, context_length=1000)
    bench = Benchmark(llm_clf=llm_clf, dataset=dataset)
    bench.run(results_root_dir=results_dir)


def run_booking_cancellation_prediction_task(data_path: Path, model_name: str, results_dir: Path):
    """Run hotel booking cancellation prediction task."""
    booking_nb_adults_col = ColumnToText(
        "number of adults",
        short_description="number of adults included in the booking",
        value_map=lambda x: f"{x}",
    )
    booking_number_children_col = ColumnToText(
        "number of children",
        short_description="number of children included in the booking",
        value_map=lambda x: f"{x}",
    )
    booking_weekend_nights_col = ColumnToText(
        "number of weekend nights",
        short_description="number of weekend nights included in the booking",
        value_map=lambda x: f"{x}",
    )
    booking_week_nights_col = ColumnToText(
        "number of week nights",
        short_description="number of week nights included in the booking",
        value_map=lambda x: f"{x}",
    )
    booking_car_col = ColumnToText(
        "car parking space",
        short_description="booking includes car parking spaces",
        value_map={0: "No", 1: "Yes"},
    )
    booking_lead_time_col = ColumnToText(
        "lead time",
        short_description="number of days between booking and arrival",
        value_map=lambda x: f"{x} days",
    )
    booking_average_price_col = ColumnToText(
        "average price",
        short_description="average price per night of the booking",
        value_map=lambda x: f"${x}",
    )
    booking_special_requests_col = ColumnToText(
        "special requests",
        short_description="number of special requests included in the booking",
        value_map=lambda x: f"{x}",
    )

    TARGET_COL = "booking status"
    booking_qa = MultipleChoiceQA(
        column=TARGET_COL,
        text="Is this booking likely to be canceled?",
        choices=(
            Choice("Canceled", 1),
            Choice("Not canceled", 0),
        ),
    )
    booking_status_col = ColumnToText(TARGET_COL, short_description="booking status", question=booking_qa)
    booking_numeric_qa = DirectNumericQA(
        column=TARGET_COL,
        text="What is the probability that this booking will be canceled?",
    )
    cols = [booking_nb_adults_col, booking_number_children_col, booking_weekend_nights_col,
            booking_week_nights_col, booking_car_col, booking_lead_time_col,
            booking_average_price_col, booking_special_requests_col, booking_status_col]
    booking_columns_map = {col.name: col for col in cols}
    booking_task = TaskMetadata(
        name="HotelBookingCancellation",
        description="Predict whether a hotel booking will be canceled based on booking details.",
        features=[col.name for col in cols if col.name != TARGET_COL],
        target=TARGET_COL,
        cols_to_text=booking_columns_map,
        sensitive_attribute=None,
        multiple_choice_qa=booking_qa,
        direct_numeric_qa=booking_numeric_qa,
    )
    booking_task.use_numeric_qa = False
    booking_df = pd.read_csv(data_path)
    dataset = Dataset(data=booking_df, task=booking_task, test_size=0.99, val_size=0, subsampling=None)
    model, tokenizer = load_model_tokenizer(model_name)
    llm_clf = TransformersLLMClassifier(model=model, tokenizer=tokenizer, task=booking_task, batch_size=10, context_length=1000)
    bench = Benchmark(llm_clf=llm_clf, dataset=dataset)

    bench.run(results_root_dir=results_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run prediction tasks with LLM classifiers")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["meps", "airline", "course", "rain", "loan", "smoking", "heart", "booking"],
        help="Task to run: meps, airline, course, or rain",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="$DSDIR/",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("./folktexts-results-metamodel-Llama-3B_full"),
        help="Directory to save results",
    )

    args = parser.parse_args()

    args.results_dir.mkdir(exist_ok=True, parents=True)

    task_functions = {
        "meps": run_meps_task,
        "airline": run_airline_satisfaction_task,
        "course": run_course_completion_task,
        "rain": run_rain_prediction_task,
        "loan": run_loan_default_task,
        "smoking": run_smoking_prediction_task,
        "heart": run_heart_disease_prediction_task,
        "booking": run_booking_cancellation_prediction_task,
    }

    data_paths = {
        "meps": DATA_DIR / "meps.csv",
        "airline": DATA_DIR / "train.csv",
        "course": DATA_DIR / "Course_Completion_Prediction.csv",
        "rain": DATA_DIR / "weatherAUS.csv",
        "loan": DATA_DIR / "loan_data.csv",
        "smoking": DATA_DIR / "smoking.csv",
        "heart": DATA_DIR / "heart_2020_cleaned.csv",
        "booking": DATA_DIR / "booking.csv",
    }

    task_fn = task_functions[args.task]
    data_path = data_paths[args.task]
    task_fn(data_path=data_path, model_name=args.model_name, results_dir=args.results_dir)
# %%
