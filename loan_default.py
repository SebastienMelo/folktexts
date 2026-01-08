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

DATA_DIR = ROOT_DIR / "data"

MODEL_NAME = "meta-llama/Llama-3.2-3B"
# MODEL_NAME = "google/gemma-2b"    # Smaller model that is faster to run

TASK_NAME = "ACSIncome"

RESULTS_ROOT_DIR = ROOT_DIR / "folktexts-results-metamodel"

from folktexts.llm_utils import load_model_tokenizer, get_model_folder_path
model_folder_path = get_model_folder_path(model_name=MODEL_NAME, root_dir=MODELS_DIR)
model, tokenizer = load_model_tokenizer(MODEL_NAME)

results_dir = RESULTS_ROOT_DIR / Path(model_folder_path).name
results_dir.mkdir(exist_ok=True, parents=True)


from folktexts.col_to_text import ColumnToText

# AGE
meps_age_col = ColumnToText(
    "person_age",
    short_description="age",
    value_map=lambda x: f"{x} years old",
)

meps_min_temp_col = ColumnToText(
    "person_gender",
    short_description="gender",
    value_map=lambda x: x,
)

meps_max_temp_col = ColumnToText(
    "person_education",
    short_description="highest education level",
    value_map=lambda x: x,
)

meps_sunshine_col = ColumnToText(
    "person_income",
    short_description="annual income",
    value_map=lambda x: f"${x}"
)

meps_wind_gust_speed_col = ColumnToText(
    "person_emp_exp",
    short_description="Years of employment experience",
    value_map=lambda x: f"{x} years",
)

meps_wind_gust_dir_col = ColumnToText(
    "person_home_ownership",
    short_description="home ownership status",
    value_map=lambda x: x,
)

meps_pressure9am_col = ColumnToText(
    "loan_amnt",
    short_description="loan amount requested",
    value_map=lambda x: f"${x}"
)

meps_pressure3pm_col = ColumnToText(
    "loan_int_rate",
    short_description="loan interest rate",
    value_map=lambda x: f"{x} %",
)

meps_cloud9am_col = ColumnToText(
    "loan_intent",
    short_description="loan intent",
    value_map=lambda x: x,
)

meps_cloud3pm_col = ColumnToText(
    "loan_percent_income",
    short_description="loan amount as a percentage of annual income",
    value_map=lambda x: f"{x} %",
)

meps_humidity9am_col = ColumnToText(
    "cb_person_cred_hist_length",
    short_description="credit history length",
    value_map=lambda x: f"{x} years",
)

# meps_humidity3pm_col = ColumnToText(
#     "",
#     short_description="humidity percent at 3pm",
#     value_map=lambda x: f"{x} %",
# )


# # HIBPDX: High blood pressure diagnosis
# meps_high_blood_pressure_col = ColumnToText(
#     "HIBPDX",
#     short_description="high blood pressure diagnosis",
#     value_map={
#         1: "Yes, diagnosed with high blood pressure",
#         2: "No, not diagnosed with high blood pressure",
#         -1: "Inapplicable - Under 17 years old",
#     },
#     missing_value_fill="Inapplicable - Under 17 years old",
# )
from folktexts.qa_interface import MultipleChoiceQA, Choice

TARGET_COL = "loan_status"

utilization_qa = MultipleChoiceQA(
    column=TARGET_COL,
    text="Is the loan likely to be accepted?",
    choices=(
        Choice("Yes", 1),
        Choice("No", 0),
    ),
)

# UTILIZATION: Number of doctor visits
meps_utilization_col = ColumnToText(
    TARGET_COL,
    short_description="loan status",
    question=utilization_qa,
)

from folktexts.qa_interface import DirectNumericQA

utilization_numeric_qa = DirectNumericQA(
    column=TARGET_COL,
    text=(
        "What is the probability that this person has high health-care utilization? "
        "(i.e., more than 10 doctor visits per year)"
    ),
)
# Helper dict to access ColumnToText objects by column name
meps_columns_map: dict[str, object] = {
    col_mapper.name: col_mapper
    for col_mapper in globals().values()
    if isinstance(col_mapper, ColumnToText)
}

from folktexts.task import TaskMetadata

meps_task = TaskMetadata(
    name="LoanDefault",
    description=(
        "Predict whether a loan application will be accepted or not based on applicant information."
    ),
    features=[col.name for col in meps_columns_map.values() if col.name != TARGET_COL],
    target=TARGET_COL,
    cols_to_text=meps_columns_map,
    sensitive_attribute=None,
    multiple_choice_qa=utilization_qa,
    direct_numeric_qa=utilization_numeric_qa,
)
meps_task.use_numeric_qa = False


from folktexts.dataset import Dataset

DATA_PATH = Path(".") / "notebooks" / "data" / "loan_data.csv"
meps_df = pd.read_csv(DATA_PATH)

dataset = Dataset(
    data=meps_df,
    task=meps_task,
    test_size=0.99,
    val_size=0,
    # subsampling=0.1,   # NOTE: Optional, for faster but noisier results!
)

from folktexts.classifier import TransformersLLMClassifier
from folktexts.llm_utils import load_model_tokenizer

model, tokenizer = load_model_tokenizer(MODEL_NAME)

llm_clf = TransformersLLMClassifier(
    model=model,
    tokenizer=tokenizer,
    task=meps_task,
    batch_size=20,
    context_length=800,
)

from folktexts.benchmark import BenchmarkConfig, Benchmark
bench = Benchmark(llm_clf=llm_clf, dataset=dataset)
bench.run(results_root_dir=results_dir)
