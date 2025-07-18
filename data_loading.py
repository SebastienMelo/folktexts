from folktexts.acs import ACSDataset

from folktexts.acs import ACSTaskMetadata


DATA_DIR = "data"
TASK_NAME = "ACSIncome"
task = ACSTaskMetadata.get_task(TASK_NAME, use_numeric_qa=False)

dataset = ACSDataset.make_from_task(task=task, cache_dir=DATA_DIR)