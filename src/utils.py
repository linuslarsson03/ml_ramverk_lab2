from pathlib import Path

# Takes me to the root
BASE_PATH = Path(__file__).parent.parent

# Path to trainLabels.csv
TRAINLABELS_CSV_PATH = BASE_PATH / "data" / "raw" /"cifar10" / "trainLabels.csv"

# Path to train data

TRAIN_DATA_PATH = BASE_PATH / "data" / "raw" / "cifar10" / "train"

# Path to test data
TEST_DATA_PATH = BASE_PATH / "data" / "raw" / "cifar10" / "test"




