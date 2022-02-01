from pathlib import Path

from sklearn.preprocessing import LabelEncoder
import pandas as pd

TRAIN_DATASET_PATH = f"{Path(__file__).parent.parent}/data/segmentation.csv"
TEST_DATASET_PATH = f"{Path(__file__).parent.parent}/data/segmentation.test"

assert Path(
    TEST_DATASET_PATH
).exists(), "Baixe o test_dataset antes de usar este módulo!"

test_dataset = pd.read_csv(TEST_DATASET_PATH)
train_dataset = pd.read_csv(TRAIN_DATASET_PATH)

test_np_dataset = test_dataset.to_numpy()
train_np_dataset = train_dataset.to_numpy()

le_test = LabelEncoder()
CLASSES_TEST = le_test.fit_transform(test_np_dataset[:, 0]).flatten()
SHAPE_TEST = test_np_dataset[:, 4:10]
RGB_TEST = test_np_dataset[:, 10:20]
SHAPE_RGB_TEST = test_np_dataset[:, 4:20]

le_train = LabelEncoder()
CLASSES_TRAIN = le_train.fit_transform(train_np_dataset[:, 0]).flatten()
SHAPE_TRAIN = train_np_dataset[:, 4:10]
RGB_TRAIN = train_np_dataset[:, 10:20]
SHAPE_RGB_TRAIN = train_np_dataset[:, 4:20]

ALL_TEST = {
    "rgb_test": RGB_TEST,
    "shape_test": SHAPE_TEST,
    "shape_rgb_test": SHAPE_RGB_TEST,
}

ALL_TRAIN = {
    "rgb_train": RGB_TRAIN,
    "shape_train": SHAPE_TRAIN,
    "shape_rgb_train": SHAPE_RGB_TRAIN,
}

del test_np_dataset, train_np_dataset

if __name__ == "__main__":
    print(len(test_dataset))
