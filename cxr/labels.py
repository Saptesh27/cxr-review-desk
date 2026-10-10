"""Shared finding list. Every agent is mapped onto these names before voting."""

LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Fibrosis",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural Thickening",
    "Pneumonia",
    "Pneumothorax",
]

XRV_NAME = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Effusion": "Effusion",
    "Fibrosis": "Fibrosis",
    "Infiltration": "Infiltration",
    "Mass": "Mass",
    "Nodule": "Nodule",
    "Pleural Thickening": "Pleural_Thickening",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
}

NIH_NAMES = [
    "No Finding",
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

NIH_TO_LABEL = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Effusion": "Effusion",
    "Infiltration": "Infiltration",
    "Mass": "Mass",
    "Nodule": "Nodule",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Fibrosis": "Fibrosis",
    "Pleural_Thickening": "Pleural Thickening",
}

FUSION_AGENTS = ("cnn", "vit", "detector")

POSITIVE_THRESHOLD = 0.50
REFER_SCORE = 0.35
REFER_STD = 0.25
MLLM_MIX = 0.25
TRAIN_COUNT = 400
TEST_COUNT = 80
DATASET_ID = "Sohaibsoussi/NIH-Chest-X-ray-dataset-small"
