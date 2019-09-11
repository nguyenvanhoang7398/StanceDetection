Text Preprocessing
=======================
This project pre-processes text datasets for classification task in https://github.com/nguyenvanhoang7398/TextClassification

## Quick start

### Preprocess FakeNewsChallenge's stance detection dataset
```
python main.py --dataset-name=stance_fnc
```

## CLI Usage
```
python main.py  [--config-path <path-to-config-file>]
                [--dataset-name <name-of-dataset>]

parameters:
    --config-path           default: "config/config.json"
    --dataset-name          default: None                   stance_fnn|stance_fnc
```
