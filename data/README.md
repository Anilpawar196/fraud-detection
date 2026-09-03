# Data

Raw data is **not** committed to this repository. The IEEE-CIS Fraud Detection
dataset is distributed under Kaggle competition rules and is ~1.3 GB.

## Expected layout

```
data/
├── raw/                     # place the Kaggle files here, unmodified
│   ├── train_transaction.csv
│   ├── train_identity.csv
│   ├── test_transaction.csv
│   └── test_identity.csv
├── interim/                 # joined + validated frames (parquet)
└── processed/               # engineered feature matrices + split indices
```

## How to obtain

The dataset comes from the
[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
competition. You must accept the competition rules on Kaggle before
downloading. Either download through the Kaggle UI and unzip into `raw/`, or
use the Kaggle CLI:

```bash
kaggle competitions download -c ieee-fraud-detection -p data/raw && unzip -o data/raw/ieee-fraud-detection.zip -d data/raw
```

## Provenance

`interim/` and `processed/` are fully reproducible from `raw/` by running the
pipeline scripts — never edit them by hand. Every artifact written there
records the config hash and random seed used to produce it.
