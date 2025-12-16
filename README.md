# Drug–Drug Interaction Prediction

This project predicts drug–drug interactions using machine learning.
It builds a drug interaction matrix, learns latent representations using SVD,
and trains classification models to predict interaction types.
The project also generates aspirin-centered interaction predictions.

## Requirements
- Python 3.9+
- macOS / Linux / Windows
- 8 GB RAM recommended (dataset is large)

## Dataset
Place the following files in the project directory:
DDI_data.csv
DDI_types.xlsx

Expected columns in 'DDI_data.csv':
- `drug1_name`
- `drug2_name`
- `interaction_type`

## Setup
1. Create and activate a virtual environment:
'''bash
python3 -m venv venv
source venv/bin/activate

3. Install dependencies:
pip install -r requirements.txt

4. Run the Pipeline
python drug-prediction.py

The script will:
- Clean and deduplicate the dataset
- Build a drug–drug interaction matrix
- Learn latent features using TruncatedSVD
- Train SVM and MLP classifiers
- Evaluate model performance
- Generate aspirin-centered predictions
