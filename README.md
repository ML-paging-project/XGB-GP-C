# XGB Training Set on Champ Sim Traces

## Installation
1. Clone this repository
2. Run get data bash script `./tools/get_data.sh`
3. Install Dependencies
  1. Create and activate virtual environment 
    ```
    python3 -m venv xgb_env
    source xgb_env/bin/activate
    ```
  2. Install python libraries `pip install -r requirements.txt`

## Running Experiment
1. Activate virtual environment `source xgb_env/bin/activate`
2. Train and test a model `python3 main_with_xgb.py`
3. (Optional) Deactivate virtual environment `deactivate`
