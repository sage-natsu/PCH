#!/bin/bash
# Update and install necessary tools
apt-get update && apt-get install -y python3-dev build-essential

# Create Conda environment
conda env create -f conda.yml

# Activate the environment
source activate reddit_dashboard_env

# Ensure correct Python version and dependencies
pip install -r requirements.txt


# Run Streamlit
streamlit run app.py
