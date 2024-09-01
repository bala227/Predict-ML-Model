# Automatic ML Streamlit Application

## Overview

This Streamlit application provides an automated machine learning pipeline. It allows users to upload a dataset, perform data profiling, set up a machine learning environment, and download the best-performing model. The app leverages PyCaret for machine learning.

## Features

- **Upload**: Upload a CSV file containing the dataset for analysis and modeling.
- **Profile**: Generate a comprehensive data profiling report using `ydata_profiling`.
- **ML**: Set up a PyCaret machine learning environment, compare models, and save the best model.
- **Download**: Download the saved best-performing model as a pickle file.

## Requirements

To run this application, you need the following Python packages:

- `streamlit`
- `pandas`
- `streamlit_pandas_profiling`
- `ydata_profiling`
- `pycaret`

You can install the required packages using pip:

```bash
pip install streamlit pandas streamlit-pandas-profiling ydata-profiling pycaret
