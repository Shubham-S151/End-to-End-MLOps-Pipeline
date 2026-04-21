import os
import sys
import json
import zipfile
import pandas as pd
import streamlit as st

# D:\Doccuments\GitHub\End-to-End-MLOps-Pipeline\data\interim\churn_data
ROOT_DIR = os.path.abspath(os.path.dirname("End-to-End-MLOps-Pipeline"))
sys.path.append(ROOT_DIR)

# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..."))
# BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# sys.path.append(BASE_DIR)

print(ROOT_DIR)

zip_data_path = os.path.join(ROOT_DIR,"data","raw_for_webuse","Data.zip")

with open(zip_data_path,'wb') as zipf:
    files = zipfile.ZipExtFile.read(zipf)

print(files)