import sys
sys.path.append(r"D:\Doccuments\GitHub\End-to-End-MLOps-Pipeline")

from pipeline.complete_pipeline_v1 import build_pipeline, save_pipeline
from xgboost import XGBClassifier
import pandas as pd

df = pd.read_csv(r"D:\Doccuments\GitHub\End-to-End-MLOps-Pipeline\data\raw\fraud test.csv")
data = df[df.columns[1:]].copy()
target = "is_fraud"
X = data.drop(columns=[target])
y = data[target]
split_ratio = 0.8
split_index = int(len(data) * split_ratio)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
print(scale_pos_weight)

model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42)

pipe = build_pipeline(model)
print(X_train.columns)
pipe.fit(X_train, y_train)

save_pipeline(pipe, "pipeline.pkl")