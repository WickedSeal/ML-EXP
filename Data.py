from sklearn.datasets import load_iris
import pandas as pd

data = load_iris()                       # 1. Load dataset

df = pd.DataFrame(data.data, columns=data.feature_names)  # 2. Convert to table
df['Species'] = data.target_names[data.target]

print(df)                                # 3. Print dataset