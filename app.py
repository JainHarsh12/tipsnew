import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

file_path = 'tips.csv'
data = pd.read_csv(file_path)

x = data.drop(columns='smoker')
y = data['smoker']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

numeric_features = x.select_dtypes(include=['int64','float64']).columns
categorical_features=x.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps = [
    ('imputer',SimpleImputer(strategy = 'mean')),
    ('scaler',StandardScaler())
]) 

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


model = DecisionTreeClassifier(random_state=42)


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])


pipeline.fit(x_train, y_train)


y_pred = pipeline.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

