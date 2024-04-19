import pandas as pd
from sklearn.linear_model import LogisticRegression  # Importa LogisticRegression

import mlflow
import mlflow.sklearn

class DataProcessor:
    def __init__(self, url, include_features):
        self.url = url
        self.include_features = include_features
    
    def load_data(self):
        df = pd.read_csv(self.url)
        return df[self.include_features]

    def preprocess_data(self, df):
        df.fillna(0, inplace=True)
        categoricals = [col for col, col_type in df.dtypes.items() if col_type == 'O']
        return pd.get_dummies(df, columns=categoricals, dummy_na=True)
    
class ModelTrainer:
    def __init__(self, model):
        self.model = model
    
    def train_model(self, x, y):
        with mlflow.start_run():
            self.model.fit(x, y)
            mlflow.sklearn.log_model(self.model, "model")

'''Define parameters'''
url = "https://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
include = ['Age', 'Sex', 'Embarked', 'Survived']
dependent_variable = 'Survived'

'''Instantiate objects'''
data_processor = DataProcessor(url, include)

'''Load and preprocess data'''
data = data_processor.load_data()
preprocessed_data = data_processor.preprocess_data(data)

'''Prepare features and target'''
x = preprocessed_data[preprocessed_data.columns.difference([dependent_variable])]
y = preprocessed_data[dependent_variable]

'''Instantiate model'''
model = LogisticRegression()

'''Train model'''
model_trainer = ModelTrainer(model)
model_trainer.train_model(x, y)
