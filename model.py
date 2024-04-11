import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

'''Data Processor class is in charge of Data processing'''
class DataProcessor:
    def __init__(self, url, include_features):
        self.url = url
        self.include_features = include_features
    
    def load_data(self):
        df = pd.read_csv(self.url)
        return df[self.include_features]

    def preprocess_data(self, df):
        '''Fill NaN values with 0 on columns'''
        df.fillna(0, inplace=True)
        '''Identify categorical values on columns'''
        categoricals = [col for col, col_type in df.dtypes.items() if col_type == 'O']
        '''Codify categorical columns with dummy variables'''
        return pd.get_dummies(df, columns=categoricals, dummy_na=True)
    
'''ModelTrainer class is in charge of Model Training'''
class ModelTrainer:
    def __init__(self, model):
        self.model = model
    
    def train_model(self, x, y):
        self.model.fit(x, y)

    def save_model(self, filename):
        joblib.dump(self.model, filename)
        print("Model dumped!")

'''ModelLoader class is in charge of Model loading/saving'''
class ModelLoader:
    @staticmethod
    def load_model(filename):
        return joblib.load(filename)

    @staticmethod
    def save_model_columns(x, filename):
        model_columns = list(x.columns)
        joblib.dump(model_columns, filename)
        print("Models columns dumped!")

'''Define parameters'''
url = "https://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
include = ['Age', 'Sex', 'Embarked', 'Survived']
dependent_variable = 'Survived'

'''Instantiate objects'''
data_processor = DataProcessor(url, include)
model_trainer = ModelTrainer(LogisticRegression())

'''Load and preprocess data'''
data = data_processor.load_data()
preprocessed_data = data_processor.preprocess_data(data)

'''Prepare features and target'''
x = preprocessed_data[preprocessed_data.columns.difference([dependent_variable])]
y = preprocessed_data[dependent_variable]

'''Train model'''
model_trainer.train_model(x, y)

'''Save model and model columns'''
model_trainer.save_model('model.pkl')
ModelLoader.save_model_columns(x, 'model_columns.pkl')

'''Load model'''
loaded_model = ModelLoader.load_model('model.pkl')
