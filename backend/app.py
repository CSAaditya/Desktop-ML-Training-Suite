
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import joblib
import os
from io import StringIO
import traceback
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://ml4dummies.vercel.app", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class DataFile(BaseModel):
    filename: str
    content: str

class TrainingParams(BaseModel):
    algorithm: str
    target: str
    features: List[str]
    params: dict
    test_size: float = 0.2  

class PredictionInput(BaseModel):
    features: Dict[str, Any]

data = None
model = None
preprocessor = None
saved_models = {}

def preprocess_data(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    categorical_columns = df.select_dtypes(include=['object']).columns
    df[categorical_columns] = df[categorical_columns].fillna('Unknown')
    return df

@app.post("/upload_file")
async def upload_file(file: DataFile):
    global data
    try:
        print(f"Received file: {file.filename}")
        print(f"Content preview: {file.content[:100]}")
        data = pd.read_csv(StringIO(file.content))
        data = preprocess_data(data)
        print(f"Data shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        return {"message": "File uploaded successfully", "columns": data.columns.tolist()}
    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/get_columns")
async def get_columns():
    if data is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    return {"columns": data.columns.tolist()}

@app.post("/train_model")
async def train_model(params: TrainingParams):
    global data, model, preprocessor, saved_models
    if data is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    try:
        print(f"Training model with params: {params}")
        X = data[params.features]
        y = data[params.target]

        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"X dtypes: {X.dtypes}")
        print(f"y dtype: {y.dtype}")

        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'bool']).columns

        print(f"Numeric features: {numeric_features}")
        print(f"Categorical features: {categorical_features}")

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        X_processed = preprocessor.fit_transform(X)

        print(f"X_processed shape: {X_processed.shape}")

        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=params.test_size, random_state=42)

        if params.algorithm == 'LinearRegression':
            model = LinearRegression(**params.params)
        elif params.algorithm == 'RandomForestClassifier':
            model = RandomForestClassifier(**params.params)
        elif params.algorithm == 'GradientBoostingClassifier':
            model = GradientBoostingClassifier(**params.params)
        elif params.algorithm == 'SVC':
            model = SVC(**params.params)
        elif params.algorithm == 'LogisticRegression':
            model = LogisticRegression(**params.params)
        else:
            raise HTTPException(status_code=400, detail="Unsupported algorithm")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        
        # Save the model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{params.algorithm}_{params.target}_{timestamp}"
        model_path = f"saved_models/{model_name}.joblib"
        os.makedirs("saved_models", exist_ok=True)
        joblib.dump((model, preprocessor), model_path)
        
        saved_models[model_name] = {
            "algorithm": params.algorithm,
            "target": params.target,
            "features": params.features,
            "path": model_path
        }

        
        if isinstance(model, (LinearRegression, LogisticRegression)):
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            return {"mse": float(mse), "r2": float(r2), "model_name": model_name}
        else:
            accuracy = accuracy_score(y_test, y_pred)
            return {"accuracy": float(accuracy), "model_name": model_name}

    except Exception as e:
        print(f"Error in train_model: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/list_models")
async def list_models():
    return {"models": list(saved_models.keys())}

@app.post("/predict/{model_name}")
async def predict(model_name: str, input_data: PredictionInput):
    if model_name not in saved_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        model_info = saved_models[model_name]
        model, preprocessor = joblib.load(model_info["path"])
        
        # Create a DataFrame with the input data
        input_df = pd.DataFrame([input_data.features])
        
        print(f"Original input data: {input_df}")  # Debug print
        
        # Ensure all expected features are present
        for feature in model_info["features"]:
            if feature not in input_df.columns:
                input_df[feature] = np.nan  # or some appropriate default value
        
        # Only keep the features used during training
        input_df = input_df[model_info["features"]]
        
        print(f"Processed input data: {input_df}")  # Debug print
        
        # Apply the preprocessor
        input_processed = preprocessor.transform(input_df)
        
        print(f"Preprocessed input shape: {input_processed.shape}")  # Debug print
        
        prediction = model.predict(input_processed)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        print(f"Error in prediction: {str(e)}")  # Debug print
        raise HTTPException(status_code=422, detail=str(e))

@app.get("/feature_info")
async def get_feature_info(feature: str):
    if data is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    try:
        feature_data = data[feature]
        if pd.api.types.is_categorical_dtype(feature_data) or pd.api.types.is_object_dtype(feature_data):
            value_counts = feature_data.value_counts().head(10).to_dict()
            unique_values = feature_data.unique().tolist()
            return {
                "feature": feature,
                "type": "categorical",
                "unique_values": unique_values,
                "top_values": value_counts,
                "null_count": int(feature_data.isnull().sum())
            }
        else:
            return {
                "feature": feature,
                "type": "numerical",
                "min": float(feature_data.min()),
                "max": float(feature_data.max()),
                "mean": float(feature_data.mean()),
                "median": float(feature_data.median()),
                "null_count": int(feature_data.isnull().sum())
            }
    except Exception as e:
        print(f"Error processing feature {feature}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing feature {feature}: {str(e)}")

@app.get("/model_features/{model_name}")
async def get_model_features(model_name: str):
    if model_name not in saved_models:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"features": saved_models[model_name]["features"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
