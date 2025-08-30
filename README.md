# Desktop ML Training Suite

A web-based application for training machine learning models through a graphical interface. Built with React and FastAPI.
Used to explore various ML algorithms quickly through a simple web interface.
Designed to help users train and save machine learning models and make predictions
## Project Structure
```bash
.
├── .gitignore
├── README.md
├── backend
│   ├── __pycache__
│   ├── app.py
│   ├── requirements.txt
│   ├── saved_models
│   └── venv
└── frontend
    ├── README.md
    ├── build
    ├── node_modules
    ├── package-lock.json
    ├── package.json
    ├── public
    └── src
```


## Features

- Upload CSV files and automatically detect columns
- Train models using multiple algorithms
- Save trained models with timestamps
- Make predictions on new data
- View feature statistics before training

## Supported Algorithms

- Linear Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Classifier
- Logistic Regression

## Tech Stack

**Backend:** FastAPI, scikit-learn, pandas, numpy

**Frontend:** React, Axios

## Installation

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd frontend
npm install
```

## Running the Application

1. Start backend (Terminal 1):
```bash
cd backend
source venv/bin/activate
uvicorn app:app --reload
```

2. Start frontend (Terminal 2):
```bash
cd frontend
npm start
```

3. Open http://localhost:3000 in your browser

## Usage

1. Upload a CSV file
2. Select your target variable
3. Choose features for training
4. Pick an algorithm
5. Train the model
6. Use saved models for predictions

## API Endpoints

- `POST /upload_file` - Upload CSV
- `POST /train_model` - Train model
- `GET /list_models` - List saved models
- `POST /predict/{model_name}` - Make predictions
- `GET /feature_info` - Get feature statistics

## Author

Aaditya Aswadhati - UC San Diego
