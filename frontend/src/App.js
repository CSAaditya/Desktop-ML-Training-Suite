import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [columns, setColumns] = useState([]);
  const [target, setTarget] = useState('');
  const [features, setFeatures] = useState([]);
  const [algorithm, setAlgorithm] = useState('LinearRegression');
  const [trainingResult, setTrainingResult] = useState(null);
  const [predictionInput, setPredictionInput] = useState({});
  const [predictionResult, setPredictionResult] = useState(null);
  const [featureInfo, setFeatureInfo] = useState(null);
  const [savedModels, setSavedModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [requiredFeatures, setRequiredFeatures] = useState([]);

  useEffect(() => {
    fetchSavedModels();
  }, []);

  useEffect(() => {
    if (selectedModel) {
      fetchModelFeatures(selectedModel);
    }
  }, [selectedModel]);

  const fetchSavedModels = async () => {
    try {
      const response = await axios.get(`${API_URL}/list_models`);
      setSavedModels(response.data.models);
    } catch (error) {
      console.error('Error fetching saved models:', error);
    }
  };

  const fetchModelFeatures = async (modelName) => {
    try {
      const response = await axios.get(`${API_URL}/model_features/${modelName}`);
      setRequiredFeatures(response.data.features);
      // Initialize predictionInput with empty strings for each feature
      setPredictionInput(response.data.features.reduce((acc, feature) => {
        acc[feature] = '';
        return acc;
      }, {}));
    } catch (error) {
      console.error('Error fetching model features:', error);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    console.log("File selected:", file.name);
    
    const reader = new FileReader();
    reader.onload = async (event) => {
      const content = event.target.result;
      console.log("File content loaded, first 100 characters:", content.substring(0, 100));
      
      try {
        console.log("Sending file to server...");
        
        const response = await axios.post(`${API_URL}/upload_file`, {
          filename: file.name,
          content: content
        }, {
          headers: {
            'Content-Type': 'application/json'
          },
          maxContentLength: Infinity,
          maxBodyLength: Infinity
        });
        
        console.log("Server response:", response.data);
        if (response.data.columns && response.data.columns.length > 0) {
          setColumns(response.data.columns);
          console.log("Columns set:", response.data.columns);
        } else {
          console.error("No columns received from server");
        }
      } catch (error) {
        console.error('Error uploading file:', error);
        if (error.response) {
          console.error('Response data:', error.response.data);
          console.error('Response status:', error.response.status);
          console.error('Response headers:', error.response.headers);
        } else if (error.request) {
          console.error('No response received:', error.request);
        } else {
          console.error('Error setting up request:', error.message);
        }
      }
    };
    reader.readAsText(file);
  };

  const handleTrainModel = async () => {
    try {
      const response = await axios.post(`${API_URL}/train_model`, {
        algorithm,
        target,
        features,
        params: {}
      });
      setTrainingResult(response.data);
      fetchSavedModels();  // Refresh the list of saved models
    } catch (error) {
      console.error('Error training model:', error);
      if (error.response) {
        console.error('Response data:', error.response.data);
      }
      alert('Error training model: ' + (error.response?.data?.detail || error.message));
    }
  };

  const handlePredict = async () => {
    if (!selectedModel) {
      alert('Please select a model for prediction');
      return;
    }
    try {
      console.log('Sending prediction data:', predictionInput);
      const response = await axios.post(`${API_URL}/predict/${selectedModel}`, {
        features: predictionInput
      });
      console.log('Prediction response:', response.data);
      setPredictionResult(response.data.prediction);
    } catch (error) {
      console.error('Error making prediction:', error);
      if (error.response) {
        console.error('Response data:', error.response.data);
        alert('Error making prediction: ' + JSON.stringify(error.response.data));
      } else {
        alert('Error making prediction: ' + error.message);
      }
    }
  };

  const handleFeatureInfo = async (feature) => {
    try {
      const response = await axios.get(`${API_URL}/feature_info?feature=${feature}`);
      setFeatureInfo(response.data);
    } catch (error) {
      console.error('Error getting feature info:', error);
      alert('Error getting feature info: ' + error.message);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>ML Interface</h1>
        <p className="App-description">
          Welcome to the Machine Learning Interface. This application allows you to upload data, 
          train machine learning models, save them for future use, and make predictions. You can choose between Linear Regression 
          and Random Forest Classifier algorithms to analyze your data.
        </p>
      </header>

      <main className="App-main">
        <section className="App-section">
          <h2>Upload Data</h2>
          <input type="file" onChange={handleFileUpload} className="App-file-input" />
          {columns.length > 0 && (
            <p className="App-info">Columns loaded: {columns.join(', ')}</p>
          )}
        </section>

        {columns.length > 0 && (
          <section className="App-section">
            <h2>Train Model</h2>
            <div className="App-form-group">
              <label>
                Target:
                <select value={target} onChange={(e) => setTarget(e.target.value)} className="App-select">
                  <option value="">Select target</option>
                  {columns.map(col => <option key={col} value={col}>{col}</option>)}
                </select>
              </label>
            </div>
            <div className="App-form-group">
              <label>
                Features:
                <p><small>(Hold Ctrl/Cmd to select multiple)</small></p>
                <select 
                  multiple 
                  value={features} 
                  onChange={(e) => setFeatures(Array.from(e.target.selectedOptions, option => option.value))}
                  className="App-select App-select-multiple"
                >
                  {columns.map(col => <option key={col} value={col}>{col}</option>)}
                </select>
              </label>
            </div>
            {features.length > 0 && (
              <p className="App-info">Selected features: {features.join(', ')}</p>
            )}
            <div className="App-form-group">
              <label>
                Algorithm:
                <select value={algorithm} onChange={(e) => setAlgorithm(e.target.value)} className="App-select">
                  <option value="LinearRegression">Linear Regression</option>
                  <option value="RandomForestClassifier">Random Forest Classifier</option>
                </select>
              </label>
            </div>
            <button onClick={handleTrainModel} className="App-button">Train Model</button>
          </section>
        )}

        {trainingResult && (
          <section className="App-section">
            <h2>Training Result</h2>
            <pre className="App-pre">{JSON.stringify(trainingResult, null, 2)}</pre>
          </section>
        )}

        <section className="App-section">
          <h2>Make Prediction</h2>
          <div className="App-form-group">
            <label>
              Select Model:
              <select 
                value={selectedModel} 
                onChange={(e) => setSelectedModel(e.target.value)} 
                className="App-select"
              >
                <option value="">Select a model</option>
                {savedModels.map(model => (
                  <option key={model} value={model}>{model}</option>
                ))}
              </select>
            </label>
          </div>
          {requiredFeatures.map(feature => (
            <div key={feature} className="App-form-group">
              <label>
                {feature}:
                <input 
                  type="text" 
                  value={predictionInput[feature] || ''}
                  onChange={(e) => setPredictionInput(prev => ({...prev, [feature]: e.target.value}))}
                  className="App-input"
                />
                <button onClick={() => handleFeatureInfo(feature)} className="App-button App-button-secondary">View Info</button>
              </label>
            </div>
          ))}
          <button onClick={handlePredict} className="App-button">Predict</button>
        </section>

        {predictionResult && (
          <section className="App-section">
            <h2>Prediction Result</h2>
            <pre className="App-pre">{JSON.stringify(predictionResult, null, 2)}</pre>
          </section>
        )}

        {featureInfo && (
          <section className="App-section">
            <h2>Feature Info: {featureInfo.feature}</h2>
            {featureInfo.type === 'categorical' ? (
              <div>
                <h3>Unique Values:</h3>
                <ul className="App-list">
                  {featureInfo.unique_values.map((value, index) => (
                    <li key={index}>{value}</li>
                  ))}
                </ul>
                <h3>Top Values:</h3>
                <ul className="App-list">
                  {Object.entries(featureInfo.top_values).map(([value, count], index) => (
                    <li key={index}>{value}: {count}</li>
                  ))}
                </ul>
              </div>
            ) : (
              <div>
                <p>Minimum: {featureInfo.min}</p>
                <p>Maximum: {featureInfo.max}</p>
                <p>Mean: {featureInfo.mean}</p>
                <p>Median: {featureInfo.median}</p>
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  );
}

export default App;