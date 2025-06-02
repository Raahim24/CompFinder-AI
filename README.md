# Property Recommendation System

A machine learning-based property recommendation system that helps users find their ideal properties based on various features and preferences.

## 📋 Overview

This project implements a property recommendation system using machine learning techniques. It processes property data, performs feature engineering, and builds a recommendation model to suggest properties based on user preferences and property characteristics.

## 🚀 Features

- Data cleaning and preprocessing
- Feature engineering and selection
- Machine learning model training
- Interactive web application for property recommendations
- SHAP value analysis for model interpretability
- Data visualization capabilities

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: XGBoost, Scikit-learn
- **Visualization**: Plotly
- **Model Interpretation**: SHAP

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Property-Recommendation.git
cd Property-Recommendation
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r script/requirements.txt
```

## 📁 Project Structure

```
Property-Recommendation/
├── data/               # Dataset files
├── model/             # Trained models
├── script/            # Source code
│   ├── app.py                # Streamlit web application
│   ├── data_cleaning.ipynb   # Data cleaning notebook
│   ├── data_processing.ipynb # Data processing notebook
│   ├── feature_engineering.ipynb # Feature engineering notebook
│   ├── model.ipynb           # Model training notebook
│   └── requirements.txt      # Project dependencies
└── script_info/       # Additional scripts and information
```

## 🚀 Usage

1. Start the Streamlit application:
```bash
streamlit run script/app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Use the interactive interface to:
   - Input property preferences
   - View property recommendations
   - Analyze feature importance
   - Visualize property data

## 📊 Data Processing Pipeline

1. **Data Cleaning** (`data_cleaning.ipynb`)
   - Handle missing values
   - Remove duplicates
   - Standardize data formats

2. **Data Processing** (`data_processing.ipynb`)
   - Feature extraction
   - Data normalization
   - Data validation

3. **Feature Engineering** (`feature_engineering.ipynb`)
   - Create new features
   - Feature selection
   - Feature scaling

4. **Model Training** (`model.ipynb`)
   - Model selection
   - Hyperparameter tuning
   - Model evaluation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- Thanks to all contributors who have helped shape this project
- Special thanks to the open-source community for the amazing tools and libraries