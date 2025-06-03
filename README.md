# 🏡 Property Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Project Overview

This project develops an intelligent **Property Recommendation System** leveraging machine learning to connect users with their ideal properties. It addresses the challenge of sifting through vast real estate data by providing personalized suggestions based on various property features and user preferences. The system integrates data processing, advanced feature engineering, robust machine learning modeling, and an interactive web application for a seamless user experience.

## 🚀 Features

-   **Data Cleaning & Preprocessing**: Robust handling of missing values, duplicates, and standardization.
-   **Feature Engineering & Selection**: Creation of impactful new features and selection of the most relevant ones.
-   **Machine Learning Model Training**: Development and evaluation of a predictive recommendation model.
-   **Interactive Web Application**: A user-friendly Streamlit interface for real-time recommendations.
-   **Model Interpretability (SHAP)**: Insights into model predictions using SHAP values.
-   **Data Visualization**: Comprehensive visualizations for deeper data understanding.

## 🛠️ Tech Stack

-   **Frontend**: Streamlit (Interactive web application) with HTML/CSS
-   **Backend**: Python
-   **Data Processing**: Pandas, NumPy (Efficient data manipulation)
-   **Machine Learning**: XGBoost (Powerful gradient boosting), Scikit-learn (Machine learning utilities)
-   **Visualization**: Plotly (Interactive plots and dashboards)
-   **Model Interpretation**: SHAP (Explainable AI)

## 📁 Project Structure

```
Property-Recommendation/
├── data/                       # 📊 Raw and processed dataset files
├── model/                     # 🧠 Trained machine learning models (e.g., XGBoost.pkl)
├── script/                    # 🐍 Core source code and notebooks
│   ├── app.py                        # 🌐 Streamlit web application for recommendations
│   ├── data_cleaning.ipynb           # 🧹 Notebook for initial data cleaning and preprocessing
│   ├── data_processing.ipynb         # ⚙️ Notebook for data transformation and validation
│   ├── feature_engineering.ipynb     # 🚀 Notebook for creating and selecting new features
│   ├── model.ipynb                   # 📈 Notebook for model training, tuning, and evaluation
│   └── requirements.txt              # 📦 Python package dependencies
└── script_info/               # 📝 Additional scripts and documentation
```

## 📊 Data Processing Pipeline

Our robust data pipeline ensures high-quality data feeds into the recommendation engine:

1.  **Data Cleaning** (`script/data_cleaning.ipynb`)
    *   **Purpose**: To refine raw data by handling inconsistencies and preparing it for analysis.
    *   **Processes**:
        *   Identification and imputation/removal of missing values.
        *   Detection and elimination of duplicate property entries.
        *   Standardization of data formats across various features.

2.  **Data Processing** (`script/data_processing.ipynb`)
    *   **Purpose**: To transform cleaned data into a structured format suitable for feature engineering and modeling.
    *   **Processes**:
        *   Extraction of relevant information from raw text fields (e.g., amenities lists).
        *   Normalization and scaling of numerical features.
        *   Validation checks to ensure data integrity and consistency.

3.  **Feature Engineering** (`script/feature_engineering.ipynb`)
    *   **Purpose**: To create new, informative features that enhance the predictive power of the model.
    *   **Processes**:
        *   Generation of interaction terms between existing features.
        *   Creation of categorical indicators from text descriptions.
        *   Application of feature scaling techniques (e.g., StandardScaler, MinMaxScaler).

4.  **Model Training** (`script/model.ipynb`)
    *   **Purpose**: To train, tune, and evaluate the machine learning model for property recommendations.
    *   **Processes**:
        *   Selection of the optimal machine learning algorithm (e.g., XGBoost).
        *   Hyperparameter tuning using techniques like GridSearchCV or RandomizedSearchCV.
        *   Cross-validation for robust model evaluation.

5.  **Streamlit Application** (`script/app.py`)
    *   **Purpose**: To serve the trained model through an interactive web interface.
    *   **Processes**:
        *   Loads the pre-trained model and preprocessing pipelines.
        *   Takes user inputs (preferences, property criteria).
        *   Generates and displays property recommendations dynamically.
        *   Visualizes feature importance and other insights.

## 📈 Feature Engineering

Effective feature engineering was crucial for capturing the most relevant similarities between the subject property and candidate comparables. Here are the engineered features used in the model:

* **`gla_diff`**: Difference in Gross Living Area (GLA) between subject and candidate.  
  **Reason**: Properties with similar living area are more comparable in valuation.

* **`lot_size_diff`**: Difference in lot size between subject and candidate.  
  **Reason**: Lot size impacts property value and is a key metric for comparison.

* **`bedroom_diff`**: Difference in number of bedrooms.  
  **Reason**: Number of bedrooms is a fundamental factor in property similarity and buyer interest.

* **`bathroom_diff`**: Difference in number of bathrooms.  
  **Reason**: Bathrooms, like bedrooms, are core to property utility and comparability.

* **`room_count_diff`**: Difference in total room count.  
  **Reason**: Overall room count gives an additional measure of property size and use.

* **`same_property_type`**: Boolean flag (1/0) if subject and candidate have the same property type.  
  **Reason**: Comparing the same property type ensures more meaningful valuation (e.g., house to house, condo to condo).

* **`same_storey_type`**: Boolean flag if both properties have the same number/type of stories (e.g., both are 2-storey homes).  
  **Reason**: Storey type affects structure, value, and buyer preference.

* **`sold_recently_90`**: Boolean flag if candidate was sold within 90 days of the subject’s valuation date.  
  **Reason**: Recent sales provide more reliable market comparisons due to similar market conditions.

*(These features were selected based on appraisal best practices and confirmed by SHAP analysis as the most influential drivers for the model’s recommendations.)*

## 🧠 Modeling

### Model Selection

We employed **XGBoost (Extreme Gradient Boosting)** for our recommendation model.

*   **Why XGBoost?**:
    *   **Performance**: Known for its high performance and speed, especially on structured data.
    *   **Robustness**: Handles various data types and missing values effectively.
    *   **Interpretability**: Integrates well with SHAP for understanding feature contributions.
    *   **Scalability**: Efficient for large datasets, suitable for real estate data.

### Training & Evaluation

The model was trained using an 80/20 train–test split on a curated property dataset, with group-aware splitting to prevent data leakage. Hyperparameter tuning was done via GridSearchCV.

* **Evaluation**: Performance was measured by Top-3 Hit Rate (94.4%), ROC AUC (0.93), and overall accuracy (98%), with additional reporting of precision, recall, and F1-score for both classes.
* **Interpretability**: SHAP analysis highlighted GLA Difference, Lot Size Difference, Bedroom Difference, and Sold Within 90 Days as the most influential features driving recommendations.

*(All metrics and insights summarized using ChatGPT/OpenAI for clarity.)*


## 🚀 How to Run

Follow these steps to set up and run the Property Recommendation System locally.

### Prerequisites

*   Python 3.8+
*   Git

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/Property-Recommendation.git
    cd Property-Recommendation
    ```

2.  **Create and activate a virtual environment**:
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    venv\Scripts\activate
    ```

3.  **Install required packages**:
    All project dependencies are listed in `script/requirements.txt`.
    ```bash
    pip install -r script/requirements.txt
    ```

### Usage

#### 1. Run the Jupyter Notebooks (for data processing and model training)

Navigate to the `script/` directory and open the `.ipynb` files in sequence using Jupyter Lab or Jupyter Notebook to understand the data pipeline:

```bash
# After activating your virtual environment
cd script/
jupyter lab # or jupyter notebook
```
Follow the notebooks in this order:
*   `data_cleaning.ipynb`
*   `data_processing.ipynb`
*   `feature_engineering.ipynb`
*   `model.ipynb`

#### 2. Run the Streamlit Web Application

After training the model (by running `model.ipynb` or ensuring a trained model is saved in `model/`), you can launch the interactive web application:

```bash
# Ensure you are in the root directory of the project: Property-Recommendation/
streamlit run script/app.py
```
Open your web browser and navigate to the local URL provided by Streamlit (typically `http://localhost:8501`).

## ✅ Results

The model demonstrated strong performance in recommending comparable properties:

*   **Key Metrics**:
    *   **Top-3 Hit Rate**: Achieved a 94.4% hit rate (17/18), meaning at least one expert-chosen comp was present in the model’s top-3 ranked candidates for nearly all test subjects.
    *   **ROC AUC Score**: Scored 0.93, indicating excellent ability to distinguish true comps from non-comps.

*   **SHAP Analysis Insights**:
    *   SHAP (SHapley Additive exPlanations) analysis identified GLA Difference, Lot Size Difference, Bedroom Difference, and Sold Within 90 Days as the most influential features in          model recommendations.
    *   SHAP results confirmed the model’s reliance on core property characteristics and sale recency, closely matching human expert logic.

Summarization: All insights and recommendation summaries were distilled using ChatGPT (OpenAI) to ensure clarity and transparency.


## 🧠 Key Decisions & Challenges

Developing this system involved several critical decisions and overcoming notable challenges:

*   **Challenge 1: Data Sparsity & Missing Values**
    *   **Decision**: Opted for a hybrid approach involving imputation (e.g., median for numerical, mode for categorical) and strategic removal of rows with excessive missing data, balancing data integrity with dataset size.
    *   **Learning**: Understanding the impact of different imputation strategies on model performance.

*   **Challenge 2: Feature Engineering Complexity**
    *   **Decision**: Focused on creating domain-specific features (`gla_difference`, `property_type_same`) that directly relate to real estate valuation, rather than relying solely on raw features.
    *   **Learning**: The significant uplift in model performance achievable through well-thought-out feature engineering.

*   **Challenge 3: Model Interpretability**
    *   **Decision**: Integrated SHAP values early in the development cycle to ensure the model's predictions could be explained, which is crucial for trust in recommendation systems.
    *   **Learning**: How SHAP helps in debugging model errors and gaining stakeholder confidence.

*   **Decision 4: Streamlit for Rapid Prototyping**
    *   **Reason**: Chosen for its ability to quickly build interactive web applications with pure Python, allowing for rapid iteration and demonstration of the recommendation system.
    *   **Learning**: Streamlit's simplicity accelerated the deployment phase significantly.


## 🤝 Contributing

Contributions are always welcome! If you have suggestions for improvements or new features, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add Your Feature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Open a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ✍️ Authors

- [**Raahim Khan**](https://www.linkedin.com/in/raahimk24/) – Initial work & Core Development


