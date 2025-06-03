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

-   **Frontend**: Streamlit (Interactive web application)
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

Effective feature engineering was crucial for capturing complex relationships in the property data. Here are some examples of engineered features:

*   **`Price_Per_SqFt`**: Calculated as `Price / Square_Footage`. **Reason**: Provides a normalized measure of property value, allowing for fair comparison across properties of different sizes.
*   **`Num_Amenities`**: Count of amenities listed for a property. **Reason**: Indicates the richness of features offered, which can significantly influence desirability.
*   **`Age_of_Property`**: Derived from `Year_Built` and current year. **Reason**: Older properties might have different maintenance needs or historical appeal, influencing buyer interest.
*   **`Distance_to_City_Center`** (Hypothetical, if location data available): Calculated based on coordinates. **Reason**: Proximity to urban centers often correlates with property value and demand.
*   **`Is_Luxury_Property`**: Binary flag based on high price, specific amenities, or location. **Reason**: Helps the model identify high-end properties, catering to specific user segments.

*(Please expand this section with actual features you engineered and their rationale.)*

## 🧠 Modeling

### Model Selection

We employed **XGBoost (Extreme Gradient Boosting)** for our recommendation model.

*   **Why XGBoost?**:
    *   **Performance**: Known for its high performance and speed, especially on structured data.
    *   **Robustness**: Handles various data types and missing values effectively.
    *   **Interpretability**: Integrates well with SHAP for understanding feature contributions.
    *   **Scalability**: Efficient for large datasets, suitable for real estate data.

### Training & Evaluation

The model was trained on a comprehensive dataset of property listings.
*   **Training**: We used a [e.g., 80/20 train-test split] approach to prepare the data. [Mention any cross-validation strategy, e.g., K-fold cross-validation]. Hyperparameter tuning was performed using [e.g., GridSearchCV] to optimize model performance.
*   **Evaluation**: The model's performance was evaluated using metrics such as [e.g., Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R-squared for regression tasks, or Precision, Recall, F1-score if framed as a classification/ranking task].

*(Please update this section with your specific training details, evaluation metrics, and any other models you experimented with.)*

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

The model achieved compelling results in recommending properties.

*   **Key Metrics**:
    *   **[Metric 1]**: [Value] (e.g., MAE: 0.05, indicating predictions are, on average, within 5% of actual values).
    *   **[Metric 2]**: [Value] (e.g., R-squared: 0.85, meaning 85% of the variance in property prices can be explained by our model).

*   **SHAP Analysis Insights**:
    *   SHAP values revealed that `Price_Per_SqFt`, `Num_Bedrooms`, and `Location_Score` were consistently the most influential features in property valuation and recommendation.
    *   [Add other specific insights from your SHAP analysis here].

*(Please fill in your actual evaluation results and specific SHAP insights.)*

## 🧠 Key Decisions & Challenges

Developing this system involved several critical decisions and overcoming notable challenges:

*   **Challenge 1: Data Sparsity & Missing Values**
    *   **Decision**: Opted for a hybrid approach involving imputation (e.g., median for numerical, mode for categorical) and strategic removal of rows with excessive missing data, balancing data integrity with dataset size.
    *   **Learning**: Understanding the impact of different imputation strategies on model performance.

*   **Challenge 2: Feature Engineering Complexity**
    *   **Decision**: Focused on creating domain-specific features (`Price_Per_SqFt`, `Num_Amenities`) that directly relate to real estate valuation, rather than relying solely on raw features.
    *   **Learning**: The significant uplift in model performance achievable through well-thought-out feature engineering.

*   **Challenge 3: Model Interpretability**
    *   **Decision**: Integrated SHAP values early in the development cycle to ensure the model's predictions could be explained, which is crucial for trust in recommendation systems.
    *   **Learning**: How SHAP helps in debugging model errors and gaining stakeholder confidence.

*   **Decision 4: Streamlit for Rapid Prototyping**
    *   **Reason**: Chosen for its ability to quickly build interactive web applications with pure Python, allowing for rapid iteration and demonstration of the recommendation system.
    *   **Learning**: Streamlit's simplicity accelerated the deployment phase significantly.

*(Please customize this section with your actual key decisions, challenges, and learnings.)*

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

## 👥 Authors

-   **Raahim Khan** - Initial work & Core Development - [Link to your GitHub/LinkedIn]

