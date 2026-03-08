# TS-ACADEMY-CAPSTONE-PROJECT

## Customer Churn Prediction (Telco Dataset)

## Project Overview

Customer churn prediction is a critical task for companies that rely on subscription-based services. Retaining existing customers is often more cost-effective than acquiring new ones, making churn analysis an important area of business analytics.

This project applies "supervised machine learning techniques" to predict whether a telecom customer is likely to churn based on demographic information, service usage, and account-related features.

The objective is to identify patterns associated with customer churn and build a predictive model capable of identifying customers at risk of leaving the service.

### Dataset

The dataset used in this project is the "Telco Customer Churn dataset", which contains information about customers of a telecommunications company.

Features in the Dataset

The dataset includes several types of variables:

Customer Demographics

1. Gender
2. Senior Citizen status
3. Partner
4. Dependents

### Service Information

1. Phone service
2. Internet service
3. Online security
4. Online backup
5. Device protection
6. Streaming TV
7. Streaming movies
   
### Account Information

1. Contract type
2. Payment method
3. Monthly charges
4. Total charges
5. Tenure
   
### Target Variable

Churn

Yes → Customer left the company

No → Customer stayed

### Project Workflow

The project follows a structured "machine learning workflow".

Data Preparation and Cleaning

Initial preprocessing steps included:

1. Loading the dataset using Pandas
2. Inspecting data structure and data types
3. Handling missing or inconsistent values
4. Preparing features for modeling
   

Exploratory Data Analysis (EDA)

Exploratory data analysis was performed to understand patterns within the dataset.

Univariate Analysis (Numeric Variables)

Numeric variables such as:

1. Tenure
2. Monthly Charges
3. Total Charges
   
were analyzed to understand their distributions.

Visualization techniques included:

1. Histograms
2. Distribution plots

   
Univariate Analysis (Categorical Variables)

Categorical variables such as:

1. Gender
2. Contract Type
3. Payment Method
4. Internet Service
   
were analyzed using "count plots" to observe the frequency distribution of each category.

These analyses helped identify potential relationships between customer characteristics and churn behavior.

Feature Engineering and Preprocessing

To prepare the dataset for machine learning, preprocessing techniques were applied:

Column Transformer

A "ColumnTransformer" was used to apply different transformations to numeric and categorical features.

Scaling

Numeric features were scaled using "StandardScaler" to normalize feature ranges.

Encoding

Categorical variables were converted into numerical form using "One-Hot Encoding".

Model Development

A "Logistic Regression classifier" was used as the baseline model.

The model was implemented using a Scikit-learn Pipeline, which combines preprocessing and model training into a single workflow.

Pipeline Components

1. Data preprocessing
2. Feature scaling and encoding
3. Logistic Regression model
   
Using pipelines improves reproducibility and ensures consistent preprocessing during training and evaluation.

Model Evaluation

The dataset was split into "training and testing sets" using train_test_split.

The model's performance was evaluated using standard classification metrics:

1. Accuracy – Overall correctness of the model
2. Precision – Correct positive predictions
3. Recall – Ability to detect churn cases
4. F1 Score – Balance between precision and recall
5. Confusion Matrix – Detailed classification results
   
These metrics provide insight into the model's ability to correctly identify customers likely to churn.

Key Insights

Exploratory analysis and modeling revealed that several factors influence churn behavior, including:

1. Customer tenure
2. Monthly charges
3. Contract type
4. Payment method
5. Internet service features
   
Understanding these relationships allows businesses to design targeted customer retention strategies.

Technologies Used

1. Python
2. Pandas
3. NumPy
4. Matplotlib
5. Seaborn
6. Scikit-learn
7. Streamlit
8. Jupyter Notebook / Google Colab

### Repository Structure

Customer-Churn-Prediction

│

├── DATASET

├── PROJECT

├── README.md

