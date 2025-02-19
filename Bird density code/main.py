import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, DMatrix, train as xgb_train
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

# Load the data
file_path = r'C:\Users\Srikar Kolluru\Downloads\Washington_DC_Airport_With_Non_Incidents.csv'
data = pd.read_csv(file_path)

# Encode categorical variables with meaningful mappings
time_of_day_mapping = {
    'Day': 0, 'Night': 1, 'Dawn': 2, 'Dusk': 3, 'Unknown': 4
}

runway_mapping = {runway: idx for idx, runway in enumerate(data['RUNWAY'].unique())}

phase_of_flight_mapping = {
    'Approach': 0, 'Climb': 1, 'Cruise': 2, 'Descent': 3,
    'Landing': 4, 'Takeoff': 5, 'Taxi': 6, 'Unknown': 7
}

# Apply the mappings
data['TIME_OF_DAY_ENC'] = data['TIME_OF_DAY'].map(time_of_day_mapping)
data['RUNWAY_ENC'] = data['RUNWAY'].map(runway_mapping)
data['PHASE_OF_FLIGHT_ENC'] = data['PHASE_OF_FLIGHT'].map(phase_of_flight_mapping)

# Handle unmapped categories
data['TIME_OF_DAY_ENC'].fillna(-1, inplace=True)
data['RUNWAY_ENC'].fillna(-1, inplace=True)
data['PHASE_OF_FLIGHT_ENC'].fillna(-1, inplace=True)

# Drop unnecessary columns for modeling
columns_to_drop = [
    'INDEX_NR', 'INCIDENT_DATE', 'TIME', 'AIRPORT_ID', 'AIRPORT',
    'TIME_OF_DAY', 'RUNWAY', 'PHASE_OF_FLIGHT','LATITUDE','LONGITUDE'
]
processed_data = data.drop(columns=columns_to_drop)

# Feature Engineering
processed_data['MONTH_PHASE_INTERACTION'] = (
    processed_data['INCIDENT_MONTH'] * processed_data['PHASE_OF_FLIGHT_ENC']
)
processed_data['RUNWAY_PHASE_INTERACTION'] = (
    processed_data['RUNWAY_ENC'] * processed_data['PHASE_OF_FLIGHT_ENC']
)

# Normalize numerical features
scaler = StandardScaler()
numerical_features = [
    'INCIDENT_MONTH', 'INCIDENT_YEAR',
    'MONTH_PHASE_INTERACTION', 'RUNWAY_PHASE_INTERACTION'
]
processed_data[numerical_features] = scaler.fit_transform(processed_data[numerical_features])

# Define features and target variable
X = processed_data.drop(columns=['IS_INCIDENT'])
y = processed_data['IS_INCIDENT']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# Feature Selection using RFE with Random Forest
rfe_model = RandomForestClassifier(random_state=42)
rfe = RFE(estimator=rfe_model, n_features_to_select=7)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

# Hyperparameter search space
param_distributions = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'scale_pos_weight': [1, 2, 5]
}

# RandomizedSearchCV for XGBoost
xgb_model = XGBClassifier(random_state=46, eval_metric='logloss')
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=20,
    scoring='accuracy',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Run RandomizedSearchCV
random_search.fit(X_train_rfe, y_train)

# Extract best parameters
best_params = random_search.best_params_

# Oversample minority class using SMOTE
smote = SMOTE(random_state=46)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_rfe, y_train)

# Convert data to DMatrix for XGBoost
dtrain = DMatrix(X_train_balanced, label=y_train_balanced)
dtest = DMatrix(X_test_rfe, label=y_test)

# Train final model with early stopping
final_params = {
    **best_params,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 49
}
evals = [(dtest, "eval")]
final_model = xgb_train(
    params=final_params,
    dtrain=dtrain,
    num_boost_round=500,
    early_stopping_rounds=10,
    evals=evals,
    verbose_eval=True
)

# Make predictions
y_prob = final_model.predict(dtest)
y_pred = (y_prob > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# Output results
print("Best Parameters:", best_params)
print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)
print("ROC-AUC Score:", roc_auc)

# Future input features for prediction
future_input = {
    'INCIDENT_DAY': [11],  # Day
    'INCIDENT_MONTH': [12],  # December
    'INCIDENT_YEAR': [2023],  # Year
    'TIME_OF_DAY_ENC': [1],  # Night
    'RUNWAY_ENC': [5],  # Runway ID
    'PHASE_OF_FLIGHT_ENC': [4],  # Landing
}

# Create a DataFrame for the input
future_df = pd.DataFrame(future_input)

# Feature Engineering: Add interaction terms
future_df['MONTH_PHASE_INTERACTION'] = (
    future_df['INCIDENT_MONTH'] * future_df['PHASE_OF_FLIGHT_ENC']
)
future_df['RUNWAY_PHASE_INTERACTION'] = (
    future_df['RUNWAY_ENC'] * future_df['PHASE_OF_FLIGHT_ENC']
)

# Include all features used in training
all_features = [
    'INCIDENT_MONTH', 'INCIDENT_YEAR', 'TIME_OF_DAY_ENC',
    'RUNWAY_ENC', 'PHASE_OF_FLIGHT_ENC',
    'MONTH_PHASE_INTERACTION', 'RUNWAY_PHASE_INTERACTION'
]

# Ensure future_df has all the necessary features
future_df = future_df[all_features]

# Normalize numerical features using the trained scaler
future_df[numerical_features] = scaler.transform(future_df[numerical_features])

# Ensure input matches the order of features used during training
future_input_processed = future_df[X_train.columns]

# Apply RFE on the processed DataFrame (with column names intact)
future_input_rfe = rfe.transform(future_input_processed)

# Convert to DMatrix for XGBoost prediction
future_dmatrix = DMatrix(future_input_rfe)

# Make predictions
future_prediction_prob = final_model.predict(future_dmatrix)

# Output prediction probabilities
incident_probability = future_prediction_prob[0] * 100  # Convert to percentage
non_incident_probability = (1 - future_prediction_prob[0]) * 100  # Convert to percentage

# Display the results
print(f"Probability of INCIDENT: {incident_probability:.2f}%")
print(f"Probability of NON-INCIDENT: {non_incident_probability:.2f}%")
