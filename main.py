import pandas as pd
df = pd.read_csv("data/Fitbit_dataset.csv")
df.drop(columns=["Unnamed: 0"], inplace=True)
print(df.head())
print("\nShape of the dataset:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nDataset information:")
print(df.info())

print("\nStatistical summary:")
print(df.describe())

print("\n Missing values")
print(df.isnull().sum())

print("\nDuplicated rows:")
print(df.duplicated().sum())

#Distribution of Calories Burned (Target Variable)
import matplotlib.pyplot as plt
plt.figure(figsize = (8,5))
plt.hist(df["Calories_Burned (kcal)"], bins = 20, color = 'skyblue', edgecolor = 'black')
plt.title("Distribution of Calories Burned")
plt.xlabel("Calories_Burned (kcal)")
plt.ylabel("Number of Workout Sessions")
plt.show()

#Box Plot of Calories Burned
plt.figure(figsize =(8,5))
plt.boxplot(df["Calories_Burned (kcal)"])
plt.title("Box Plot of Calories Burned")
plt.ylabel("Calories_Burned (kcal)")
plt.show()

# Calories Burned by Workout Type
import seaborn as sns
plt.figure(figsize = (12,6))
sns.boxplot(x="Workout_Type", y= "Calories_Burned (kcal)", data = df)
plt.title("Calories Burned by Workout Type")
plt.xlabel("Workout Type")
plt.ylabel("Calories_Burned (kcal)")
plt.show()

# Correlation Heatmap
numerical_df = df.select_dtypes(include = ["number"])
correlation_matrix = numerical_df.corr()
plt.figure(figsize = (10,8))
sns.heatmap(correlation_matrix, annot = True, cmap = "coolwarm", fmt = ".2f")
plt.title("Correlation Heatmap")
plt.show()

# Session Duration vs Calories Burned
plt.figure(figsize = (10,6))
sns.scatterplot(x = "Session_Duration (hours)", y = "Calories_Burned (kcal)", data = df, )
plt.title("Session Duration vs Calories Burned")
plt.xlabel("Session Duration (hours)")
plt.ylabel("Calories_Burned (kcal)")
plt.show()

# Workout Type Distribution
plt.figure(figsize = (8,5))
sns.countplot(x = "Workout_Type", data = df)
plt.title("Workout Type Distribution")
plt.xlabel("Workout Type")
plt.ylabel("Number of Workout Sessions")
plt.show()

#Data Preprocessing
#Renaming target column
df.rename(columns = {"Calories_Burned (kcal)":"Calories_Burned"}, inplace = True)
print("\nRenamed target column:")
print(df.columns)

# Separate features and target
x = df.drop(columns = ["Calories_Burned"])
y = df["Calories_Burned"]
print("Features shape:", x.shape)
print("Target shape:", y.shape)

# Select numerical columns from X
numerical_columns = x.select_dtypes(include = ["number"]).columns
print("Numerical columns in features:", numerical_columns)

# Apply IQR capping
for column in numerical_columns:
    Q1 = x[column].quantile(0.25)
    Q3 = x[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    x[column] = x[column].clip(lower=lower_bound, upper=upper_bound)
print(x.dtypes)
# One-Hot Encoding
x = pd.get_dummies(x, columns = ["Gender","Workout_Type"], drop_first = True)
print("Features shape after one-hot encoding:", x.shape)

#Train-Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)


#Scaling numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print("Training Features:", x_train.shape)
print("Testing Features :", x_test.shape)

print("Training Target :", y_train.shape)
print("Testing Target  :", y_test.shape)

#Model Building
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(y_pred[:10])

#Model Evaluation
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score)
linear_mae = mean_absolute_error(y_test,y_pred)
linear_mse = mean_squared_error(y_test,y_pred)
linear_rmse = mean_squared_error(y_test,y_pred) **0.5
linear_r2 = r2_score(y_test, y_pred)
print("Linear Regression Results")
print("\nMAE:", linear_mae)
print("RMSE:", linear_rmse)
print("R2 Score:", linear_r2)

#Ridge Regression
from sklearn.linear_model import Ridge 
ridge_model = Ridge(alpha = 1.0)
ridge_model.fit(x_train, y_train)
ridge_pred = ridge_model.predict(x_test)

#Evaluation of Ridge Regression
ridge_mae = mean_absolute_error(y_test, ridge_pred)
ridge_rmse = mean_squared_error(y_test, ridge_pred) **0.5
ridge_r2 = r2_score(y_test,ridge_pred)
print("Ridge Regression Results")
print("MAE :", ridge_mae)
print("RMSE:", ridge_rmse)
print("R² Score:", ridge_r2)

#Lasso Regression
from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha = 0.1)
lasso_model.fit(x_train, y_train)

lasso_pred = lasso_model.predict(x_test)
lasso_mae = mean_absolute_error(y_test, lasso_pred)
lasso_rmse = mean_squared_error(y_test, lasso_pred)**0.5
lasso_r2 = r2_score(y_test, lasso_pred)
print("Lasso Regression Results")
print("MAE :", lasso_mae)
print("RMSE:", lasso_rmse)
print("R² Score:", lasso_r2)

#KNN Regressor
from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors = 5)
knn_model.fit(x_train, y_train)
knn_pred =knn_model.predict(x_test)

knn_mae = mean_absolute_error(y_test, knn_pred)
knn_rmse = mean_squared_error(y_test, knn_pred)** 0.5
knn_r2 = r2_score(y_test, knn_pred)
print("KNN Regression Results")
print("MAE :", knn_mae)
print("RMSE:", knn_rmse)
print("R² Score:", knn_r2)

#XGBoost Regressor
from xgboost import XGBRegressor
xgb_model = XGBRegressor(
    n_estimators=100,
    random_state=42
)
xgb_model.fit(x_train, y_train)
xgb_pred = xgb_model.predict(x_test)

xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = mean_squared_error(y_test, xgb_pred) ** 0.5
xgb_r2 = r2_score(y_test, xgb_pred)

print("\nXGBoost Regression Results")
print("--------------------------")
print("MAE :", xgb_mae)
print("RMSE:", xgb_rmse)
print("R² Score:", xgb_r2)

#Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor(random_state = 42)
dt_model.fit(x_train, y_train)
dt_pred = dt_model.predict(x_test)

dt_mae = mean_absolute_error(y_test, dt_pred)
dt_rmse = mean_squared_error(y_test, dt_pred) ** 0.5
dt_r2 = r2_score(y_test, dt_pred)

print("\nDecision Tree Regression Results")
print("--------------------------------")
print("MAE :", dt_mae)
print("RMSE:", dt_rmse)
print("R² Score:", dt_r2)

#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf_model.fit(x_train, y_train)
rf_pred = rf_model.predict(x_test)  

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = mean_squared_error(y_test, rf_pred) ** 0.5
rf_r2 = r2_score(y_test, rf_pred)
print("\nRandom Forest Regression Results")
print("--------------------------------")
print("MAE :", rf_mae)
print("RMSE:", rf_rmse)
print("R² Score:", rf_r2)

#Support Vector Regressor
from sklearn.svm import SVR
svr_model = SVR(kernel = 'rbf')
svr_model.fit(x_train, y_train)
svr_pred = svr_model.predict(x_test)

svr_mae = mean_absolute_error(y_test, svr_pred)
svr_rmse = mean_squared_error(y_test, svr_pred) ** 0.5
svr_r2 = r2_score(y_test, svr_pred)
print("\nSupport Vector Regression Results")
print("--------------------------------")
print("MAE :", svr_mae)
print("RMSE:", svr_rmse)
print("R² Score:", svr_r2)

import pandas as pd

results = pd.DataFrame({
    "Model":[
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression",
        "KNN Regressor",
        "XGBoost Regressor",
        "Decision Tree",
        "Random Forest",
        "Support Vector Regression"
    ],

    "MAE":[
        linear_mae,
        ridge_mae,
        lasso_mae,
        knn_mae,
        xgb_mae,
        dt_mae,
        rf_mae,
        svr_mae
    ],

    "RMSE":[
        linear_rmse,
        ridge_rmse,
        lasso_rmse,
        knn_rmse,
        xgb_rmse,
        dt_rmse,
        rf_rmse,
        svr_rmse
    ],

    "R2 Score":[
        linear_r2,
        ridge_r2,
        lasso_r2,
        knn_r2,
        xgb_r2,
        dt_r2,
        rf_r2,
        svr_r2
    ]}
)

print(results)
results = results.sort_values(
    by="R2 Score",
    ascending=False
)

print(results)

results.to_csv(
    "model_comparison.csv",
    index=False
)