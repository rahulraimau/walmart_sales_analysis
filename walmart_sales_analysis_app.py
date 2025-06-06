#!/usr/bin/env python
# coding: utf-8

# # 🛍️ Walmart Sales Analysis – Advanced Models & Visualizations

# In[ ]:

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r"C:\Users\DELL\Downloads\Walmart.csv")

st.write("Shape:", df.shape)
st.write("Columns:\n", df.columns)

st.write("\nMissing Values:\n", df.isnull().sum())
st.write("\nData Types:\n", df.dtypes)
df.head()


# In[ ]:


df.select_dtypes(include=['float64', 'int64']).hist(figsize=(15, 10), bins=20)
plt.suptitle("Distribution of Numerical Columns")
plt.tight_layout()
plt.show()

for col in df.select_dtypes(include='object').columns:
    plt.figure(figsize=(10, 4))
    sns.countplot(data=df, x=col)
    plt.title(f'Count of {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

df['Date'] = pd.to_datetime(df['Date'],dayfirst='true')
df.groupby('Date')['Weekly_Sales'].sum().plot(figsize=(14, 6), title="Total Weekly Sales Over Time")
plt.ylabel("Weekly Sales")
plt.xlabel("Date")
plt.show()


# In[ ]:


df.fillna(method='ffill', inplace=True)
df['Month'] = df['Date'].dt.month
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop('Weekly_Sales', axis=1)
y = df_encoded['Weekly_Sales']


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

datetime_cols = X.select_dtypes(include=['datetime64']).columns
X_numeric = X.drop(columns=datetime_cols)




X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# VIF
vif_df = pd.DataFrame()
vif_df['features'] = X_numeric.columns
vif_df['VIF'] = [variance_inflation_factor(X_numeric.values, i) for i in range(X_numeric.shape[1])]
st.write(vif_df.sort_values(by="VIF", ascending=False).head())

# RFE
rfe_model = LinearRegression()
rfe = RFE(rfe_model, n_features_to_select=10)
rfe.fit(X_train_scaled, y_train)
st.write("RFE selected features:\n", X_numeric.columns[rfe.support_])

# PCA
n_features = X_train_scaled.shape[1]
pca = PCA(n_components=min(10, n_features))
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
st.write("Explained Variance Ratio:\n", pca.explained_variance_ratio_)


# In[ ]:


from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_squared_error
import numpy as np



def evaluate_model(name, y_true, y_pred):
    st.write(f"📘 {name}")
    st.write("R2 Score:", round(r2_score(y_true, y_pred), 4))
    st.write("MAE:", round(mean_absolute_error(y_true, y_pred), 4))
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    st.write("RMSE:", round(rmse, 4))
    st.write("-" * 40)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
evaluate_model("Linear", y_test, lr.predict(X_test_scaled))

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
evaluate_model("Ridge", y_test, ridge.predict(X_test_scaled))

lasso = Lasso(alpha=0.01)
lasso.fit(X_train_scaled, y_train)
evaluate_model("Lasso", y_test, lasso.predict(X_test_scaled))

elastic = ElasticNet(alpha=0.01, l1_ratio=0.5)
elastic.fit(X_train_scaled, y_train)
evaluate_model("ElasticNet", y_test, elastic.predict(X_test_scaled))

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train_scaled)
X_poly_test = poly.transform(X_test_scaled)
poly_model = LinearRegression()
poly_model.fit(X_poly, y_train)
evaluate_model("Polynomial", y_test, poly_model.predict(X_poly_test))

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
evaluate_model("Random Forest", y_test, rf.predict(X_test_scaled))


# In[ ]:


from sklearn.model_selection import GridSearchCV

ridge_params = {'alpha': [0.01, 0.1, 1.0, 10]}
ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=5)
ridge_grid.fit(X_train_scaled, y_train)
st.write("Best Ridge Alpha:", ridge_grid.best_params_)
evaluate_model("Tuned Ridge", y_test, ridge_grid.predict(X_test_scaled))

lasso_params = {'alpha': [0.001, 0.01, 0.1, 1]}
lasso_grid = GridSearchCV(Lasso(), lasso_params, cv=5)
lasso_grid.fit(X_train_scaled, y_train)

st.write("Best Lasso Alpha:", lasso_grid.best_params_)
evaluate_model("Tuned Lasso", y_test, lasso_grid.predict(X_test_scaled))

rf_params = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=3)
rf_grid.fit(X_train_scaled, y_train)
st.write("Best RF Params:", rf_grid.best_params_)
evaluate_model("Tuned Random Forest", y_test, rf_grid.predict(X_test_scaled))


# In[ ]:


plt.figure(figsize=(10,6))
plt.scatter(y_test, lr.predict(X_test_scaled), label='Linear', alpha=0.6)
plt.scatter(y_test, rf.predict(X_test_scaled), label='Random Forest', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.title("Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.legend()
plt.show()

