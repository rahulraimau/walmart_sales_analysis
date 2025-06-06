
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

warnings.filterwarnings('ignore')

st.title("🛍️ Walmart Sales Analysis Dashboard")

uploaded_file = st.file_uploader("Upload Walmart CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File Uploaded Successfully!")

    st.subheader("📊 Dataset Overview")
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))
    st.write("Missing Values:", df.isnull().sum())
    st.write("Data Types:", df.dtypes)
    st.dataframe(df.head())

    st.subheader("📈 Numerical Distribution")
    fig, ax = plt.subplots(figsize=(15, 10))
    df.select_dtypes(include=['float64', 'int64']).hist(ax=ax, bins=20)
    st.pyplot(fig)

    st.subheader("📊 Categorical Distributions")
    for col in df.select_dtypes(include='object').columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.countplot(data=df, x=col, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.subheader("🔗 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Month'] = df['Date'].dt.month

    st.subheader("📅 Sales Over Time")
    fig, ax = plt.subplots(figsize=(14, 6))
    df.groupby('Date')['Weekly_Sales'].sum().plot(ax=ax)
    plt.title("Total Weekly Sales Over Time")
    plt.xlabel("Date")
    plt.ylabel("Weekly Sales")
    st.pyplot(fig)

    df.fillna(method='ffill', inplace=True)
    df_encoded = pd.get_dummies(df, drop_first=True)

    X = df_encoded.drop('Weekly_Sales', axis=1)
    y = df_encoded['Weekly_Sales']
    datetime_cols = X.select_dtypes(include=['datetime64']).columns
    X_numeric = X.drop(columns=datetime_cols)

    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.subheader("📌 Feature Engineering")

    vif_df = pd.DataFrame()
    vif_df["features"] = X_numeric.columns
    vif_df["VIF"] = [variance_inflation_factor(X_numeric.values, i) for i in range(X_numeric.shape[1])]
    st.write("Top VIF Features:")
    st.dataframe(vif_df.sort_values(by="VIF", ascending=False).head())

    rfe = RFE(LinearRegression(), n_features_to_select=10)
    rfe.fit(X_train_scaled, y_train)
    st.write("✅ RFE Selected Features:")
    st.write(list(X_numeric.columns[rfe.support_]))

    pca = PCA(n_components=min(10, X_train_scaled.shape[1]))
    X_train_pca = pca.fit_transform(X_train_scaled)
    st.write("🎯 PCA Explained Variance Ratio:")
    st.bar_chart(pca.explained_variance_ratio_)

    st.subheader("📈 Model Evaluation")

    def evaluate_model(name, y_true, y_pred):
        st.markdown(f"### 📘 {name}")
        st.write("R² Score:", round(r2_score(y_true, y_pred), 4))
        st.write("MAE:", round(mean_absolute_error(y_true, y_pred), 4))
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        st.write("RMSE:", round(rmse, 4))
        st.markdown("---")

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        evaluate_model(name, y_test, y_pred)

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X_train_scaled)
    X_poly_test = poly.transform(X_test_scaled)
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y_train)
    evaluate_model("Polynomial Regression", y_test, poly_model.predict(X_poly_test))

    st.subheader("🛠️ Model Tuning")

    ridge_grid = GridSearchCV(Ridge(), {'alpha': [0.01, 0.1, 1.0, 10]}, cv=5)
    ridge_grid.fit(X_train_scaled, y_train)
    st.write("Best Ridge Params:", ridge_grid.best_params_)
    evaluate_model("Tuned Ridge", y_test, ridge_grid.predict(X_test_scaled))

    lasso_grid = GridSearchCV(Lasso(), {'alpha': [0.001, 0.01, 0.1, 1]}, cv=5)
    lasso_grid.fit(X_train_scaled, y_train)
    st.write("Best Lasso Params:", lasso_grid.best_params_)
    evaluate_model("Tuned Lasso", y_test, lasso_grid.predict(X_test_scaled))

    rf_grid = GridSearchCV(RandomForestRegressor(random_state=42),
                           {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]},
                           cv=3)
    rf_grid.fit(X_train_scaled, y_train)
    st.write("Best RF Params:", rf_grid.best_params_)
    evaluate_model("Tuned Random Forest", y_test, rf_grid.predict(X_test_scaled))

    st.subheader("📉 Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(y_test, models["Linear"].predict(X_test_scaled), label='Linear', alpha=0.6)
    ax.scatter(y_test, models["Random Forest"].predict(X_test_scaled), label='Random Forest', alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    ax.set_title("Actual vs Predicted")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend()
    st.pyplot(fig)
