import streamlit as st
import pandas as pd
import numpy as np
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Federated Attack Simulation", layout="wide")
st.title("🚨 Federated Attack Simulation with Enhanced SHAP Visualizations")

uploaded_file = st.file_uploader("Upload dataset (Excel/CSV)", type=["xlsx", "csv"])

def st_shap(plot, height=300):
    """Display SHAP plot in Streamlit"""
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

if uploaded_file is not None:
    # Read dataset
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    # Normalize column names
    df.columns = [col.strip().lower() for col in df.columns]

    # Required features
    selected_features = ["length", "protocol", "info", "source", "destination", "time"]

    missing_features = [f for f in selected_features if f not in df.columns]
    if missing_features:
        st.error(f"⚠️ Missing required columns: {missing_features}")
    else:
        # Select features
        X = df[selected_features].copy()
        # Encode categorical features
        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = X[col].astype("category").cat.codes

        # Label
        if "label" not in df.columns:
            st.warning("⚠️ No 'label' column found. Creating synthetic labels for demo...")
            y = np.random.randint(0, 2, size=len(X))
        else:
            y = df["label"]

        st.write("### Raw Dataset Preview", df.head())

        # Federated Training
        st.subheader("⚡ Federated Training (LightGBM Clients)")
        num_clients = 3
        client_models = []
        client_explainers = []

        partition_size = len(X) // num_clients

        for i in range(num_clients):
            start = i * partition_size
            end = (i + 1) * partition_size if i < num_clients - 1 else len(X)
            X_client = X.iloc[start:end]
            y_client = y[start:end]
            model = lgb.LGBMClassifier(n_estimators=50, random_state=42)
            model.fit(X_client, y_client)
            client_models.append(model)
            explainer = shap.TreeExplainer(model)
            client_explainers.append(explainer)

        # Simulate multiple attacks
        num_simulations = st.slider("Number of attack simulations", 1, 5, 3)
        st.subheader("🔹 Random Rows for Simulation:")
        random_rows = X.sample(num_simulations, random_state=42)
        st.write(random_rows)

        # Ensemble prediction
        ensemble_probas = np.mean([model.predict_proba(random_rows)[:, 1] for model in client_models], axis=0)
        for idx, proba in zip(random_rows.index, ensemble_probas):
            st.write(f"Row {idx} predicted attack probability: {proba:.4f}")
            if proba > 0.5:
                st.error("🚨 Attack Detected!")
            else:
                st.success("✅ Normal Traffic")

        # SHAP Explainability
        st.subheader("📈 SHAP Explainability for Random Rows")
        shap_values_list = []
        base_values_list = []

        for explainer in client_explainers:
            sv = explainer.shap_values(random_rows)
            base_val = explainer.expected_value
            # For classification, take class 1
            if isinstance(sv, list):
                sv = sv[1]
                base_val = base_val[1]
            shap_values_list.append(sv)
            base_values_list.append(base_val)

        # Average SHAP values across clients
        shap_values_avg = np.mean(np.array(shap_values_list), axis=0)
        base_value_avg = np.mean(base_values_list)

        # Force plots for each row
        st.write("🔎 SHAP Force Plots for each simulated attack:")
        for i in range(len(random_rows)):
            st.write(f"**Row {random_rows.index[i]}**")
            force_plot = shap.force_plot(
                base_value_avg,
                shap_values_avg[i],
                random_rows.iloc[i],
                matplotlib=False
            )
            st_shap(force_plot, height=300)

        # SHAP bar plots for each row
        st.write("📊 SHAP Bar Plots for each simulated attack:")
        for i in range(len(random_rows)):
            st.write(f"**Row {random_rows.index[i]}**")
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.bar_plot(shap_values_avg[i], feature_names=selected_features, max_display=10)
            st.pyplot(fig)

        # SHAP summary bar plot (overall feature importance)
        st.write("📊 SHAP Summary Bar Plot (Overall Feature Importance)")
        fig_summary, ax_summary = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_values_avg, random_rows, plot_type="bar", max_display=10, show=False)
        st.pyplot(fig_summary)

        # SHAP beeswarm plot
        st.write("🐝 SHAP Beeswarm Plot (Impact of Features Across All Simulated Attacks)")
        fig_beeswarm, ax_beeswarm = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_values_avg, random_rows, plot_type="dot", max_display=10, show=False)
        st.pyplot(fig_beeswarm)

        # Save outputs
        os.makedirs("outputs", exist_ok=True)
        random_rows.to_csv("outputs/random_rows.csv", index=False)
        fig_summary.savefig("outputs/shap_summary_bar.png")
        fig_beeswarm.savefig("outputs/shap_beeswarm.png")
        st.success("✅ All outputs saved to ./outputs/ (CSV + SHAP images)")

else:
    st.info("📂 Please upload a dataset to begin.")
