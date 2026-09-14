import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Page configuration
st.set_page_config(
    page_title="Fitbit Calorie Prediction",
    page_icon="🔥",
    layout="wide"
)


# Load Fitbit dataset
df = pd.read_csv("data/Fitbit_dataset.csv")

df.drop(
    columns=["Unnamed: 0"],
    inplace=True,
    errors="ignore"
)

# Load supervised results
model_results = pd.read_csv(
    "output/model_comparison.csv"
)


# Load unsupervised results
cluster_summary = pd.read_csv(
    "output/cluster_summary.csv"
)

pca_data = pd.read_csv(
    "output/pca_data.csv"
)


# Sidebar menu
page = st.sidebar.radio(
    "Select a page",
    [
        "Project Overview",
        "Supervised Learning",
        "Unsupervised Learning"
    ]
)


# Main heading
st.title("🔥 Fitbit Calorie Prediction Dashboard")

st.write(
    """
    This dashboard presents the supervised and unsupervised
    machine-learning results from the Fitbit dataset.
    """
)

st.divider()

# Sidebar menu
page = st.sidebar.selectbox(
    "Select Analysis",
    [
        "Project Overview",
        "Supervised Learning",
        "Unsupervised Learning"
    ]
)

# Display selected page
if page == "Project Overview":

    st.header("📌 Project Overview")

    st.write(
        """
        The main objective of this project is to analyse Fitbit
        workout information and predict the number of calories
        burned during a workout.

        The project uses:

        - Supervised learning for calorie prediction
        - Unsupervised learning for grouping similar workout records
        """
    )

elif page == "Supervised Learning":

    st.header("🤖 Supervised Learning Results")

    st.write(
        """
        Multiple regression models were trained to predict calories
        burned. The models were evaluated using MAE, RMSE and R² score.
        """
    )

    st.divider()

    st.subheader("📊 R² Score Comparison")

    # Sort models by R² score
    r2_data = model_results.sort_values(
        by="R2 Score",
        ascending=False
    )

    # Create chart
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=r2_data,
        x="R2 Score",
        y="Model",
        hue="Model",
        palette="viridis",
        legend=False,
        ax=ax
    )
        # Zoom the x-axis to show small differences
    minimum_r2 = r2_data["R2 Score"].min()
    maximum_r2 = r2_data["R2 Score"].max()

    ax.set_xlim(
        minimum_r2 - 0.02,
        min(1.01, maximum_r2 + 0.02)
    )

    # Add the exact R² value beside every bar
    for bar in ax.patches:

        r2_value = bar.get_width()

        ax.text(
            r2_value + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{r2_value:.4f}",
            va="center",
            fontsize=10,
            fontweight="bold"
        )
         


    ax.set_title("R² Score of Supervised Models")
    ax.set_xlabel("R² Score")
    ax.set_ylabel("Model")

    # Display chart in Streamlit
    st.pyplot(fig)

    st.divider()

    st.subheader("📉 MAE Comparison")

    # Arrange models from lowest MAE to highest MAE
    mae_data = model_results.sort_values(
        by="MAE",
        ascending=True
    )

    # Create the graph
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=mae_data,
        x="MAE",
        y="Model",
        hue="Model",
        palette="magma",
        legend=False,
        ax=ax
    )

    # Display exact MAE value beside each bar
    for bar in ax.patches:

        mae_value = bar.get_width()

        ax.text(
            mae_value + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{mae_value:.2f}",
            va="center",
            fontsize=10,
            fontweight="bold"
        )

    ax.set_title("Mean Absolute Error of Supervised Models")
    ax.set_xlabel("MAE in Calories (Lower is Better)")
    ax.set_ylabel("Model")

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.divider()

    st.subheader("📉 RMSE Comparison")

    # Arrange models from lowest RMSE to highest RMSE
    rmse_data = model_results.sort_values(
        by="RMSE",
        ascending=True
    )

    # Create chart
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=rmse_data,
        x="RMSE",
        y="Model",
        hue="Model",
        palette="coolwarm",
        legend=False,
        ax=ax
    )

    # Display exact RMSE value
    for bar in ax.patches:

        rmse_value = bar.get_width()

        ax.text(
            rmse_value + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{rmse_value:.2f}",
            va="center",
            fontsize=10,
            fontweight="bold"
        )

    # Give space for values at the end of bars
    ax.set_xlim(
        0,
        rmse_data["RMSE"].max() * 1.15
    )

    ax.set_title("Root Mean Squared Error of Supervised Models")
    ax.set_xlabel("RMSE in Calories (Lower is Better)")
    ax.set_ylabel("Model")

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.divider()

    st.subheader("🏆 Best Model Summary")

    # Find the row with the highest R² score
    best_model_index = model_results["R2 Score"].idxmax()

    best_model = model_results.loc[
        best_model_index,
        "Model"
    ]

    best_mae = model_results.loc[
        best_model_index,
        "MAE"
    ]

    best_rmse = model_results.loc[
        best_model_index,
        "RMSE"
    ]

    best_r2 = model_results.loc[
        best_model_index,
        "R2 Score"
    ]

    # Display the best model across the full width
    st.success(f"🏆 Best Performing Model: {best_model}")

    # Create three columns for evaluation scores
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Lowest MAE",
            value=f"{best_mae:.2f} kcal"
        )

    with col2:
        st.metric(
            label="Lowest RMSE",
            value=f"{best_rmse:.2f} kcal"
        )

    with col3:
        st.metric(
            label="Highest R²",
            value=f"{best_r2:.4f}"
        )

        

elif page == "Unsupervised Learning":

    st.header("🔍 Unsupervised Learning Results")

    st.write(
        """
        K-Means clustering was used to group similar Fitbit workout
        records. PCA reduced the features to two components so that
        the clusters could be visualised on a two-dimensional graph.
        """
    )

    st.divider()

    st.subheader("📊 Clustering Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Clustering Algorithm",
            value="K-Means"
        )

    with col2:
        st.metric(
            label="Number of Clusters",
            value=pca_data["Cluster"].nunique()
        )

    with col3:
        st.metric(
            label="Silhouette Score",
            value="0.3806"
        )

    st.info(
        """
        A Silhouette Score of 0.3806 indicates moderate cluster
        separation. The workout groups have some meaningful
        differences, but they also overlap.
        """
    )

    st.divider()

    st.subheader("🎯 PCA Cluster Visualization")

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=pca_data,
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette="Set1",
        alpha=0.6,
        s=50,
        ax=ax
    )

    ax.set_title("K-Means Clusters Visualized Using PCA")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    ax.legend(
        title="Cluster"
    )

    ax.grid(
        linestyle="--",
        alpha=0.3
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.divider()

    st.subheader("📋 Cluster Summary")

    st.dataframe(
        cluster_summary,
        use_container_width=True
    )

    st.divider()

    st.subheader("🌈 Understanding the Workout Clusters")

    st.write(
        """
        K-Means identified three different workout patterns. Based on the
        average characteristics of each group, we can interpret them as follows:
        """
    )

    # Cluster 0
    st.error(
        """
        🔥 CLUSTER 0 — QUICK POWER WORKOUTS

        • Average age: Around 37 years  
        • Average workout duration: Around 40 minutes  
        • Average heart rate: Around 155 BPM  
        • Resting heart rate: Around 73 BPM  

        This group performs shorter workouts with a high average heart rate.
        It represents quick but intense workout sessions.
        """
    )

    # Cluster 1
    st.warning(
        """
        🚶 CLUSTER 1 — BALANCED WORKOUTS

        • Average age: Around 42 years  
        • Average workout duration: Around 49 minutes  
        • Average heart rate: Around 132 BPM  
        • Maximum heart rate: Around 177 BPM  

        This group performs moderate-duration workouts at a comparatively
        lower intensity. It represents a steady and balanced workout pattern.
        """
    )

    # Cluster 2
    st.success(
        """
        🏃 CLUSTER 2 — ACTIVE ENDURANCE WORKOUTS

        • Average age: Around 38 years  
        • Average workout duration: Around 65 minutes  
        • Average heart rate: Around 151 BPM  
        • Resting heart rate: Around 63 BPM  

        This group performs the longest workouts while maintaining a high
        average heart rate. The lower resting heart rate and body-fat percentage
        may indicate a more active and physically fit group.
        """
    )

    st.info(
        """
        💡 Important: K-Means only created the labels 0, 1 and 2.
        The descriptive names above were assigned after examining the
        average characteristics in the cluster summary.
        """
    )