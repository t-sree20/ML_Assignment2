import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

# --- Page Config ---
st.set_page_config(
    page_title="ML Assignment 2",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced Custom CSS ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Main Background & Text */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        color: #e8eaf6;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e2e 0%, #262637 100%);
        border-right: 2px solid #3d3d5c;
        box-shadow: 4px 0 15px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stSidebar"] .element-container {
        padding: 0.5rem 0;
    }
    
    /* Headers with Gradient */
    h1 {
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1.5rem !important;
        letter-spacing: -0.5px;
    }
    
    h2 {
        color: #a5b4fc !important;
        font-weight: 600 !important;
        font-size: 1.8rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        border-bottom: 2px solid #4c1d95;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        color: #c4b5fd !important;
        font-weight: 600 !important;
        font-size: 1.3rem !important;
        margin-top: 1.5rem !important;
    }
    
    /* Metric Containers */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #2d2d44 0%, #1f1f2e 100%);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #4a4a6a;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 24px rgba(102, 126, 234, 0.3);
    }
    
    [data-testid="metric-container"] label {
        color: #a5b4fc !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.7rem 1.8rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: #2d2d44;
        border-radius: 10px;
        padding: 1.5rem;
        border: 2px dashed #4a4a6a;
        transition: border-color 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
    }
    
    /* Select Box */
    .stSelectbox > div > div {
        background-color: #2d2d44;
        border-radius: 8px;
        border: 1px solid #4a4a6a;
    }
    
    /* Dataframes */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, #4c1d95 0%, #5b21b6 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 12px !important;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    
    .dataframe tbody tr:nth-child(odd) {
        background-color: #1f1f2e;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: #2d2d44;
    }
    
    .dataframe tbody tr:hover {
        background-color: #3d3d5c !important;
    }
    
    /* Info/Warning/Error Boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid;
        padding: 1rem 1.5rem;
        background: #2d2d44;
    }
    
    [data-baseweb="notification"] {
        border-radius: 10px;
    }
    
    /* Radio Buttons */
    .stRadio > label {
        color: #a5b4fc !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem;
    }
    
    .stRadio > div {
        background: #2d2d44;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    /* Dividers */
    hr {
        border-color: #4a4a6a;
        margin: 2rem 0;
    }
    
    /* Cards/Containers */
    .card {
        background: linear-gradient(135deg, #2d2d44 0%, #1f1f2e 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #4a4a6a;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #2d2d44;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2d2d44;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# --- Load Models & Utilities ---
@st.cache_resource
def load_resources():
    models = {}
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "kNN": "knn.pkl",
        "Naive Bayes": "naive_bayes.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl"
    }
    
    for name, filename in model_files.items():
        path = os.path.join("model", filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    
    # Load Scaler
    scaler = None
    if os.path.exists("model/scaler.pkl"):
        with open("model/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
            
    # Load Pre-calculated Metrics
    metrics_df = None
    if os.path.exists("model_metrics.csv"):
        metrics_df = pd.read_csv("model_metrics.csv")
        
    return models, scaler, metrics_df

models, scaler, metrics_df = load_resources()

# --- Sidebar ---
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("#### Navigation")
    app_mode = st.radio(
        "",
        ["Prediction & Analysis", "Model Comparison"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")

    # Model Stats
    if models:
        st.markdown("#### Loaded Models")
        st.success(f"{len(models)} models ready")
        with st.expander("View Models"):
            for model_name in models.keys():
                st.markdown(f"• {model_name}")
    
    st.markdown("---")
    st.markdown("""
    """, unsafe_allow_html=True)

# --- Main Content ---

if "Prediction & Analysis" in app_mode:
    
    # Configuration Section
    config_col1, config_col2 = st.columns([1, 2])
    
    with config_col1:
        st.markdown("### Model Selection")
        selected_model_name = st.selectbox(
            "Choose a machine learning model",
            list(models.keys()),
            help="Select the model you want to use for predictions"
        )
        selected_model = models[selected_model_name]
        
        # Show model info from metrics
        if metrics_df is not None and selected_model_name in metrics_df['Model'].values:
            model_metrics = metrics_df[metrics_df['Model'] == selected_model_name].iloc[0]
            
            st.markdown("#### Model Performance")
            perf_col1, perf_col2 = st.columns(2)
            with perf_col1:
                st.metric("Accuracy", f"{model_metrics['Accuracy']:.2%}")
                st.metric("Precision", f"{model_metrics['Precision']:.2%}")
            with perf_col2:
                st.metric("Recall", f"{model_metrics['Recall']:.2%}")
                st.metric("F1 Score", f"{model_metrics['F1']:.2%}")
    
    with config_col2:
        st.markdown("### Data Upload")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a Dataset CSV file with the same features as the training dataset"
        )
        
        #if not uploaded_file:
            #st.caption("Your CSV should contain 30 numerical features (radius, texture, perimeter, area, etc.)")


    st.markdown("---")
    
    # Results Section
    if uploaded_file is not None:
        try:
            # Read Data
            input_df = pd.read_csv(uploaded_file)
            
            # Data Preview
            with st.expander("View Uploaded Data", expanded=False):
                st.dataframe(input_df.head(10), use_container_width=True)
                st.caption(f"Showing first 10 of {len(input_df)} rows")
            
            # Check for ID columns to drop
            cols_to_drop = ['id', 'Unnamed: 32', 'diagnosis']
            cols_present_to_drop = [c for c in cols_to_drop if c in input_df.columns]
            
            features_df = input_df.drop(columns=cols_present_to_drop)
            
            # Scale Features
            if scaler:
                X_scaled = scaler.transform(features_df)
            else:
                X_scaled = features_df
            
            # Make Predictions
            predictions = selected_model.predict(X_scaled)
            pred_labels = ["Malignant" if p == 1 else "Benign" for p in predictions]
            
            # Add Probability if supported
            if hasattr(selected_model, "predict_proba"):
                probs = selected_model.predict_proba(X_scaled)[:, 1]
            else:
                probs = [0] * len(predictions)
            
            # === RESULTS DISPLAY ===
            st.markdown("## Prediction Results")
            
            # Summary Metrics
            malignant_count = np.sum(predictions)
            benign_count = len(predictions) - malignant_count
            malignant_percentage = (malignant_count / len(predictions)) * 100
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric(
                    "Total Samples",
                    len(predictions),
                    help="Total number of samples analyzed"
                )
            
            with metric_col2:
                st.metric(
                    "Malignant Cases",
                    int(malignant_count),
                    delta=f"{malignant_percentage:.1f}%",
                    delta_color="inverse",
                    help="Number of malignant tumors detected"
                )
            
            with metric_col3:
                st.metric(
                    "Benign Cases",
                    int(benign_count),
                    delta=f"{100-malignant_percentage:.1f}%",
                    delta_color="normal",
                    help="Number of benign tumors detected"
                )
            
            with metric_col4:
                if hasattr(selected_model, "predict_proba"):
                    avg_confidence = np.max(selected_model.predict_proba(X_scaled), axis=1).mean()
                    st.metric(
                        "Avg Confidence",
                        f"{avg_confidence:.1%}",
                        help="Average prediction confidence"
                    )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Detailed Results Table
            st.markdown("### Detailed Predictions")
            
            result_df = input_df.copy()
            result_df["Prediction"] = pred_labels
            
            if hasattr(selected_model, "predict_proba"):
                result_df["Confidence"] = np.max(selected_model.predict_proba(X_scaled), axis=1)
                result_df["Malignancy Score"] = probs
            
            # Reorder columns
            cols_order = []
            if 'id' in result_df.columns:
                cols_order.append('id')
            if 'diagnosis' in result_df.columns:
                cols_order.append('diagnosis')
            cols_order.extend(['Prediction'])
            if 'Confidence' in result_df.columns:
                cols_order.extend(['Confidence', 'Malignancy Score'])
            
            # Add remaining columns
            remaining_cols = [col for col in result_df.columns if col not in cols_order]
            cols_order.extend(remaining_cols)
            
            result_df = result_df[cols_order]
            
            # Style the dataframe
            def highlight_predictions(row):
                if row['Prediction'] == 'Malignant':
                    return ['background-color: rgba(239, 68, 68, 0.2)'] * len(row)
                else:
                    return ['background-color: rgba(34, 197, 94, 0.2)'] * len(row)
            
            styled_df = result_df.style.apply(highlight_predictions, axis=1)
            
            if 'Confidence' in result_df.columns:
                styled_df = styled_df.format({
                    'Confidence': '{:.2%}',
                    'Malignancy Score': '{:.3f}'
                })
            
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Download Results
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name=f"predictions_{selected_model_name.replace(' ', '_')}.csv",
                mime="text/csv",
            )
            
            # === EVALUATION (If labels exist) ===
            if 'diagnosis' in input_df.columns:
                st.markdown("---")
                st.markdown("## Model Evaluation")
                
                st.success("Ground truth labels detected - calculating performance metrics...")
                
                try:
                    # Encode True Labels
                    y_true = input_df['diagnosis'].map({'M': 1, 'B': 0})
                    if y_true.isnull().any():
                        y_true = input_df['diagnosis']
                    
                    # Calculate Metrics
                    acc = accuracy_score(y_true, predictions)
                    precision = precision_score(y_true, predictions, zero_division=0)
                    recall = recall_score(y_true, predictions, zero_division=0)
                    f1 = f1_score(y_true, predictions, zero_division=0)
                    
                    # Display Metrics
                    st.markdown("### Performance Metrics")
                    
                    eval_col1, eval_col2, eval_col3, eval_col4 = st.columns(4)
                    
                    with eval_col1:
                        st.metric("Accuracy", f"{acc:.2%}")
                    with eval_col2:
                        st.metric("Precision", f"{precision:.2%}")
                    with eval_col3:
                        st.metric("Recall", f"{recall:.2%}")
                    with eval_col4:
                        st.metric("F1 Score", f"{f1:.2%}")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Confusion Matrix and Classification Report
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        st.markdown("### Confusion Matrix")
                        cm = confusion_matrix(y_true, predictions)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        fig.patch.set_facecolor('#1a1f3a')
                        ax.set_facecolor('#1a1f3a')
                        
                        sns.heatmap(
                            cm, 
                            annot=True, 
                            fmt='d', 
                            cmap='Blues',
                            ax=ax,
                            xticklabels=['Benign', 'Malignant'],
                            yticklabels=['Benign', 'Malignant'],
                            cbar_kws={'label': 'Count'},
                            annot_kws={'size': 14, 'weight': 'bold'}
                        )
                        
                        ax.set_ylabel('Actual', fontsize=12, color='white')
                        ax.set_xlabel('Predicted', fontsize=12, color='white')
                        ax.tick_params(colors='white')
                        
                        plt.setp(ax.get_xticklabels(), color='white')
                        plt.setp(ax.get_yticklabels(), color='white')
                        
                        cbar = ax.collections[0].colorbar
                        cbar.ax.tick_params(colors='white')
                        cbar.ax.yaxis.label.set_color('white')
                        
                        st.pyplot(fig)
                    
                    with viz_col2:
                        st.markdown("### Classification Report")
                        
                        report = classification_report(
                            y_true, 
                            predictions,
                            target_names=['Benign', 'Malignant'],
                            output_dict=True
                        )
                        
                        report_df = pd.DataFrame(report).transpose()
                        report_df = report_df.round(3)
                        
                        # Format for display
                        display_df = report_df[['precision', 'recall', 'f1-score', 'support']].copy()
                        display_df.columns = ['Precision', 'Recall', 'F1-Score', 'Support']
                        
                        st.dataframe(
                            display_df.style.format({
                                'Precision': '{:.3f}',
                                'Recall': '{:.3f}',
                                'F1-Score': '{:.3f}',
                                'Support': '{:.0f}'
                            }),
                            use_container_width=True
                        )
                    
                except Exception as e:
                    st.warning(f"Could not calculate evaluation metrics: {e}")
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("Please ensure your CSV file has the correct format and features.")
    
    else:
        # No file uploaded - just show model training performance
        if metrics_df is not None:
            st.markdown(f"## Training Performance - {selected_model_name}")
            
            model_perf = metrics_df[metrics_df['Model'] == selected_model_name]
            
            if not model_perf.empty:
                # Create a nice display of metrics
                display_metrics = model_perf.iloc[0].to_dict()
                
                # Determine number of columns based on available metrics
                available_metrics = []
                # Map display names to actual column names
                metric_map = {
                    'Accuracy': 'Accuracy',
                    'Precision': 'Precision', 
                    'Recall': 'Recall', 
                    'F1': 'F1', 
                    'ROC-AUC': 'AUC', 
                    'MCC': 'MCC'
                }
                
                for display_name, col_name in metric_map.items():
                    if col_name in display_metrics:
                        available_metrics.append((display_name, col_name))
                
                # Create columns for all metrics in one row
                num_cols = len(available_metrics)
                cols = st.columns(num_cols)
                
                # Display all metrics in a single row
                for idx, (display_name, col_name) in enumerate(available_metrics):
                    with cols[idx]:
                        value = display_metrics[col_name]
                        # Format based on metric type
                        if col_name in ['Accuracy', 'Precision', 'Recall', 'F1']:
                            formatted_value = f"{value:.2%}"
                        else:  # ROC-AUC (AUC), MCC
                            formatted_value = f"{value:.4f}"
                        
                        st.metric(
                            display_name,
                            formatted_value,
                            help=f"Training set {display_name} score"
                        )

elif "Model Comparison" in app_mode:
    st.markdown("<h1>Model Performance Benchmarking</h1>", unsafe_allow_html=True)

    st.markdown("---")
    
    if metrics_df is not None:
        # Overall Metrics Table
        st.markdown("## Comprehensive Metrics Comparison")
        
        # Style the dataframe
        def highlight_best(s):
            if s.name == 'Model':
                return [''] * len(s)
            is_max = s == s.max()
            return ['background-color: rgba(102, 126, 234, 0.3); font-weight: bold' if v else '' for v in is_max]
        
        styled_metrics = metrics_df.style.apply(highlight_best, axis=0).format({
            'Accuracy': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1': '{:.4f}',
            'AUC': '{:.4f}',
            'MCC': '{:.4f}'
        })
        
        st.dataframe(styled_metrics, use_container_width=True, height=300)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Visualizations
        st.markdown("## Performance Visualizations")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Key Metrics", "Detailed Analysis", "Rankings"])
        
        with tab1:
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.markdown("### Accuracy Comparison")
                fig, ax = plt.subplots(figsize=(10, 6))
                fig.patch.set_facecolor('#1a1f3a')
                ax.set_facecolor('#1a1f3a')
                
                colors = sns.color_palette("viridis", len(metrics_df))
                bars = ax.barh(metrics_df["Model"], metrics_df["Accuracy"], color=colors)
                
                ax.set_xlabel('Accuracy', fontsize=12, color='white', fontweight='bold')
                ax.set_ylabel('Model', fontsize=12, color='white', fontweight='bold')
                ax.set_xlim(metrics_df["Accuracy"].min() - 0.02, 1.0)
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='x', alpha=0.3, color='white')
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2, 
                           f'{width:.4f}', 
                           ha='left', va='center', color='white', 
                           fontweight='bold', fontsize=9, 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='#2d2d44', edgecolor='none'))
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with viz_col2:
                st.markdown("### F1 Score Comparison")
                fig, ax = plt.subplots(figsize=(10, 6))
                fig.patch.set_facecolor('#1a1f3a')
                ax.set_facecolor('#1a1f3a')
                
                colors = sns.color_palette("magma", len(metrics_df))
                bars = ax.barh(metrics_df["Model"], metrics_df["F1"], color=colors)
                
                ax.set_xlabel('F1 Score', fontsize=12, color='white', fontweight='bold')
                ax.set_ylabel('Model', fontsize=12, color='white', fontweight='bold')
                ax.set_xlim(metrics_df["F1"].min() - 0.02, 1.0)
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='x', alpha=0.3, color='white')
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2, 
                           f'{width:.4f}', 
                           ha='left', va='center', color='white', 
                           fontweight='bold', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='#2d2d44', edgecolor='none'))
                
                plt.tight_layout()
                st.pyplot(fig)
        
        with tab2:
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.markdown("### Precision vs Recall")
                fig, ax = plt.subplots(figsize=(10, 8))
                fig.patch.set_facecolor('#1a1f3a')
                ax.set_facecolor('#1a1f3a')
                
                scatter = ax.scatter(
                    metrics_df["Precision"], 
                    metrics_df["Recall"],
                    s=200,
                    c=range(len(metrics_df)),
                    cmap='plasma',
                    alpha=0.7,
                    edgecolors='white',
                    linewidth=2
                )
                
                # Add model labels
                for idx, row in metrics_df.iterrows():
                    ax.annotate(
                        row['Model'], 
                        (row['Precision'], row['Recall']),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=9,
                        color='white',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#2d2d44', alpha=0.8, edgecolor='none')
                    )
                
                ax.set_xlabel('Precision', fontsize=12, color='white', fontweight='bold')
                ax.set_ylabel('Recall', fontsize=12, color='white', fontweight='bold')
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(alpha=0.3, color='white')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with detail_col2:
                st.markdown("### ROC-AUC Scores")
                fig, ax = plt.subplots(figsize=(10, 8))
                fig.patch.set_facecolor('#1a1f3a')
                ax.set_facecolor('#1a1f3a')
                
                colors = sns.color_palette("coolwarm", len(metrics_df))
                bars = ax.barh(metrics_df["Model"], metrics_df["AUC"], color=colors)
                
                ax.set_xlabel('ROC-AUC Score', fontsize=12, color='white', fontweight='bold')
                ax.set_ylabel('Model', fontsize=12, color='white', fontweight='bold')
                ax.set_xlim(metrics_df["AUC"].min() - 0.02, 1.0)
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='x', alpha=0.3, color='white')
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2, 
                           f'{width:.4f}', 
                           ha='left', va='center', color='white', 
                           fontweight='bold', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='#2d2d44', edgecolor='none'))
                
                plt.tight_layout()
                st.pyplot(fig)
        
        with tab3:
            st.markdown("### Model Rankings")
            
            # Create ranking for each metric
            # Map display name to actual column name
            ranking_metrics_map = {
                'Accuracy': 'Accuracy', 
                'Precision': 'Precision', 
                'Recall': 'Recall', 
                'F1': 'F1', 
                'ROC-AUC': 'AUC', 
                'MCC': 'MCC'
            }
            
            rankings = {}
            for display_metric, col_metric in ranking_metrics_map.items():
                sorted_df = metrics_df.sort_values(col_metric, ascending=False)
                rankings[display_metric] = sorted_df[['Model', col_metric]].reset_index(drop=True)
                rankings[display_metric].index = rankings[display_metric].index + 1
                rankings[display_metric].index.name = 'Rank'
            
            # Display rankings in columns
            rank_col1, rank_col2, rank_col3 = st.columns(3)
            
            cols = [rank_col1, rank_col2, rank_col3]
            for idx, (display_metric, col_metric) in enumerate(ranking_metrics_map.items()):
                with cols[idx % 3]:
                    st.markdown(f"#### {display_metric}")
                    st.dataframe(
                        rankings[display_metric].style.format({col_metric: '{:.4f}'}),
                        use_container_width=True,
                        height=250
                    )
        
        # Best Model Summary
        st.markdown("---")
        st.markdown("## Best Performing Model")
        
        best_model_idx = metrics_df['F1'].idxmax()
        best_model = metrics_df.loc[best_model_idx]
        
        st.success(f"**{best_model['Model']}** achieves the highest F1 Score of **{best_model['F1']:.4f}**")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        
        with summary_col1:
            st.metric("Accuracy", f"{best_model['Accuracy']:.2%}")
        with summary_col2:
            st.metric("Precision", f"{best_model['Precision']:.2%}")
        with summary_col3:
            st.metric("Recall", f"{best_model['Recall']:.2%}")
        with summary_col4:
            st.metric("ROC-AUC", f"{best_model['AUC']:.4f}")
    
    
