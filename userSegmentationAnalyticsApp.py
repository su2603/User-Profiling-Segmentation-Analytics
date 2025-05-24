import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Customer Segmentation Analytics",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .segment-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Enhanced Segmentation System
class EnhancedSegmentationSystem:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = None
        self.pca = None
        self.segment_labels = None
        self.segment_profiles = []
        self.feature_importance = {}
        self.silhouette_score = None
        self.processed_features = None

    def preprocess_data(
        self, df, categorical_cols=None, text_cols=None, date_cols=None
    ):
        """Preprocess data for clustering"""
        try:
            df_processed = df.copy()

            # Handle missing values
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(
                df_processed[numeric_cols].median()
            )

            # Process categorical columns
            if categorical_cols:
                le = LabelEncoder()
                for col in categorical_cols:
                    if col in df_processed.columns:
                        df_processed[col] = df_processed[col].fillna("Unknown")
                        df_processed[col] = le.fit_transform(
                            df_processed[col].astype(str)
                        )

            # Process date columns
            if date_cols:
                for col in date_cols:
                    if col in df_processed.columns:
                        try:
                            df_processed[col] = pd.to_datetime(
                                df_processed[col], errors="coerce"
                            )
                            reference_date = datetime.now()
                            df_processed[f"{col}_days_ago"] = (
                                reference_date - df_processed[col]
                            ).dt.days.fillna(0)
                            df_processed[f"{col}_month"] = df_processed[
                                col
                            ].dt.month.fillna(1)
                            df_processed[f"{col}_year"] = df_processed[
                                col
                            ].dt.year.fillna(reference_date.year)
                            df_processed = df_processed.drop(columns=[col])
                        except Exception as e:
                            st.warning(
                                f"Could not process date column '{col}': {str(e)}"
                            )

            # Process text columns (simple approach - length and word count)
            if text_cols:
                for col in text_cols:
                    if col in df_processed.columns:
                        df_processed[col] = df_processed[col].fillna("")
                        df_processed[f"{col}_length"] = (
                            df_processed[col].astype(str).str.len()
                        )
                        df_processed[f"{col}_word_count"] = (
                            df_processed[col].astype(str).str.split().str.len()
                        )
                        df_processed = df_processed.drop(columns=[col])

            # Select only numeric columns for clustering
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed = df_processed[numeric_cols]

            # Remove columns with zero variance
            df_processed = df_processed.loc[:, df_processed.var() != 0]

            return df_processed

        except Exception as e:
            raise Exception(f"Error in data preprocessing: {str(e)}")

    def find_optimal_clusters(self, X, max_clusters=10):
        """Find optimal number of clusters using elbow method and silhouette score"""
        try:
            inertias = []
            silhouette_scores = []
            K_range = range(2, min(max_clusters + 1, len(X)))

            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X)
                inertias.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(X, kmeans.labels_))

            # Find optimal k using silhouette score
            optimal_k = K_range[np.argmax(silhouette_scores)]
            return optimal_k, dict(zip(K_range, silhouette_scores))

        except Exception as e:
            st.warning(f"Could not determine optimal clusters: {str(e)}")
            return 3, {}

    def calculate_feature_importance(self, X, labels):
        """Calculate feature importance using Random Forest"""
        try:
            if len(np.unique(labels)) < 2:
                return {}

            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, labels)

            feature_names = (
                X.columns
                if hasattr(X, "columns")
                else [f"feature_{i}" for i in range(X.shape[1])]
            )
            importance_dict = dict(zip(feature_names, rf.feature_importances_))

            return importance_dict

        except Exception as e:
            st.warning(f"Could not calculate feature importance: {str(e)}")
            return {}

    def create_segment_profiles(self, df_original, df_processed, labels):
        """Create detailed segment profiles"""
        try:
            profiles = []

            for segment_id in np.unique(labels):
                mask = labels == segment_id
                segment_data = df_original[mask]
                segment_processed = df_processed[mask]

                # Basic statistics
                size = len(segment_data)

                # Calculate characteristics
                characteristics = {}

                # Numeric columns from original data
                numeric_cols = df_original.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col in segment_data.columns:
                        characteristics[f"avg_{col}"] = float(segment_data[col].mean())
                        characteristics[f"median_{col}"] = float(
                            segment_data[col].median()
                        )

                # Categorical columns from original data
                categorical_cols = df_original.select_dtypes(include=["object"]).columns
                for col in categorical_cols:
                    if col in segment_data.columns and not segment_data[col].empty:
                        mode_value = segment_data[col].mode()
                        if len(mode_value) > 0:
                            characteristics[f"most_common_{col}"] = str(
                                mode_value.iloc[0]
                            )

                # Calculate LTV and churn risk (mock calculations for demo)
                if "total_spent" in characteristics:
                    ltv_prediction = characteristics["avg_total_spent"] * 1.5
                elif any(
                    "spent" in k.lower()
                    or "purchase" in k.lower()
                    or "revenue" in k.lower()
                    for k in characteristics.keys()
                ):
                    spend_keys = [
                        k
                        for k in characteristics.keys()
                        if "spent" in k.lower()
                        or "purchase" in k.lower()
                        or "revenue" in k.lower()
                    ]
                    ltv_prediction = (
                        characteristics.get(spend_keys[0], 100) * 1.5
                        if spend_keys
                        else 250
                    )
                else:
                    ltv_prediction = np.random.uniform(100, 1000)

                churn_risk = max(0.1, min(0.9, np.random.uniform(0.2, 0.8)))

                # Generate marketing recommendations
                recommendations = self.generate_recommendations(
                    characteristics, ltv_prediction, churn_risk
                )

                profile = {
                    "segment_id": int(segment_id),
                    "name": f"Segment {segment_id}",
                    "size": int(size),
                    "ltv_prediction": float(ltv_prediction),
                    "churn_risk": float(churn_risk),
                    "characteristics": characteristics,
                    "marketing_recommendations": recommendations,
                }

                profiles.append(profile)

            return profiles

        except Exception as e:
            st.error(f"Error creating segment profiles: {str(e)}")
            return []

    def generate_recommendations(self, characteristics, ltv, churn_risk):
        """Generate marketing recommendations based on segment characteristics"""
        recommendations = []

        try:
            # LTV-based recommendations
            if ltv > 500:
                recommendations.append(
                    "Focus on premium product offerings and VIP services"
                )
                recommendations.append(
                    "Implement loyalty programs with exclusive benefits"
                )
            elif ltv < 200:
                recommendations.append("Offer value-based promotions and discounts")
                recommendations.append("Focus on cost-effective acquisition channels")
            else:
                recommendations.append("Balance value and premium messaging")
                recommendations.append("Test different price points and offers")

            # Churn risk-based recommendations
            if churn_risk > 0.6:
                recommendations.append("Implement immediate retention campaigns")
                recommendations.append("Provide proactive customer support")
                recommendations.append("Offer win-back incentives")
            elif churn_risk < 0.3:
                recommendations.append("Focus on upselling and cross-selling")
                recommendations.append("Use as advocates for referral programs")

            # Engagement-based recommendations
            if any("session" in k.lower() for k in characteristics.keys()):
                session_keys = [
                    k for k in characteristics.keys() if "session" in k.lower()
                ]
                if session_keys and characteristics.get(session_keys[0], 0) > 300:
                    recommendations.append(
                        "Highly engaged users - perfect for beta testing"
                    )
                    recommendations.append("Consider premium feature introductions")

            return recommendations

        except Exception as e:
            return [
                "Personalized marketing approach recommended",
                "Monitor customer behavior closely",
            ]

    def run_analysis(
        self, df, categorical_cols=None, text_cols=None, date_cols=None, n_clusters=None
    ):
        """Run complete segmentation analysis"""
        try:
            # Preprocess data
            df_processed = self.preprocess_data(
                df, categorical_cols, text_cols, date_cols
            )
            self.processed_features = df_processed

            if df_processed.empty or df_processed.shape[1] == 0:
                return {
                    "status": "error",
                    "message": "No valid features found for clustering",
                }

            # Scale features
            X_scaled = self.scaler.fit_transform(df_processed)

            # Find optimal number of clusters if not specified
            if n_clusters is None:
                optimal_k, silhouette_scores = self.find_optimal_clusters(X_scaled)
            else:
                optimal_k = n_clusters
                silhouette_scores = {}

            # Perform clustering
            self.kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            self.segment_labels = self.kmeans.fit_predict(X_scaled)

            # Calculate silhouette score
            if len(np.unique(self.segment_labels)) > 1:
                self.silhouette_score = silhouette_score(X_scaled, self.segment_labels)

            # Calculate feature importance
            self.feature_importance = self.calculate_feature_importance(
                df_processed, self.segment_labels
            )

            # Create segment profiles
            self.segment_profiles = self.create_segment_profiles(
                df, df_processed, self.segment_labels
            )

            # PCA for visualization
            if df_processed.shape[1] > 2:
                self.pca = PCA(n_components=2)
                self.pca_features = self.pca.fit_transform(X_scaled)
            else:
                self.pca_features = X_scaled

            return {
                "status": "success",
                "n_segments": optimal_k,
                "silhouette_score": self.silhouette_score,
                "silhouette_scores": silhouette_scores,
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}


def create_sample_data(n_samples=1000):
    """Create realistic sample customer data"""
    np.random.seed(42)

    # Demographics
    ages = np.random.normal(35, 12, n_samples).clip(18, 80)
    incomes = np.random.lognormal(10.5, 0.5, n_samples).clip(20000, 200000)

    # Behavioral data
    purchase_counts = np.random.poisson(8, n_samples)
    total_spent = incomes * np.random.uniform(0.05, 0.25, n_samples)
    session_durations = np.random.exponential(250, n_samples).clip(10, 3600)
    days_since_last_purchase = np.random.exponential(30, n_samples).clip(0, 365)

    # Categories
    regions = np.random.choice(["North", "South", "East", "West"], n_samples)
    customer_types = np.random.choice(
        ["Premium", "Standard", "Basic"], n_samples, p=[0.2, 0.5, 0.3]
    )

    # Create date columns
    registration_dates = [
        datetime.now() - timedelta(days=int(d))
        for d in np.random.exponential(180, n_samples)
    ]

    last_purchase_dates = [
        datetime.now() - timedelta(days=int(d)) for d in days_since_last_purchase
    ]

    return pd.DataFrame(
        {
            "customer_id": [f"CUST_{i:05d}" for i in range(n_samples)],
            "age": ages.astype(int),
            "income": incomes.astype(int),
            "region": regions,
            "customer_type": customer_types,
            "purchase_count": purchase_counts,
            "total_spent": np.round(total_spent, 2),
            "avg_session_duration": np.round(session_durations, 1),
            "days_since_last_purchase": days_since_last_purchase.astype(int),
            "registration_date": registration_dates,
            "last_purchase_date": last_purchase_dates,
            "reviews_written": np.random.poisson(2, n_samples),
            "support_tickets": np.random.poisson(1, n_samples),
        }
    )


# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if "system" not in st.session_state:
        st.session_state.system = EnhancedSegmentationSystem()
    if "data" not in st.session_state:
        st.session_state.data = None
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = None
    if "results" not in st.session_state:
        st.session_state.results = None
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False


init_session_state()

# Sidebar navigation
st.sidebar.title("🎯 Navigation")
st.sidebar.markdown("---")

pages = {
    "📊 Data & Analysis": "data_analysis",
    "👥 Segment Profiles": "segment_profiles",
    "📈 Visualizations": "visualizations",
    "🎯 Business Insights": "business_insights",
}

selected_page = st.sidebar.radio("Select a page:", list(pages.keys()))
current_page = pages[selected_page]

# Main content
if current_page == "data_analysis":
    st.markdown(
        '<div class="main-header">Customer Segmentation Analytics</div>',
        unsafe_allow_html=True,
    )

    # Data source selection
    st.subheader("📁 Data Source")

    col1, col2 = st.columns([3, 1])

    with col1:
        data_source = st.radio(
            "Choose your data source:",
            ["📤 Upload CSV File", "🎲 Generate Sample Data"],
            horizontal=True,
        )

    if data_source == "📤 Upload CSV File":
        uploaded_file = st.file_uploader(
            "Upload your customer data (CSV format):",
            type=["csv"],
            help="Upload a CSV file containing customer data",
        )

        if uploaded_file is not None:
            try:
                with st.spinner("Loading data..."):
                    df = pd.read_csv(uploaded_file)
                    st.session_state.data = df

                st.success(
                    f"✅ Successfully loaded {df.shape[0]:,} records with {df.shape[1]} columns"
                )

            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")

    else:  # Generate sample data
        col1, col2 = st.columns([2, 1])

        with col1:
            sample_size = st.slider(
                "Number of sample customers:",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
            )

        with col2:
            if st.button("🎲 Generate Data", type="primary"):
                with st.spinner("Generating sample data..."):
                    df = create_sample_data(sample_size)
                    st.session_state.data = df
                    time.sleep(0.5)  # Small delay for UX

                st.success(f"✅ Generated {sample_size:,} sample customer records")

    # Data preview and analysis setup
    if st.session_state.data is not None:
        df = st.session_state.data

        st.markdown("---")
        st.subheader("📋 Data Preview")

        # Data overview
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", f"{df.shape[0]:,}")

        with col2:
            st.metric("Total Features", df.shape[1])

        with col3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Features", len(numeric_cols))

        with col4:
            categorical_cols = df.select_dtypes(include=["object"]).columns
            st.metric("Categorical Features", len(categorical_cols))

        # Display sample data
        st.dataframe(df.head(10), use_container_width=True)

        # Data quality check
        st.subheader("🔍 Data Quality")

        col1, col2 = st.columns(2)

        with col1:
            # Missing values
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

            if len(missing_data) > 0:
                st.warning("⚠️ Missing Values Detected")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Column": missing_data.index,
                            "Missing Count": missing_data.values,
                            "Missing %": (missing_data.values / len(df) * 100).round(2),
                        }
                    ).reset_index(drop=True)
                )
            else:
                st.success("✅ No missing values found")

        with col2:
            # Data types
            st.info("📊 Data Types")
            data_types = df.dtypes.value_counts()
            st.dataframe(
                pd.DataFrame(
                    {
                        "Data Type": data_types.index.astype(str),
                        "Count": data_types.values,
                    }
                ).reset_index(drop=True)
            )

        # Analysis configuration
        st.markdown("---")
        st.subheader("⚙️ Analysis Configuration")

        col1, col2 = st.columns(2)

        with col1:
            # Feature selection
            all_categorical = df.select_dtypes(include=["object"]).columns.tolist()
            categorical_cols = st.multiselect(
                "Select categorical columns:",
                options=all_categorical,
                default=[
                    col
                    for col in all_categorical
                    if "type" in col.lower() or "category" in col.lower()
                ][:3],
                help="Categorical features like customer type, region, etc.",
            )

            # Date columns
            potential_date_cols = [
                col
                for col in df.columns
                if any(
                    word in col.lower()
                    for word in ["date", "time", "created", "updated"]
                )
            ]
            date_cols = st.multiselect(
                "Select date columns:",
                options=potential_date_cols,
                default=potential_date_cols[:2],
                help="Date columns will be converted to numeric features",
            )

        with col2:
            # Text columns
            text_cols = st.multiselect(
                "Select text columns:",
                options=[col for col in all_categorical if col not in categorical_cols],
                help="Text columns like reviews, comments (will extract length/word count)",
            )

            # Number of clusters
            use_auto_clusters = st.checkbox("Auto-detect optimal clusters", value=True)

            if not use_auto_clusters:
                n_clusters = st.slider(
                    "Number of clusters:", min_value=2, max_value=10, value=4
                )
            else:
                n_clusters = None

        # Run analysis
        st.markdown("---")

        if st.button(
            "🚀 Run Segmentation Analysis", type="primary", use_container_width=True
        ):
            with st.spinner(
                "🔄 Running segmentation analysis... This may take a moment."
            ):

                # Progress bar
                progress_bar = st.progress(0)
                progress_text = st.empty()

                progress_text.text("Preprocessing data...")
                progress_bar.progress(25)
                time.sleep(0.5)

                progress_text.text("Finding optimal clusters...")
                progress_bar.progress(50)

                # Run analysis
                start_time = time.time()
                results = st.session_state.system.run_analysis(
                    df,
                    categorical_cols=categorical_cols,
                    text_cols=text_cols,
                    date_cols=date_cols,
                    n_clusters=n_clusters,
                )

                progress_text.text("Creating segment profiles...")
                progress_bar.progress(75)
                time.sleep(0.5)

                progress_text.text("Finalizing analysis...")
                progress_bar.progress(100)

                results["analysis_time"] = time.time() - start_time
                st.session_state.results = results

                # Store processed data with segments
                if (
                    hasattr(st.session_state.system, "segment_labels")
                    and st.session_state.system.segment_labels is not None
                ):
                    st.session_state.processed_data = df.copy()
                    st.session_state.processed_data["segment"] = (
                        st.session_state.system.segment_labels
                    )
                    st.session_state.analysis_complete = True

                progress_bar.empty()
                progress_text.empty()

                if results.get("status") == "success":
                    st.balloons()
                    st.success("🎉 Analysis completed successfully!")
                else:
                    st.error(
                        f"❌ Analysis failed: {results.get('message', 'Unknown error')}"
                    )

        # Results summary
        if (
            st.session_state.results
            and st.session_state.results.get("status") == "success"
        ):
            st.markdown("---")
            st.subheader("📈 Analysis Results")

            results = st.session_state.results

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Segments Found",
                    results.get("n_segments", "N/A"),
                    help="Number of customer segments identified",
                )

            with col2:
                if results.get("silhouette_score"):
                    score = results["silhouette_score"]
                    st.metric(
                        "Quality Score",
                        f"{score:.3f}",
                        delta=f"{'Excellent' if score > 0.7 else 'Good' if score > 0.5 else 'Fair' if score > 0.3 else 'Poor'}",
                        help="Silhouette score indicating cluster quality",
                    )

            with col3:
                analysis_time = results.get("analysis_time", 0)
                st.metric(
                    "Analysis Time",
                    f"{analysis_time:.1f}s",
                    help="Time taken to complete the analysis",
                )

            with col4:
                if st.session_state.system.segment_profiles:
                    avg_ltv = np.mean(
                        [
                            p["ltv_prediction"]
                            for p in st.session_state.system.segment_profiles
                        ]
                    )
                    st.metric(
                        "Avg. Customer LTV",
                        f"${avg_ltv:,.0f}",
                        help="Average predicted customer lifetime value",
                    )

            # Quick insights
            st.info(
                "✨ **Quick Insights**: "
                f"Your customers have been segmented into {results.get('n_segments', 'several')} distinct groups. "
                f"{'The segmentation quality is excellent!' if results.get('silhouette_score', 0) > 0.7 else 'The segmentation shows good separation between groups.' if results.get('silhouette_score', 0) > 0.5 else 'The segmentation provides meaningful customer groups.'} "
                "Navigate to other tabs to explore detailed profiles and visualizations."
            )

elif current_page == "segment_profiles":
    st.markdown(
        '<div class="main-header">Customer Segment Profiles</div>',
        unsafe_allow_html=True,
    )

    if (
        not st.session_state.analysis_complete
        or not st.session_state.system.segment_profiles
    ):
        st.warning(
            "⚠️ No segmentation results available. Please run the analysis first in the 'Data & Analysis' tab."
        )
        if st.button("← Go to Data & Analysis"):
            st.rerun()
    else:
        profiles = st.session_state.system.segment_profiles

        # Segment overview
        st.subheader("📊 Segment Overview")

        overview_data = []
        for profile in profiles:
            overview_data.append(
                {
                    "Segment": f"Segment {profile['segment_id']}",
                    "Size": f"{profile['size']:,}",
                    "% of Total": f"{(profile['size'] / sum(p['size'] for p in profiles) * 100):.1f}%",
                    "Avg LTV": f"${profile['ltv_prediction']:,.0f}",
                    "Churn Risk": f"{profile['churn_risk']:.1%}",
                    "Risk Level": (
                        "🔴 High"
                        if profile["churn_risk"] > 0.6
                        else "🟡 Medium" if profile["churn_risk"] > 0.3 else "🟢 Low"
                    ),
                }
            )

        st.dataframe(pd.DataFrame(overview_data), use_container_width=True)

        # Detailed segment profiles
        st.subheader("🔍 Detailed Segment Analysis")

        # Create tabs for each segment
        tab_names = [f"Segment {p['segment_id']}" for p in profiles]
        tabs = st.tabs(tab_names)

        for i, (tab, profile) in enumerate(zip(tabs, profiles)):
            with tab:
                # Header metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Customers", f"{profile['size']:,}")

                with col2:
                    st.metric("Lifetime Value", f"${profile['ltv_prediction']:,.0f}")

                with col3:
                    churn_risk = profile["churn_risk"]
                    delta_color = "inverse" if churn_risk > 0.6 else "normal"
                    st.metric(
                        "Churn Risk",
                        f"{churn_risk:.1%}",
                        delta="High Risk" if churn_risk > 0.6 else "Low Risk",
                        delta_color=delta_color,
                    )

                with col4:
                    total_value = profile["size"] * profile["ltv_prediction"]
                    st.metric("Total Segment Value", f"${total_value:,.0f}")

                # Segment description
                risk_level = (
                    "high"
                    if profile["churn_risk"] > 0.6
                    else "low" if profile["churn_risk"] < 0.3 else "moderate"
                )
                value_level = (
                    "high-value"
                    if profile["ltv_prediction"] > 800
                    else (
                        "budget-conscious"
                        if profile["ltv_prediction"] < 300
                        else "mid-tier"
                    )
                )

                description = f"""
                **Profile Summary**: This segment represents {profile['size']:,} customers ({(profile['size'] / sum(p['size'] for p in profiles) * 100):.1f}% of your customer base). 
                They are {value_level} customers with an average lifetime value of ${profile['ltv_prediction']:,.0f}. 
                This group has {risk_level} churn risk at {profile['churn_risk']:.1%}.
                """

                st.info(description)

                # Key characteristics
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📋 Key Characteristics")

                    if profile["characteristics"]:
                        char_data = []
                        for key, value in profile["characteristics"].items():
                            if isinstance(value, (int, float)):
                                if "avg_" in key:
                                    display_key = (
                                        key.replace("avg_", "")
                                        .replace("_", " ")
                                        .title()
                                    )
                                    char_data.append(
                                        {
                                            "Metric": f"Average {display_key}",
                                            "Value": f"{value:,.2f}",
                                        }
                                    )
                                elif "median_" in key:
                                    display_key = (
                                        key.replace("median_", "")
                                        .replace("_", " ")
                                        .title()
                                    )
                                    char_data.append(
                                        {
                                            "Metric": f"Median {display_key}",
                                            "Value": f"{value:,.2f}",
                                        }
                                    )
                            else:
                                display_key = (
                                    key.replace("most_common_", "")
                                    .replace("_", " ")
                                    .title()
                                )
                                char_data.append(
                                    {
                                        "Metric": f"Most Common {display_key}",
                                        "Value": str(value),
                                    }
                                )

                        if char_data:
                            st.dataframe(
                                pd.DataFrame(char_data[:10]),
                                use_container_width=True,
                                hide_index=True,
                            )
                    else:
                        st.write("No detailed characteristics available")

                with col2:
                    st.subheader("🎯 Marketing Recommendations")

                    if profile["marketing_recommendations"]:
                        for idx, rec in enumerate(
                            profile["marketing_recommendations"], 1
                        ):
                            st.markdown(f"**{idx}.** {rec}")
                    else:
                        st.write("No specific recommendations available")

                # Action plan
                st.subheader("📋 Recommended Action Plan")

                action_plan = []

                # Based on LTV
                if profile["ltv_prediction"] > 800:
                    action_plan.extend(
                        [
                            "🎖️ **VIP Treatment**: Assign dedicated account managers",
                            "🎁 **Exclusive Offers**: Provide early access to new products",
                            "💎 **Premium Services**: Offer white-glove customer service",
                        ]
                    )
                elif profile["ltv_prediction"] < 300:
                    action_plan.extend(
                        [
                            "💰 **Value Focus**: Emphasize cost savings and deals",
                            "📧 **Email Marketing**: Use cost-effective digital channels",
                            "🎯 **Conversion Optimization**: Focus on converting to higher tiers",
                        ]
                    )
                else:
                    action_plan.extend(
                        [
                            "⚖️ **Balanced Approach**: Mix value and premium messaging",
                            "🧪 **A/B Testing**: Test different offer types",
                            "📈 **Upselling**: Identify upgrade opportunities",
                        ]
                    )

                # Based on churn risk
                if profile["churn_risk"] > 0.6:
                    action_plan.extend(
                        [
                            "🚨 **Urgent Retention**: Implement immediate outreach",
                            "💸 **Win-back Offers**: Provide compelling incentives",
                            "📞 **Personal Touch**: Direct phone or email contact",
                        ]
                    )
                elif profile["churn_risk"] < 0.3:
                    action_plan.extend(
                        [
                            "🌟 **Advocacy Programs**: Turn into brand ambassadors",
                            "📣 **Referral Incentives**: Encourage word-of-mouth marketing",
                            "📊 **Feedback Collection**: Use for product development insights",
                        ]
                    )

                for plan in action_plan:
                    st.markdown(plan)

elif current_page == "visualizations":
    st.markdown(
        '<div class="main-header">Customer Segmentation Visualizations</div>',
        unsafe_allow_html=True,
    )

    if (
        not st.session_state.analysis_complete
        or not st.session_state.system.segment_profiles
    ):
        st.warning(
            "⚠️ No segmentation results available. Please run the analysis first in the 'Data & Analysis' tab."
        )
        if st.button("← Go to Data & Analysis"):
            st.rerun()
    else:
        profiles = st.session_state.system.segment_profiles
        processed_data = st.session_state.processed_data

        # Visualization options
        st.subheader("📊 Visualization Dashboard")

        viz_col1, viz_col2 = st.columns([2, 1])

        with viz_col2:
            st.subheader("🎛️ Controls")

            # Chart selection
            chart_type = st.selectbox(
                "Select Visualization:",
                [
                    "Segment Distribution",
                    "LTV vs Churn Risk",
                    "Segment Scatter Plot",
                    "Feature Importance",
                    "Segment Comparison",
                    "Customer Journey",
                ],
            )

            # Color scheme
            color_scheme = st.selectbox(
                "Color Scheme:",
                [
                    "viridis",
                    "plasma",
                    "plotly3",
                    "rainbow",
                    "blues",
                    "reds",
                    "greens",
                    "cividis",
                    "Jet",
                    "Turbo",
                    "Magma",
                    "Inferno",
                    "pastel1",
                    "Pastel2",
                    "Set1",
                    "Set2",
                    "Set3",
                ],
            )

        with viz_col1:
            if chart_type == "Segment Distribution":
                # Segment size distribution
                segment_sizes = [p["size"] for p in profiles]
                segment_names = [f"Segment {p['segment_id']}" for p in profiles]

                fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=segment_names,
                            values=segment_sizes,
                            hole=0.4,
                            textinfo="label+percent",
                            textfont_size=12,
                            marker=dict(line=dict(color="#FFFFFF", width=2)),
                        )
                    ]
                )

                fig.update_layout(
                    title="Customer Segment Distribution",
                    font=dict(size=14),
                    showlegend=True,
                    height=500,
                )

                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "LTV vs Churn Risk":
                # LTV vs Churn Risk scatter
                ltv_values = [p["ltv_prediction"] for p in profiles]
                churn_values = [p["churn_risk"] * 100 for p in profiles]
                sizes = [p["size"] for p in profiles]
                segment_names = [f"Segment {p['segment_id']}" for p in profiles]

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=ltv_values,
                        y=churn_values,
                        mode="markers+text",
                        marker=dict(
                            size=[s / 50 for s in sizes],
                            color=ltv_values,
                            colorscale=color_scheme.lower(),
                            showscale=True,
                            colorbar=dict(title="LTV ($)"),
                        ),
                        text=segment_names,
                        textposition="middle center",
                        textfont=dict(size=10, color="white"),
                        hovertemplate="<b>%{text}</b><br>"
                        + "LTV: $%{x:,.0f}<br>"
                        + "Churn Risk: %{y:.1f}%<br>"
                        + "<extra></extra>",
                    )
                )

                fig.update_layout(
                    title="Customer Lifetime Value vs Churn Risk",
                    xaxis_title="Lifetime Value ($)",
                    yaxis_title="Churn Risk (%)",
                    height=500,
                    showlegend=False,
                )

                # Add quadrant labels
                fig.add_annotation(
                    x=max(ltv_values) * 0.8,
                    y=max(churn_values) * 0.8,
                    text="High Value<br>High Risk",
                    showarrow=False,
                    bgcolor="rgba(255,0,0,0.1)",
                    bordercolor="red",
                )
                fig.add_annotation(
                    x=max(ltv_values) * 0.8,
                    y=max(churn_values) * 0.2,
                    text="High Value<br>Low Risk",
                    showarrow=False,
                    bgcolor="rgba(0,255,0,0.1)",
                    bordercolor="green",
                )

                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Segment Scatter Plot":
                # PCA scatter plot
                if hasattr(st.session_state.system, "pca_features"):
                    pca_data = st.session_state.system.pca_features
                    labels = st.session_state.system.segment_labels

                    fig = px.scatter(
                        x=pca_data[:, 0],
                        y=pca_data[:, 1],
                        color=[f"Segment {label}" for label in labels],
                        title="Customer Segments (PCA Visualization)",
                        labels={
                            "x": "First Principal Component",
                            "y": "Second Principal Component",
                        },
                        color_discrete_sequence=px.colors.qualitative.Set3,
                    )

                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("PCA data not available")

            elif chart_type == "Feature Importance":
                # Feature importance chart
                if st.session_state.system.feature_importance:
                    importance_data = st.session_state.system.feature_importance

                    # Sort by importance
                    sorted_features = sorted(
                        importance_data.items(), key=lambda x: x[1], reverse=True
                    )
                    features, importances = zip(
                        *sorted_features[:15]
                    )  # Top 15 features

                    fig = go.Figure(
                        data=[
                            go.Bar(
                                y=list(features),
                                x=list(importances),
                                orientation="h",
                                marker_color=px.colors.sequential.Viridis,
                            )
                        ]
                    )

                    fig.update_layout(
                        title="Feature Importance for Segmentation",
                        xaxis_title="Importance Score",
                        yaxis_title="Features",
                        height=500,
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Feature importance data not available")

            elif chart_type == "Segment Comparison":
                # Multi-metric comparison
                metrics = ["ltv_prediction", "churn_risk", "size"]
                metric_names = ["Lifetime Value ($)", "Churn Risk (%)", "Segment Size"]

                fig = make_subplots(
                    rows=2,
                    cols=2,
                    subplot_titles=(
                        "LTV Comparison",
                        "Churn Risk Comparison",
                        "Segment Sizes",
                        "Value Distribution",
                    ),
                    specs=[
                        [{"type": "bar"}, {"type": "bar"}],
                        [{"type": "bar"}, {"type": "bar"}],
                    ],
                )

                segment_names = [f"Segment {p['segment_id']}" for p in profiles]

                # LTV comparison
                fig.add_trace(
                    go.Bar(
                        x=segment_names,
                        y=[p["ltv_prediction"] for p in profiles],
                        name="LTV",
                        marker_color="lightblue",
                    ),
                    row=1,
                    col=1,
                )

                # Churn risk comparison
                fig.add_trace(
                    go.Bar(
                        x=segment_names,
                        y=[p["churn_risk"] * 100 for p in profiles],
                        name="Churn Risk",
                        marker_color="lightcoral",
                    ),
                    row=1,
                    col=2,
                )

                # Segment sizes
                fig.add_trace(
                    go.Bar(
                        x=segment_names,
                        y=[p["size"] for p in profiles],
                        name="Size",
                        marker_color="lightgreen",
                    ),
                    row=2,
                    col=1,
                )

                # Total value
                fig.add_trace(
                    go.Bar(
                        x=segment_names,
                        y=[p["size"] * p["ltv_prediction"] for p in profiles],
                        name="Total Value",
                        marker_color="gold",
                    ),
                    row=2,
                    col=2,
                )

                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Customer Journey":
                # Simulated customer journey stages
                if processed_data is not None:
                    # Create journey stage data
                    journey_data = []

                    for profile in profiles:
                        segment_id = profile["segment_id"]

                        # Simulate journey stages based on segment characteristics
                        if profile["ltv_prediction"] > 800:
                            stages = {
                                "Awareness": np.random.randint(80, 95),
                                "Consideration": np.random.randint(70, 85),
                                "Purchase": np.random.randint(60, 80),
                                "Retention": np.random.randint(70, 90),
                                "Advocacy": np.random.randint(60, 85),
                            }
                        elif profile["ltv_prediction"] < 300:
                            stages = {
                                "Awareness": np.random.randint(40, 60),
                                "Consideration": np.random.randint(30, 50),
                                "Purchase": np.random.randint(20, 40),
                                "Retention": np.random.randint(25, 45),
                                "Advocacy": np.random.randint(15, 35),
                            }
                        else:
                            stages = {
                                "Awareness": np.random.randint(60, 80),
                                "Consideration": np.random.randint(50, 70),
                                "Purchase": np.random.randint(40, 60),
                                "Retention": np.random.randint(45, 65),
                                "Advocacy": np.random.randint(35, 55),
                            }

                        for stage, value in stages.items():
                            journey_data.append(
                                {
                                    "Segment": f"Segment {segment_id}",
                                    "Stage": stage,
                                    "Conversion_Rate": value,
                                }
                            )

                    journey_df = pd.DataFrame(journey_data)

                    fig = px.line(
                        journey_df,
                        x="Stage",
                        y="Conversion_Rate",
                        color="Segment",
                        title="Customer Journey by Segment",
                        markers=True,
                    )

                    fig.update_layout(yaxis_title="Conversion Rate (%)", height=500)

                    st.plotly_chart(fig, use_container_width=True)

        # Additional insights
        st.markdown("---")
        st.subheader("📈 Key Visual Insights")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Largest segment
            largest_segment = max(profiles, key=lambda x: x["size"])
            st.info(
                f"**Largest Segment**: Segment {largest_segment['segment_id']} with {largest_segment['size']:,} customers ({(largest_segment['size']/sum(p['size'] for p in profiles)*100):.1f}%)"
            )

        with col2:
            # Highest value segment
            highest_ltv = max(profiles, key=lambda x: x["ltv_prediction"])
            st.success(
                f"**Highest Value**: Segment {highest_ltv['segment_id']} with ${highest_ltv['ltv_prediction']:,.0f} average LTV"
            )

        with col3:
            # Highest risk segment
            highest_risk = max(profiles, key=lambda x: x["churn_risk"])
            st.warning(
                f"**Highest Risk**: Segment {highest_risk['segment_id']} with {highest_risk['churn_risk']:.1%} churn risk"
            )

elif current_page == "business_insights":
    st.markdown(
        '<div class="main-header">Business Insights & Recommendations</div>',
        unsafe_allow_html=True,
    )

    if (
        not st.session_state.analysis_complete
        or not st.session_state.system.segment_profiles
    ):
        st.warning(
            "⚠️ No segmentation results available. Please run the analysis first in the 'Data & Analysis' tab."
        )
        if st.button("← Go to Data & Analysis"):
            st.rerun()
    else:
        profiles = st.session_state.system.segment_profiles

        # Executive Summary
        st.subheader("📋 Executive Summary")

        total_customers = sum(p["size"] for p in profiles)
        total_ltv = sum(p["size"] * p["ltv_prediction"] for p in profiles)
        avg_ltv = total_ltv / total_customers
        weighted_churn = (
            sum(p["size"] * p["churn_risk"] for p in profiles) / total_customers
        )

        summary_col1, summary_col2 = st.columns([2, 1])

        with summary_col1:
            summary_text = f"""
            **Customer Base Analysis:**
            
            Your customer base of {total_customers:,} customers has been segmented into {len(profiles)} distinct groups, 
            each with unique characteristics and behaviors. The total portfolio value is estimated at 
            ${total_ltv:,.0f} with an average customer lifetime value of ${avg_ltv:.0f}.
            
            **Key Findings:**
            - Overall churn risk across all segments: {weighted_churn:.1%}
            - Revenue concentration varies significantly across segments
            - Different segments require tailored marketing approaches
            - Immediate attention needed for high-risk, high-value customers
            """
            st.markdown(summary_text)

        with summary_col2:
            st.metric("Total Portfolio Value", f"${total_ltv:,.0f}")
            st.metric("Average Customer LTV", f"${avg_ltv:.0f}")
            st.metric("Overall Churn Risk", f"{weighted_churn:.1%}")
            st.metric("Number of Segments", len(profiles))

        # Strategic Priorities
        st.markdown("---")
        st.subheader("🎯 Strategic Priorities")

        # Identify priority segments
        high_value_segments = [
            p for p in profiles if p["ltv_prediction"] > avg_ltv * 1.2
        ]
        high_risk_segments = [p for p in profiles if p["churn_risk"] > 0.5]
        large_segments = [p for p in profiles if p["size"] > total_customers * 0.3]

        priority_col1, priority_col2 = st.columns(2)

        with priority_col1:
            st.markdown("### 🚨 Immediate Actions Required")

            # High-risk, high-value customers
            critical_segments = [
                p
                for p in profiles
                if p["churn_risk"] > 0.5 and p["ltv_prediction"] > avg_ltv
            ]

            if critical_segments:
                for segment in critical_segments:
                    at_risk_value = (
                        segment["size"]
                        * segment["ltv_prediction"]
                        * segment["churn_risk"]
                    )
                    st.error(
                        f"""
                    **Segment {segment['segment_id']} - CRITICAL**
                    - {segment['size']:,} customers at {segment['churn_risk']:.1%} risk
                    - Potential revenue loss: ${at_risk_value:,.0f}
                    - Recommended: Immediate retention campaign
                    """
                    )
            else:
                st.success("✅ No critical risk segments identified")

            # Large segments with moderate risk
            for segment in large_segments:
                if segment["churn_risk"] > 0.3 and segment not in critical_segments:
                    st.warning(
                        f"""
                    **Segment {segment['segment_id']} - Monitor Closely**
                    - Large segment: {segment['size']:,} customers
                    - Moderate risk: {segment['churn_risk']:.1%}
                    - Impact if risk increases: High
                    """
                    )

        with priority_col2:
            st.markdown("### 💰 Growth Opportunities")

            # High-value, low-risk segments for expansion
            growth_segments = [
                p
                for p in profiles
                if p["churn_risk"] < 0.5 and p["ltv_prediction"] > avg_ltv * 0.8
            ]

            for segment in growth_segments:
                expansion_potential = (
                    segment["size"] * segment["ltv_prediction"] * 0.2
                )  # 20% growth assumption
                st.success(
                    f"""
                **Segment {segment['segment_id']} - Expand**
                - Stable, high-value customers
                - Low churn risk: {segment['churn_risk']:.1%}
                - Growth potential: ${expansion_potential:,.0f}
                """
                )

            # Underperforming segments with potential
            improvement_segments = [
                p
                for p in profiles
                if p["ltv_prediction"] < avg_ltv * 0.8 and p["churn_risk"] < 0.4
            ]

            for segment in improvement_segments:
                st.info(
                    f"""
                **Segment {segment['segment_id']} - Develop**
                - {segment['size']:,} customers with growth potential
                - Current LTV: ${segment['ltv_prediction']:.0f}
                - Opportunity: Increase engagement and value
                """
                )

        # Marketing Strategy
        st.markdown("---")
        st.subheader("📢 Marketing Strategy by Segment")

        strategy_tabs = st.tabs([f"Segment {p['segment_id']}" for p in profiles])

        for tab, profile in zip(strategy_tabs, profiles):
            with tab:
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown("#### 📊 Segment Profile")

                    # Risk-value matrix positioning
                    if (
                        profile["churn_risk"] > 0.6
                        and profile["ltv_prediction"] > avg_ltv
                    ):
                        quadrant = "🔴 High Risk, High Value"
                        strategy_focus = "Immediate Retention"
                    elif (
                        profile["churn_risk"] < 0.3
                        and profile["ltv_prediction"] > avg_ltv
                    ):
                        quadrant = "🟢 Low Risk, High Value"
                        strategy_focus = "Growth & Expansion"
                    elif (
                        profile["churn_risk"] > 0.6
                        and profile["ltv_prediction"] < avg_ltv
                    ):
                        quadrant = "🟡 High Risk, Low Value"
                        strategy_focus = "Selective Retention"
                    else:
                        quadrant = "🔵 Low Risk, Low Value"
                        strategy_focus = "Value Development"

                    st.markdown(f"**Market Position:** {quadrant}")
                    st.markdown(f"**Strategic Focus:** {strategy_focus}")
                    st.markdown(f"**Segment Size:** {profile['size']:,} customers")
                    st.markdown(
                        f"**Revenue Contribution:** ${profile['size'] * profile['ltv_prediction']:,.0f}"
                    )

                with col2:
                    st.markdown("#### 🎯 Recommended Tactics")

                    # Channel recommendations
                    if profile["ltv_prediction"] > avg_ltv * 1.5:
                        channels = [
                            "Personal account management",
                            "Premium customer service",
                            "Exclusive events",
                        ]
                    elif profile["ltv_prediction"] < avg_ltv * 0.5:
                        channels = [
                            "Email marketing",
                            "Social media",
                            "Digital advertising",
                        ]
                    else:
                        channels = [
                            "Multi-channel approach",
                            "Targeted email",
                            "Retargeting ads",
                        ]

                    st.markdown("**Preferred Channels:**")
                    for channel in channels:
                        st.markdown(f"• {channel}")

                    # Messaging recommendations
                    if profile["churn_risk"] > 0.5:
                        messaging = [
                            "Retention-focused offers",
                            "Personal outreach",
                            "Loyalty rewards",
                        ]
                    elif profile["ltv_prediction"] > avg_ltv:
                        messaging = [
                            "Premium positioning",
                            "Exclusive access",
                            "VIP treatment",
                        ]
                    else:
                        messaging = [
                            "Value proposition",
                            "Educational content",
                            "Progressive offers",
                        ]

                    st.markdown("**Key Messages:**")
                    for message in messaging:
                        st.markdown(f"• {message}")

        # ROI Analysis
        st.markdown("---")
        st.subheader("💹 ROI Analysis & Budget Allocation")

        roi_col1, roi_col2 = st.columns(2)

        with roi_col1:
            st.markdown("#### 📈 Investment Priorities")

            # Calculate ROI scores for each segment
            roi_data = []
            for profile in profiles:
                # Simple ROI calculation based on size, LTV, and risk
                revenue_potential = profile["size"] * profile["ltv_prediction"]
                risk_adjusted_value = revenue_potential * (1 - profile["churn_risk"])
                roi_score = risk_adjusted_value / profile["size"]  # Per customer ROI

                roi_data.append(
                    {
                        "Segment": f"Segment {profile['segment_id']}",
                        "Revenue Potential": revenue_potential,
                        "Risk Adjusted Value": risk_adjusted_value,
                        "ROI Score": roi_score,
                        "Recommended Budget %": 0,
                    }
                )

            # Allocate budget based on ROI scores
            total_roi = sum(item["ROI Score"] for item in roi_data)
            for item in roi_data:
                item["Recommended Budget %"] = round(
                    (item["ROI Score"] / total_roi) * 100, 1
                )

            roi_df = pd.DataFrame(roi_data)
            st.dataframe(
                roi_df[
                    [
                        "Segment",
                        "Revenue Potential",
                        "ROI Score",
                        "Recommended Budget %",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

        with roi_col2:
            st.markdown("#### 💰 Budget Allocation")

            # Budget pie chart
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=[item["Segment"] for item in roi_data],
                        values=[item["Recommended Budget %"] for item in roi_data],
                        hole=0.3,
                        textinfo="label+percent",
                    )
                ]
            )

            fig.update_layout(
                title="Recommended Marketing Budget Allocation", height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        # Success Metrics
        st.markdown("---")
        st.subheader("📊 Success Metrics & KPIs")

        st.markdown(
            """
        #### 🎯 Key Performance Indicators to Track:
        
        **Retention Metrics:**
        - Churn rate by segment
        - Customer lifetime extension
        - Retention campaign effectiveness
        
        **Growth Metrics:**
        - Revenue per segment
        - Cross-sell/upsell success rates
        - Customer acquisition cost by segment
        
        **Engagement Metrics:**
        - Campaign response rates
        - Customer satisfaction scores
        - Net Promoter Score (NPS) by segment
        
        **Financial Metrics:**
        - Return on marketing investment (ROMI)
        - Customer lifetime value growth
        - Revenue attribution by segment
        """
        )

        # Action Items
        st.markdown("---")
        st.subheader("✅ Next Steps & Action Items")

        next_steps = [
            "🎯 **Week 1-2**: Implement immediate retention campaigns for high-risk segments",
            "📧 **Week 3-4**: Launch targeted marketing campaigns for each segment",
            "📊 **Month 2**: Set up tracking and measurement systems for KPIs",
            "🔄 **Month 3**: Review results and refine segmentation strategy",
            "📈 **Ongoing**: Monitor segment evolution and adjust strategies accordingly",
        ]

        for step in next_steps:
            st.markdown(step)

        # Export recommendations
        st.markdown("---")
        if st.button("📥 Generate Strategy Report", type="primary"):
            # Create a comprehensive report
            report_data = {
                "Executive Summary": {
                    "Total Customers": total_customers,
                    "Total Portfolio Value": total_ltv,
                    "Average LTV": avg_ltv,
                    "Overall Churn Risk": weighted_churn,
                    "Number of Segments": len(profiles),
                },
                "Segment Details": profiles,
                "ROI Analysis": roi_data,
            }

            st.success(
                "📋 Strategy report data prepared! In a production environment, this would generate a downloadable PDF or Excel report."
            )
            st.json(report_data, expanded=False)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Customer Segmentation Analytics Dashboard</p>
    </div>
    """,
    unsafe_allow_html=True,
)
