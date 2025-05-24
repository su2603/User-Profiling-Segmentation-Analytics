# 📘 User Segmentation Analytics App

## Overview

This project is a full-featured Streamlit web application for advanced **customer segmentation analytics**. It enables businesses to:

- Upload or generate customer data
- Perform intelligent segmentation using machine learning
- View detailed profiles, visualizations, and actionable insights
- Export segment information and predictions

The system includes two main components:

- `userSegmentationAnalyticsApp.py`: Streamlit-based front-end
- `userSegmentationSys.py`: Back-end engine for processing, clustering, modeling, and insights

---

## 📁 File Structure

| File                              | Description                                                                                        |
| --------------------------------- | -------------------------------------------------------------------------------------------------- |
| `userSegmentationAnalyticsApp.py` | Streamlit frontend with UI, user interactions, and visualization                                   |
| `userSegmentationSys.py`          | Core backend logic including preprocessing, clustering, LTV/churn modeling, and profile generation |

---

## 🔧 Technical Architecture

### 1. **Data Ingestion**

- **Sources**: Upload CSV or generate synthetic customer data.
- **Format**: Tabular data with numeric, categorical, date, and text fields.
- **Preview & Profiling**: Missing values, data types, and basic statistics displayed to the user.

### 2. **Preprocessing Pipeline**

Handled via:

- Imputation of missing numeric values using medians.
- Encoding:
  - Label Encoding for simple categorical features.
  - One-Hot Encoding with filtering for high-cardinality features.
- Date transformation:
  - Extracts recency (`_days_ago`), month, and year.
- Text transformation:
  - Basic: word count and string length.
  - Advanced (if NLP enabled): sentiment analysis and TF-IDF features.

### 3. **Segmentation Engine**

Implemented using multiple clustering methods:

- **KMeans** (default)
- **Gaussian Mixture**
- **Agglomerative Clustering**
- **DBSCAN** (optional)

#### Cluster Optimization

- **Silhouette Score** and **Elbow Method** used to find optimal `k`.

#### Dimensionality Reduction

- **PCA (2D)** for visualization purposes.

### 4. **Feature Importance**

- Trained **Random Forest Classifier** to rank features by their contribution to segmentation.
- Mutual Information score is optionally combined to improve robustness.

### 5. **Predictive Models**

#### Churn Prediction

- Synthetic churn labels created using engagement recency (`days_since_last_visit`).
- Trained using **RandomForestClassifier**.

#### LTV Prediction

- Predicts customer LTV using **GradientBoostingRegressor** trained on monetary features.

### 6. **Segment Profile Creation**

Each segment includes:

- Statistical metrics (average, median per feature)
- Behavior analysis: engagement, recency, sentiment
- Predicted **LTV** and **churn risk**
- **Marketing recommendations** based on:
  - Segment engagement
  - Risk level
  - Value tier
- Human-readable segment name (e.g., “Engaged High-Value Segment”)

### 7. **Interactive Streamlit UI**

#### Tabs:

- 📊 **Data & Analysis**: Upload/generate data, configure analysis, run clustering
- 👥 **Segment Profiles**: Explore segment stats, characteristics, and recommendations
- 📈 **Visualizations**: Interactive Plotly charts
- 🎯 **Business Insights**: Recommendations and next actions

#### Visualizations:

- Segment size pie chart
- LTV vs Churn bubble chart
- PCA scatter plot
- Feature importance bars
- Segment comparison grid
- Customer journey patterns

---

## ⚙️ Configuration Options

| Option             | Description                             |
| ------------------ | --------------------------------------- |
| `n_clusters`       | User-defined or automatically detected  |
| `categorical_cols` | Manually selected for encoding          |
| `text_cols`        | Used for sentiment or textual patterns  |
| `date_cols`        | Used for recency analysis               |
| `color_scheme`     | Custom themes for Plotly visualizations |

---

## 🧠 Smart Recommendations Logic

Marketing recommendations are rule-based and include:

- High-value → Loyalty/VIP programs
- High-risk → Retention/win-back campaigns
- Low engagement → Tutorials, gamification
- Low value → Bundled offers, upsell tactics

---

## 🧪 Model Evaluation

- **Silhouette Score**: Clustering quality
- **Model Scores**: Churn and LTV model accuracy via train/test split
- **Segment LTV Distribution**: Used for customer value ranking

---

## 📤 Output

- **Segment Profiles**: JSON-like dictionaries per segment
- **Prediction API** (`predict_segment`) that returns:
  - Predicted segment
  - Risk scores
  - Top recommendations
  - Key feature values

---

## 📦 Dependencies

```bash
pip install streamlit pandas numpy scikit-learn plotly matplotlib textblob
```

> Optional NLP support: `textblob`, `TfidfVectorizer`

---

## 🚀 Running the App

```bash
streamlit run userSegmentationAnalyticsApp.py
```

---

## The Business Problem It Solves:

Imagine you're running an e-commerce store with 10,000 customers. Instead of treating everyone the same, this system helps you:

- Identify your most valuable customers (high LTV, low churn risk) → Focus retention efforts here
- Find customers about to leave (high churn risk) → Send them special offers
- Discover growth opportunities (stable customers with expansion potential)
- Optimize marketing spend by targeting the right segments with the right messages

**How It Works:**

Customer Data → Machine Learning → Segments → Predictions → Business Insights
↓ ↓ ↓ ↓ ↓
Purchase Algorithm Group 1: LTV: $500 "Focus on
History Groups VIP Users Churn: 5% retention"
Demographics Similar Group 2: LTV: $200 "Upsell
Behavior Customers Bargain Churn: 30% products"
Hunters

## Key Features:

    Interactive Charts:
        -Scatter plots showing LTV vs Churn Risk
        -Different visualization options
        -Color-coded segments
    Growth Opportunities Section:
        -Identifies low-risk, high-value segments for expansion
        -Calculates growth potential in dollars
        -Flags underperforming segments with development potential
    Marketing Strategy Tabs:
        -Segment-specific recommendations
        -Risk-value matrix positioning
        -Revenue contribution analysis

## Real-World Example:

**Let's say the system finds:**

    Segment 1: 500 customers, $800 LTV, 10% churn risk → "Premium loyalists - upsell premium products"
    Segment 2: 1200 customers, $200 LTV, 60% churn risk → "At-risk budget buyers - send discount offers"
    Segment 3: 800 customers, $400 LTV, 20% churn risk → "Growth potential - cross-sell complementary products"

This type of system is extremely valuable for businesses because it transforms raw customer data into actionable business insights, helping companies increase revenue, reduce churn, and optimize their marketing efforts.
