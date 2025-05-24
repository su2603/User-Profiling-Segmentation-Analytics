import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import json
import hashlib
import pickle
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union
import time
from pathlib import Path

# Import core ML libraries
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split

# Optional imports with fallbacks
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from textblob import TextBlob
    from sklearn.feature_extraction.text import TfidfVectorizer
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class UserSegment:
    """User segment profile with enhanced fields"""
    segment_id: int
    name: str
    size: int
    characteristics: Dict[str, Any]
    behavioral_patterns: Dict[str, Any]
    marketing_recommendations: List[str]
    ltv_prediction: float
    churn_risk: float
    feature_importance: Dict[str, float] = None  # Important features for this segment

class EnhancedSegmentationSystem:
    """Enhanced user segmentation system"""
    
    def __init__(self):
        self.preprocessor = None
        self.clustering_model = None
        self.churn_model = None
        self.ltv_model = None
        self.feature_names = []
        self.segment_profiles = []
        self.processed_data = None
        self.feature_importance = {}
    
    def run_analysis(self, df: pd.DataFrame, 
                   categorical_cols: List[str] = None,
                   text_cols: List[str] = None,
                   date_cols: List[str] = None) -> Dict[str, Any]:
        """Run enhanced segmentation analysis"""
        logger.info("Starting enhanced segmentation analysis...")
        
        results = {
            'status': 'processing',
            'timestamp': datetime.now().isoformat(),
            'data_shape': df.shape
        }
        
        try:
            # 1. Process features
            df_processed, feature_names, preprocessor = self._process_features(
                df, categorical_cols, text_cols, date_cols
            )
            
            # Store for later use
            self.preprocessor = preprocessor
            self.feature_names = feature_names
            
            # 2. Find optimal clustering
            clustering_results = self._find_best_clustering(df_processed)
            segment_labels = clustering_results['labels']
            
            # 3. Calculate feature importance
            self.feature_importance = self._calculate_feature_importance(
                df_processed, segment_labels, feature_names
            )
            
            # 4. Train predictive models
            self._train_predictive_models(df_processed, segment_labels, df)
            
            # 5. Create segment profiles
            self.segment_profiles = self._create_segment_profiles(
                df, segment_labels, feature_names
            )
            
            # 6. Save processed data with segments
            self.processed_data = df.copy()
            self.processed_data['segment'] = segment_labels
            
            # Prepare results summary
            results.update({
                'status': 'success',
                'n_segments': len(np.unique(segment_labels)),
                'clustering_algorithm': clustering_results['algorithm'],
                'clustering_quality': {
                    'silhouette_score': clustering_results.get('silhouette_score', 0)
                },
                'top_features': dict(sorted(
                    self.feature_importance.items(), key=lambda x: x[1], reverse=True
                )[:5]),
                'segment_profiles': [
                    {
                        'id': p.segment_id,
                        'name': p.name,
                        'size': p.size,
                        'ltv': p.ltv_prediction,
                        'churn_risk': p.churn_risk
                    }
                    for p in self.segment_profiles
                ]
            })
            
            logger.info(f"Analysis completed with {results['n_segments']} segments")
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            results.update({
                'status': 'failed',
                'error': str(e)
            })
            
        return results
    
    def _process_features(self, df, categorical_cols=None, text_cols=None, date_cols=None):
        """Process features for segmentation"""
        df_processed = df.copy()
        
        # Identify column types if not specified
        if categorical_cols is None:
            categorical_cols = list(df.select_dtypes(include=['object', 'category']).columns)
            # Exclude ID-like columns
            categorical_cols = [col for col in categorical_cols 
                               if not any(term in col.lower() 
                                        for term in ['id', 'uuid', 'email'])]
        
        # Identify numeric columns
        numeric_cols = list(df.select_dtypes(include=['number']).columns)
        # Exclude ID-like columns
        numeric_cols = [col for col in numeric_cols 
                      if not any(term in col.lower() 
                               for term in ['id', 'uuid', 'index'])]
        
        # 1. Process text features
        if text_cols and NLP_AVAILABLE:
            for col in text_cols:
                if col in df.columns:
                    # Add sentiment analysis
                    df_processed[f'{col}_sentiment'] = df[col].astype(str).apply(
                        lambda x: TextBlob(x).sentiment.polarity if x else 0
                    )
                    
                    # Add text length
                    df_processed[f'{col}_length'] = df[col].astype(str).apply(len)
        
        # 2. Process date features
        if date_cols:
            for col in date_cols:
                if col in df.columns:
                    # Convert to datetime
                    df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
                    
                    # Add recency (days since)
                    try:
                       df_processed[f'{col}_days_since'] = (
                                    datetime.now() - df_processed[col]
                        ).dt.days.fillna(0).astype(int)
                    except Exception as e:
                        logger.warning(f"Error processing date column {col}: {e}")
                        # Create dummy column to avoid errors
                        df_processed[f'{col}_days_since'] = 0
                    
                    # Add day of week
                    try:
                       df_processed[f'{col}_weekday'] = df_processed[col].dt.weekday.fillna(0).astype(int)
                    except:
                       df_processed[f'{col}_weekday'] = 0
                
                    # Add month
                    try:
                       df_processed[f'{col}_month'] = df_processed[col].dt.month.fillna(1).astype(int)
                    except:
                       df_processed[f'{col}_month'] = 1

                    # Remove the original date column to avoid conversion errors
                    df_processed = df_processed.drop(columns=[col])
        
        # 3. Process categorical features
        if categorical_cols:
            # Create one-hot encoding for categorical features
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            
            # Filter out columns with too many unique values
            valid_cat_cols = []
            for col in categorical_cols:
                if col in df.columns and df[col].nunique() < 20:  # Limit categories
                    valid_cat_cols.append(col)
            
            if valid_cat_cols:
                # Fit and transform
                encoded = encoder.fit_transform(df_processed[valid_cat_cols])
                
                # Create DataFrame with encoded values
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=[f"{col}_{cat}" for col, cats in 
                            zip(valid_cat_cols, encoder.categories_) 
                            for cat in cats]
                )
                
                # Concatenate with original dataframe
                df_processed = pd.concat([df_processed, encoded_df], axis=1)
        
        # 4. Prepare final feature set
        # Combine original numeric with engineered features
        feature_cols = numeric_cols.copy()
        
        # Add text features
        if text_cols and NLP_AVAILABLE:
            text_feature_cols = []
            for col in text_cols:
                text_feature_cols.extend([f'{col}_sentiment', f'{col}_length'])
            feature_cols.extend([col for col in text_feature_cols if col in df_processed.columns])
        
        # Add date features  
        if date_cols:
            date_feature_cols = []
            for col in date_cols:
                date_feature_cols.extend([f'{col}_days_since', f'{col}_weekday', f'{col}_month'])
            feature_cols.extend([col for col in date_feature_cols if col in df_processed.columns])
        
        # Add one-hot encoded columns
        if categorical_cols:
            encoded_cols = [col for col in df_processed.columns 
                          if any(f"{cat_col}_" in col for cat_col in valid_cat_cols)]
            feature_cols.extend(encoded_cols)
        
        # Make sure all columns exist
        feature_cols = [col for col in feature_cols if col in df_processed.columns]
        
        # Create feature matrix
        X = df_processed[feature_cols].values
        
        # Create preprocessing pipeline for scaling
        preprocessor = StandardScaler()
        X_processed = preprocessor.fit_transform(X)
        
        return X_processed, feature_cols, preprocessor
    
    def _find_best_clustering(self, X: np.ndarray) -> Dict[str, Any]:
        """Find the optimal clustering algorithm and parameters"""
        results = {}
        
        # Try different clustering algorithms
        algorithms = {
            'kmeans': {
                'clusters_range': range(2, min(11, X.shape[0] // 30 + 1))
            },
            'gaussian_mixture': {
                'clusters_range': range(2, min(11, X.shape[0] // 30 + 1))
            },
            'hierarchical': {
                'clusters_range': range(2, min(11, X.shape[0] // 30 + 1))
            }
        }
        
        best_score = -1
        best_algo = None
        best_model = None
        best_labels = None
        
        for algo_name, params in algorithms.items():
            for n_clusters in params['clusters_range']:
                try:
                    # Train clustering model
                    model = self._train_clustering_model(algo_name, n_clusters, X)
                    
                    # Get cluster labels
                    if hasattr(model, 'predict'):
                        labels = model.predict(X)
                    else:
                        labels = model.labels_
                    
                    # Check if we have more than one cluster
                    if len(np.unique(labels)) <= 1:
                        continue
                    
                    # Calculate silhouette score
                    score = silhouette_score(X, labels)
                    
                    if score > best_score:
                        best_score = score
                        best_algo = algo_name
                        best_model = model
                        best_labels = labels
                        
                except Exception as e:
                    logger.warning(f"Error with {algo_name}, {n_clusters} clusters: {e}")
                    continue
        
        # If no good clustering found, use simple KMeans
        if best_model is None:
            logger.warning("No optimal clustering found, using default KMeans with 3 clusters")
            best_model = KMeans(n_clusters=3, random_state=42)
            best_labels = best_model.fit_predict(X)
            best_algo = 'kmeans'
            best_score = -1
        
        # Store the best model
        self.clustering_model = best_model
        
        # Create result dictionary
        results = {
            'algorithm': best_algo,
            'model': best_model,
            'labels': best_labels,
            'silhouette_score': best_score,
            'n_clusters': len(np.unique(best_labels)),
            'cluster_sizes': dict(pd.Series(best_labels).value_counts().to_dict())
        }
        
        return results
    
    def _train_clustering_model(self, algorithm: str, n_clusters: int, X: np.ndarray):
        """Train a clustering model with given algorithm"""
        if algorithm == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            model.fit(X)
            
        elif algorithm == 'gaussian_mixture':
            model = GaussianMixture(n_components=n_clusters, random_state=42)
            model.fit(X)
            
        elif algorithm == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=n_clusters)
            model.fit(X)
            
        elif algorithm == 'dbscan':
            # Estimate epsilon using nearest neighbors
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=5).fit(X)
            distances, indices = nbrs.kneighbors(X)
            distances = np.sort(distances[:, 4], axis=0)  # 5th nearest neighbor
            
            # Use knee point for epsilon (simplified)
            epsilon = np.percentile(distances, 95)
            
            model = DBSCAN(eps=epsilon, min_samples=5)
            model.fit(X)
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return model
    
    def _calculate_feature_importance(self, X: np.ndarray, 
                                  labels: np.ndarray,
                                  feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance for segmentation"""
        importance = {}
        
        try:
            # Method 1: Train a classifier to predict segments
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X, labels)
            
            # Get feature importance from random forest
            for i, importance_value in enumerate(clf.feature_importances_):
                if i < len(feature_names):
                    importance[feature_names[i]] = float(importance_value)
            
            # Method 2: Use mutual information
            try:
                mi_scores = mutual_info_classif(X, labels)
                for i, mi_score in enumerate(mi_scores):
                    if i < len(feature_names):
                        # Average with random forest importance if exists
                        if feature_names[i] in importance:
                            importance[feature_names[i]] = (importance[feature_names[i]] + mi_score) / 2
                        else:
                            importance[feature_names[i]] = mi_score
            except:
                pass
                
        except Exception as e:
            logger.warning(f"Error calculating feature importance: {e}")
        
        # Normalize values
        if importance:
            total = sum(importance.values())
            if total > 0:
                importance = {k: v/total for k, v in importance.items()}
        
        return importance
    
    def _train_predictive_models(self, X: np.ndarray, 
                               labels: np.ndarray,
                               original_df: pd.DataFrame) -> None:
        """Train predictive models for churn and LTV"""
        # Train churn model if we have engagement metrics
        engagement_cols = [col for col in original_df.columns if any(
            term in col.lower() for term in ['visit', 'session', 'view', 'active', 'login']
        )]
        
        if engagement_cols:
            try:
                # Create synthetic churn labels for demonstration
                # In practice, you would use actual churn data
                if 'days_since_last_visit' in original_df.columns:
                    # Users with high "days since last" are considered churned
                    churn_threshold = np.percentile(original_df['days_since_last_visit'], 75)
                    churn_labels = (original_df['days_since_last_visit'] > churn_threshold).astype(int)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, churn_labels, test_size=0.2, random_state=42
                    )
                    
                    # Train model
                    self.churn_model = RandomForestClassifier(n_estimators=100, random_state=42)
                    self.churn_model.fit(X_train, y_train)
                    
                    logger.info(f"Churn model trained with score: {self.churn_model.score(X_test, y_test):.3f}")
            except Exception as e:
                logger.warning(f"Error training churn model: {e}")
        
        # Train LTV model if we have revenue metrics
        revenue_cols = [col for col in original_df.columns if any(
            term in col.lower() for term in ['revenue', 'spent', 'purchase', 'order']
        )]
        
        if revenue_cols:
            try:
                # Use the first revenue column as target
                target_col = revenue_cols[0]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, original_df[target_col], test_size=0.2, random_state=42
                )
                
                # Train model
                self.ltv_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                self.ltv_model.fit(X_train, y_train)
                
                logger.info(f"LTV model trained with score: {self.ltv_model.score(X_test, y_test):.3f}")
            except Exception as e:
                logger.warning(f"Error training LTV model: {e}")
    
    def _create_segment_profiles(self, df: pd.DataFrame, 
                               labels: np.ndarray,
                               feature_names: List[str]) -> List[UserSegment]:
        """Create comprehensive segment profiles"""
        profiles = []
        
        for segment_id in np.unique(labels):
            # Get data for this segment
            segment_data = df[labels == segment_id]
            
            # Basic characteristics
            characteristics = {
                'size': len(segment_data),
                'size_percent': len(segment_data) / len(df) * 100
            }
            
            # Add each numeric column's average
            for col in df.select_dtypes(include=['number']).columns:
                if not any(term in col.lower() for term in ['id', 'index', 'segment']):
                    characteristics[f'avg_{col}'] = float(segment_data[col].mean())
            
            # Add top categorical values
            for col in df.select_dtypes(include=['object', 'category']).columns:
                if not any(term in col.lower() for term in ['id', 'email', 'name']):
                    try:
                        top_values = segment_data[col].value_counts(normalize=True).nlargest(3).to_dict()
                        characteristics[f'top_{col}'] = {str(k): float(v) for k, v in top_values.items()}
                    except:
                        pass
            
            # Behavioral patterns
            behavioral_patterns = self._extract_behavioral_patterns(segment_data)
            
            # Calculate segment-specific feature importance
            segment_feature_importance = {}
            for feature, global_importance in self.feature_importance.items():
                # Find the feature in the original data if possible
                if feature in df.columns:
                    # Compare segment mean to overall mean
                    segment_mean = segment_data[feature].mean()
                    overall_mean = df[feature].mean()
                    overall_std = df[feature].std()
                    
                    # Importance is higher if segment differs from overall
                    if overall_std > 0:
                        # Z-score difference
                        z_diff = abs(segment_mean - overall_mean) / overall_std
                        segment_feature_importance[feature] = global_importance * (1 + z_diff)
                    else:
                        segment_feature_importance[feature] = global_importance
            
            # Normalize importances
            if segment_feature_importance:
                total = sum(segment_feature_importance.values())
                if total > 0:
                    segment_feature_importance = {
                        k: v/total for k, v in segment_feature_importance.items()
                    }
            
            # Generate segment name
            segment_name = self._generate_segment_name(
                segment_id, characteristics, behavioral_patterns
            )
            
            # Predict LTV
            if self.ltv_model is not None:
                try:
                    # Extract features for this segment
                    X_segment = self.preprocessor.transform(
                        segment_data[self.feature_names].values
                    )
                    ltv = float(self.ltv_model.predict(X_segment).mean())
                except:
                    # Fallback to simple calculation
                    ltv = self._estimate_ltv(segment_data)
            else:
                ltv = self._estimate_ltv(segment_data)
            
            # Estimate churn risk
            if self.churn_model is not None:
                try:
                    # Extract features for this segment
                    X_segment = self.preprocessor.transform(
                        segment_data[self.feature_names].values
                    )
                    churn_proba = self.churn_model.predict_proba(X_segment)[:, 1]
                    churn_risk = float(churn_proba.mean())
                except:
                    # Fallback to simple calculation
                    churn_risk = self._estimate_churn_risk(segment_data)
            else:
                churn_risk = self._estimate_churn_risk(segment_data)
            
            # Generate marketing recommendations
            recommendations = self._generate_recommendations(
                segment_data, characteristics, behavioral_patterns, churn_risk, ltv
            )
            
            # Create profile
            profile = UserSegment(
                segment_id=int(segment_id),
                name=segment_name,
                size=len(segment_data),
                characteristics=characteristics,
                behavioral_patterns=behavioral_patterns,
                marketing_recommendations=recommendations,
                ltv_prediction=ltv,
                churn_risk=churn_risk,
                feature_importance=segment_feature_importance
            )
            
            profiles.append(profile)
        
        return profiles
    
    def _extract_behavioral_patterns(self, segment_data: pd.DataFrame) -> Dict[str, Any]:
        """Extract behavioral patterns from segment data"""
        patterns = {}
        
        # Engagement metrics
        engagement_cols = [col for col in segment_data.columns if any(
            term in col.lower() for term in ['view', 'session', 'duration', 'click', 'open']
        )]
        
        if engagement_cols:
            # Example: Use session_duration as engagement indicator
            if 'session_duration' in segment_data.columns:
                patterns['avg_session_duration'] = float(segment_data['session_duration'].mean())
                patterns['engagement_score'] = min(1.0, segment_data['session_duration'].mean() / 600)
        
        # Recency metrics
        if 'days_since_last_visit' in segment_data.columns:
            patterns['avg_recency'] = float(segment_data['days_since_last_visit'].mean())
            patterns['recency_score'] = max(0, 1 - patterns['avg_recency'] / 30)
        
        # Purchase patterns
        if 'purchases_count' in segment_data.columns:
            patterns['avg_purchases'] = float(segment_data['purchases_count'].mean())
            
            if 'total_spent' in segment_data.columns and segment_data['purchases_count'].sum() > 0:
                patterns['avg_order_value'] = float(
                    segment_data['total_spent'].sum() / segment_data['purchases_count'].sum()
                )
        
        # Sentiment patterns (if available)
        sentiment_cols = [col for col in segment_data.columns if 'sentiment' in col.lower()]
        if sentiment_cols:
            patterns['avg_sentiment'] = float(segment_data[sentiment_cols[0]].mean())
        
        # Calculate overall value score
        value_factors = []
        if 'total_spent' in segment_data.columns:
            value_factors.append(min(1.0, segment_data['total_spent'].mean() / 1000))
        
        if value_factors:
            patterns['value_score'] = float(np.mean(value_factors))
        else:
            patterns['value_score'] = 0.5  # Default
        
        return patterns
    
    def _generate_segment_name(self, segment_id: int, 
                            characteristics: Dict[str, Any],
                            behavioral: Dict[str, Any]) -> str:
        """Generate descriptive name for segment"""
        # Identify key characteristics to include in name
        descriptors = []
        
        # Value-based descriptors
        if 'avg_total_spent' in characteristics:
            total_spent = characteristics['avg_total_spent']
            if total_spent > 800:
                descriptors.append("High-Value")
            elif total_spent < 200:
                descriptors.append("Budget-Conscious")
        
        # Engagement-based descriptors
        if 'engagement_score' in behavioral:
            engagement = behavioral['engagement_score']
            if engagement > 0.7:
                descriptors.append("Engaged")
            elif engagement < 0.3:
                descriptors.append("Passive")
        
        # Recency-based descriptors
        if 'avg_recency' in behavioral:
            recency = behavioral['avg_recency']
            if recency < 7:
                descriptors.append("Active")
            elif recency > 30:
                descriptors.append("Dormant")
        
        # Age-based descriptors (if available)
        if 'avg_age' in characteristics:
            age = characteristics['avg_age']
            if age < 30:
                descriptors.append("Young")
            elif age > 55:
                descriptors.append("Mature")
        
        # Create name
        if descriptors:
            return f"{' '.join(descriptors[:2])} Segment {segment_id}"
        else:
            return f"Segment {segment_id}"
    
    def _estimate_ltv(self, segment_data: pd.DataFrame) -> float:
        """Estimate lifetime value for a segment"""
        if 'total_spent' in segment_data.columns:
            avg_spent = segment_data['total_spent'].mean()
            
            # Apply multipliers based on behavioral patterns
            multiplier = 3.0  # Base multiplier
            
            # Higher multiplier for engaged customers
            if 'session_duration' in segment_data.columns:
                engagement = min(1.0, segment_data['session_duration'].mean() / 600) 
                multiplier += engagement * 2
            
            # Lower multiplier for dormant customers
            if 'days_since_last_visit' in segment_data.columns:
                recency = min(1.0, segment_data['days_since_last_visit'].mean() / 60)
                multiplier -= recency * 2
            
            return float(avg_spent * max(1, multiplier))
        else:
            return 100.0  # Default value
    
    def _estimate_churn_risk(self, segment_data: pd.DataFrame) -> float:
        """Estimate churn risk for a segment"""
        risk_factors = []
        
        # Recency factor
        if 'days_since_last_visit' in segment_data.columns:
            recency = segment_data['days_since_last_visit'].mean()
            risk_factors.append(min(1.0, recency / 60))  # Cap at 60 days
        
        # Engagement factor (inverse)
        if 'session_duration' in segment_data.columns:
            engagement = segment_data['session_duration'].mean()
            risk_factors.append(max(0, 1.0 - (engagement / 600)))
        
        # Sentiment factor (if available)
        sentiment_cols = [col for col in segment_data.columns if 'sentiment' in col.lower()]
        if sentiment_cols:
            sentiment = segment_data[sentiment_cols[0]].mean()
            # Negative sentiment increases churn risk
            risk_factors.append(max(0, 0.5 - sentiment))
        
        # If we have factors, average them; otherwise return moderate risk
        if risk_factors:
            return float(np.mean(risk_factors))
        else:
            return 0.5
    
    def _generate_recommendations(self, segment_data: pd.DataFrame,
                               characteristics: Dict[str, Any],
                               behavioral_patterns: Dict[str, Any],
                               churn_risk: float,
                               ltv: float) -> List[str]:
        """Generate marketing recommendations based on segment profile"""
        recommendations = []
        
        # High-value customer recommendations
        if ltv > 800:
            recommendations.extend([
                "Implement VIP loyalty program",
                "Offer exclusive early access to new products/features",
                "Develop personalized concierge service"
            ])
        
        # Churn prevention recommendations
        if churn_risk > 0.6:
            recommendations.extend([
                "Launch targeted win-back campaign",
                "Offer special retention discounts",
                "Create re-engagement email sequence"
            ])
        
        # Engagement recommendations
        if behavioral_patterns.get('engagement_score', 0) < 0.4:
            recommendations.extend([
                "Create interactive product tutorials",
                "Implement gamification elements",
                "Send regular feature highlight emails"
            ])
        
        # Value growth recommendations
        if ltv < 400 and behavioral_patterns.get('engagement_score', 0) > 0.5:
            recommendations.extend([
                "Create bundle offers for complementary products",
                "Implement strategic upselling at key moments",
                "Develop tiered pricing model"
            ])
        
        # Default recommendations
        if not recommendations:
            recommendations = [
                "Send regular newsletters with personalized content",
                "Implement seasonal promotional campaigns",
                "Conduct customer satisfaction surveys"
            ]
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def predict_segment(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict segment for a new user"""
        response = {
            'segment': 0,
            'risk_scores': {
                'churn_risk': 0.5,
                'value_risk': 0.5
            }
        }
        
        try:
            # Check if we have a trained model
            if self.clustering_model is None:
                response['error'] = "No trained model available"
                return response
                
            # Extract features matching our training features
            features = self._extract_prediction_features(user_data)
            
            # Scale features
            if self.preprocessor:
                try:
                    features_scaled = self.preprocessor.transform(features)
                except Exception as e:
                    logger.warning(f"Error scaling features: {e}")
                    # Use unscaled features as fallback
                    features_scaled = features
            else:
                features_scaled = features
            
            # Predict segment
            if hasattr(self.clustering_model, 'predict'):
                segment = int(self.clustering_model.predict(features_scaled)[0])
            else:
                # For models without predict method
                segment = 0
                logger.warning("Model doesn't support prediction")
                
            response['segment'] = segment
            
            # Add segment profile information
            for profile in self.segment_profiles:
                if profile.segment_id == segment:
                    response['segment_name'] = profile.name
                    response['recommendations'] = profile.marketing_recommendations[:3]
                    
                    # Add key characteristics
                    response['characteristics'] = {
                        k: v for k, v in profile.characteristics.items()
                        if k in ['avg_total_spent', 'avg_session_duration', 'avg_purchases_count'][:3]
                    }
                    break
            
            # Calculate risk scores
            churn_risk = self._calculate_user_churn_risk(user_data)
            value_risk = self._calculate_user_value_risk(user_data)
            
            response['risk_scores'] = {
                'churn_risk': churn_risk,
                'value_risk': value_risk
            }
            
            # Predict LTV if model available
            if self.ltv_model is not None:
                try:
                    ltv = float(self.ltv_model.predict(features_scaled)[0])
                    response['ltv_prediction'] = ltv
                except Exception as e:
                    logger.warning(f"Error predicting LTV: {e}")
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            response['error'] = str(e)
        
        return response
    
    def _extract_prediction_features(self, user_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from user data for prediction"""
        # Initialize with zeros to handle missing features
        features = np.zeros((1, len(self.feature_names)))
        
        try:
            # Extract known features using feature names
            for i, feature in enumerate(self.feature_names):
                # Handle basic numeric features
                if feature in user_data:
                    try:
                        value = user_data[feature]
                        # Handle different data types
                        if isinstance(value, (int, float)):
                            features[0, i] = float(value)
                        elif isinstance(value, str) and value.replace('.', '').isdigit():
                            features[0, i] = float(value)
                    except:
                        pass  # Keep as zero if conversion fails
                
                # Handle feature patterns like 'col_sentiment'
                elif '_sentiment' in feature and feature.replace('_sentiment', '') in user_data:
                    base_col = feature.replace('_sentiment', '')
                    if isinstance(user_data[base_col], str) and NLP_AVAILABLE:
                        try:
                            features[0, i] = TextBlob(user_data[base_col]).sentiment.polarity
                        except:
                            pass
                
                # Handle days_since features
                elif '_days_since' in feature and feature.replace('_days_since', '') in user_data:
                    base_col = feature.replace('_days_since', '')
                    try:
                        date_val = pd.to_datetime(user_data[base_col])
                        features[0, i] = (datetime.now() - date_val).days
                    except:
                        pass
                
                # Handle categorical one-hot features
                for col in user_data:
                    if feature.startswith(f"{col}_") and user_data[col] in feature:
                        features[0, i] = 1.0
                        break
                
        except Exception as e:
            logger.warning(f"Error extracting prediction features: {e}")
            # On error, return a simple array of the correct size
            return np.zeros((1, len(self.feature_names)))
        
        return features
    
    def _calculate_user_churn_risk(self, user_data: Dict[str, Any]) -> float:
        """Calculate churn risk for a single user"""
        try:
            # Use churn model if available
            if self.churn_model is not None:
                features = self._extract_prediction_features(user_data)
                if self.preprocessor:
                    features = self.preprocessor.transform(features)
                return float(self.churn_model.predict_proba(features)[0, 1])
            
            # Manual calculation
            risk_factors = []
            
            # Recency factor
            if 'days_since_last_visit' in user_data:
                days_val = user_data['days_since_last_visit']
                if isinstance(days_val, (int, float)):
                    days = float(days_val)
                    risk_factors.append(min(1.0, days / 30))
            
            # Engagement factor
            if 'session_duration' in user_data:
                duration_val = user_data['session_duration']
                if isinstance(duration_val, (int, float)):
                    duration = float(duration_val)
                    risk_factors.append(max(0, 1.0 - (duration / 600)))
            
            if risk_factors:
                return float(np.mean(risk_factors))
            else:
                return 0.5  # Default moderate risk
                
        except Exception as e:
            logger.warning(f"Error calculating churn risk: {e}")
            return 0.5
    
    def _calculate_user_value_risk(self, user_data: Dict[str, Any]) -> float:
        """Calculate value risk for a single user"""
        try:
            # Value risk based on spending patterns
            if 'total_spent' in user_data and 'purchases_count' in user_data:
                total_spent_val = user_data['total_spent']
                purchases_val = user_data['purchases_count']
                
                if isinstance(total_spent_val, (int, float)) and isinstance(purchases_val, (int, float)):
                    total_spent = float(total_spent_val)
                    purchases = float(purchases_val)
                    
                    if purchases > 0:
                        avg_order = total_spent / purchases
                        return max(0.0, 1.0 - (avg_order / 500))
            
            return 0.6  # Default moderate-high value risk
                
        except Exception as e:
            logger.warning(f"Error calculating value risk: {e}")
            return 0.5
    
    def create_visualizations(self) -> Dict[str, Any]:
        """Create visualizations for segment analysis"""
        if not PLOTLY_AVAILABLE:
            return {"error": "Plotly not available for visualizations"}
            
        if not self.segment_profiles or self.processed_data is None:
            return {"error": "No segmentation results available"}
        
        visualizations = {}
        
        try:
            # 1. Segment distribution
            segment_sizes = self.processed_data['segment'].value_counts().sort_index()
            
            fig_distribution = go.Figure(data=[
                go.Pie(
                    labels=[f"Segment {i}" for i in segment_sizes.index],
                    values=segment_sizes.values,
                    hole=0.3
                )
            ])
            fig_distribution.update_layout(title_text="Segment Distribution")
            
            visualizations['segment_distribution'] = fig_distribution
            
            # 2. Key metrics by segment
            metrics = []
            
            if 'total_spent' in self.processed_data.columns:
                metrics.append('total_spent')
                
            if 'session_duration' in self.processed_data.columns:
                metrics.append('session_duration')
                
            if 'purchases_count' in self.processed_data.columns:
                metrics.append('purchases_count')
            
            if metrics:
                fig_metrics = go.Figure()
                
                for metric in metrics[:3]:  # Limit to 3 metrics
                    metric_by_segment = self.processed_data.groupby('segment')[metric].mean()
                    fig_metrics.add_trace(
                        go.Bar(
                            x=[f"Segment {i}" for i in metric_by_segment.index],
                            y=metric_by_segment.values,
                            name=metric.replace('_', ' ').title()
                        )
                    )
                    
                fig_metrics.update_layout(
                    title_text="Key Metrics by Segment",
                    barmode='group'
                )
                
                visualizations['metrics_by_segment'] = fig_metrics
            
            # 3. Feature importance
            if self.feature_importance:
                top_features = dict(sorted(
                    self.feature_importance.items(), key=lambda x: x[1], reverse=True
                )[:10])
                
                fig_importance = go.Figure([
                    go.Bar(
                        x=list(top_features.keys()),
                        y=list(top_features.values()),
                        marker_color='indianred'
                    )
                ])
                
                fig_importance.update_layout(
                    title_text="Top Feature Importance",
                    xaxis_tickangle=-45
                )
                
                visualizations['feature_importance'] = fig_importance
            
            # 4. Segment comparison radar
            if len(self.segment_profiles) > 0:
                # Select metrics for radar chart
                radar_metrics = ['churn_risk', 'ltv_prediction']
                
                # Add behavioral patterns
                for pattern in ['engagement_score', 'value_score', 'recency_score']:
                    for profile in self.segment_profiles:
                        if pattern in profile.behavioral_patterns:
                            radar_metrics.append(pattern)
                            break
                
                # Remove duplicates while preserving order
                radar_metrics = list(dict.fromkeys(radar_metrics))
                
                fig_radar = go.Figure()
                
                for profile in self.segment_profiles:
                    values = []
                    
                    for metric in radar_metrics:
                        if metric == 'churn_risk':
                            values.append(profile.churn_risk)
                        elif metric == 'ltv_prediction':
                            # Normalize LTV to 0-1 range
                            max_ltv = max(p.ltv_prediction for p in self.segment_profiles)
                            values.append(profile.ltv_prediction / max_ltv if max_ltv > 0 else 0)
                        else:
                            values.append(profile.behavioral_patterns.get(metric, 0))
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=radar_metrics,
                        fill='toself',
                        name=profile.name
                    ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    title="Segment Comparison"
                )
                
                visualizations['segment_comparison'] = fig_radar
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    def save_models(self, path: str = './models'):
        """Save trained models and profiles to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            
            # Save clustering model
            if self.clustering_model is not None:
                with open(os.path.join(path, 'clustering_model.pkl'), 'wb') as f:
                    pickle.dump(self.clustering_model, f)
            
            # Save preprocessor
            if self.preprocessor is not None:
                with open(os.path.join(path, 'preprocessor.pkl'), 'wb') as f:
                    pickle.dump(self.preprocessor, f)
            
            # Save predictive models
            if self.churn_model is not None:
                with open(os.path.join(path, 'churn_model.pkl'), 'wb') as f:
                    pickle.dump(self.churn_model, f)
                    
            if self.ltv_model is not None:
                with open(os.path.join(path, 'ltv_model.pkl'), 'wb') as f:
                    pickle.dump(self.ltv_model, f)
            
            # Save feature names
            with open(os.path.join(path, 'feature_names.json'), 'w') as f:
                json.dump(self.feature_names, f)
            
            # Save feature importance
            with open(os.path.join(path, 'feature_importance.json'), 'w') as f:
                json.dump(self.feature_importance, f)
            
            # Save segment profiles as JSON
            segment_profiles_json = []
            for profile in self.segment_profiles:
                profile_dict = {
                    'segment_id': profile.segment_id,
                    'name': profile.name,
                    'size': profile.size,
                    'characteristics': profile.characteristics,
                    'behavioral_patterns': profile.behavioral_patterns,
                    'marketing_recommendations': profile.marketing_recommendations,
                    'ltv_prediction': profile.ltv_prediction,
                    'churn_risk': profile.churn_risk,
                    'feature_importance': profile.feature_importance or {}
                }
                segment_profiles_json.append(profile_dict)
            
            with open(os.path.join(path, 'segment_profiles.json'), 'w') as f:
                json.dump(segment_profiles_json, f)
                
            logger.info(f"Models and profiles saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def load_models(self, path: str = './models'):
        """Load trained models and profiles from disk"""
        try:
            # Check if directory exists
            if not os.path.exists(path):
                logger.error(f"Path does not exist: {path}")
                return False
                
            # Load clustering model
            clustering_path = os.path.join(path, 'clustering_model.pkl')
            if os.path.exists(clustering_path):
                with open(clustering_path, 'rb') as f:
                    self.clustering_model = pickle.load(f)
            
            # Load preprocessor
            preprocessor_path = os.path.join(path, 'preprocessor.pkl')
            if os.path.exists(preprocessor_path):
                with open(preprocessor_path, 'rb') as f:
                    self.preprocessor = pickle.load(f)
            
            # Load predictive models
            churn_path = os.path.join(path, 'churn_model.pkl')
            if os.path.exists(churn_path):
                with open(churn_path, 'rb') as f:
                    self.churn_model = pickle.load(f)
                    
            ltv_path = os.path.join(path, 'ltv_model.pkl')
            if os.path.exists(ltv_path):
                with open(ltv_path, 'rb') as f:
                    self.ltv_model = pickle.load(f)
            
            # Load feature names
            feature_names_path = os.path.join(path, 'feature_names.json')
            if os.path.exists(feature_names_path):
                with open(feature_names_path, 'r') as f:
                    self.feature_names = json.load(f)
            
            # Load feature importance
            importance_path = os.path.join(path, 'feature_importance.json')
            if os.path.exists(importance_path):
                with open(importance_path, 'r') as f:
                    self.feature_importance = json.load(f)
            
            # Load segment profiles
            profiles_path = os.path.join(path, 'segment_profiles.json')
            if os.path.exists(profiles_path):
                with open(profiles_path, 'r') as f:
                    profiles_data = json.load(f)
                    
                self.segment_profiles = []
                for profile_data in profiles_data:
                    profile = UserSegment(
                        segment_id=profile_data['segment_id'],
                        name=profile_data['name'],
                        size=profile_data['size'],
                        characteristics=profile_data['characteristics'],
                        behavioral_patterns=profile_data['behavioral_patterns'],
                        marketing_recommendations=profile_data['marketing_recommendations'],
                        ltv_prediction=profile_data['ltv_prediction'],
                        churn_risk=profile_data['churn_risk'],
                        feature_importance=profile_data.get('feature_importance', {})
                    )
                    self.segment_profiles.append(profile)
                    
            logger.info(f"Models and profiles loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample user data for testing"""
    np.random.seed(42)
    
    # Generate user base data
    data = {
        'user_id': [f"user_{i:04d}" for i in range(n_samples)],
        'age': np.random.normal(35, 12, n_samples).clip(18, 80).astype(int),
        'gender': np.random.choice(['M', 'F', 'Other'], n_samples, p=[0.45, 0.45, 0.1]),
        'location': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'], n_samples),
        'income': np.random.lognormal(10.5, 0.5, n_samples).astype(int),
        'purchases_count': np.random.poisson(5, n_samples),
        'total_spent': np.random.lognormal(6, 1, n_samples),
        'session_duration': np.random.exponential(300, n_samples),
        'page_views': np.random.poisson(10, n_samples),
        'days_since_last_visit': np.random.exponential(7, n_samples),
        'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], n_samples, p=[0.6, 0.3, 0.1]),
        'subscription_type': np.random.choice(['free', 'premium', 'enterprise'], n_samples, p=[0.7, 0.25, 0.05]),
    }
    
    # Add dates
    data['signup_date'] = pd.date_range('2023-01-01', periods=n_samples, freq='h')[:n_samples]
    
    # Add review text
    if NLP_AVAILABLE:
        # Generate simple review texts
        sentiments = ['I love this product!', 
                    'Very good service', 
                    'It was okay, could be better', 
                    'Disappointed with quality',
                    'Excellent support team']
        
        data['review_text'] = np.random.choice(sentiments, n_samples)
    
    # Add correlations to make segments more realistic
    df = pd.DataFrame(data)
    
    # Premium users spend more
    mask_premium = df['subscription_type'] == 'premium'
    df.loc[mask_premium, 'total_spent'] *= 2.5
    df.loc[mask_premium, 'purchases_count'] *= 1.5
    
    # Enterprise users have longer sessions
    mask_enterprise = df['subscription_type'] == 'enterprise'
    df.loc[mask_enterprise, 'session_duration'] *= 2
    df.loc[mask_enterprise, 'page_views'] *= 1.8
    
    # Mobile users have shorter sessions
    mask_mobile = df['device_type'] == 'mobile'
    df.loc[mask_mobile, 'session_duration'] *= 0.7
    
    # Users with higher income spend more
    high_income_mask = df['income'] > df['income'].median()
    df.loc[high_income_mask, 'total_spent'] *= 1.4
    
    # Add domain-specific features
    
    # E-commerce features
    df['cart_abandonment_rate'] = np.random.uniform(0, 1, n_samples)
    df['days_since_last_purchase'] = np.random.exponential(14, n_samples)
    
    # SaaS features 
    df['feature_usage_score'] = np.random.uniform(0, 10, n_samples)
    df['subscription_tenure_months'] = np.random.geometric(p=0.1, size=n_samples)
    
    # Content engagement
    df['articles_read'] = np.random.poisson(3, n_samples)
    df['videos_watched'] = np.random.poisson(2, n_samples)
    
    return df

def run_demo():
    """Run demonstration with sample data"""
    print("Creating sample data...")
    df = create_sample_data(1000)
    
    print("Initializing enhanced segmentation system...")
    system = EnhancedSegmentationSystem()
    
    print("Running analysis...")
    results = system.run_analysis(
        df=df,
        categorical_cols=['gender', 'location', 'device_type', 'subscription_type'],
        text_cols=['review_text'] if 'review_text' in df.columns else None,
        date_cols=['signup_date']
    )
    
    print(f"Analysis complete: {results['status']}")
    
    if results['status'] == 'success':
        print(f"Found {results['n_segments']} segments using {results['clustering_algorithm']}")
        
        print("\nSegment Profiles:")
        for profile in results['segment_profiles']:
            print(f"- {profile['name']}: {profile['size']} users, "
                 f"LTV: ${profile['ltv']:.2f}, "
                 f"Churn Risk: {profile['churn_risk']:.2%}")
        
        print("\nTop Features:")
        for feature, importance in list(results['top_features'].items())[:5]:
            print(f"- {feature}: {importance:.4f}")
        
        print("\nCreating visualizations...")
        if PLOTLY_AVAILABLE:
            visualizations = system.create_visualizations()
            print(f"Created {len(visualizations)} visualizations")
            
            # Show the first visualization
            first_viz = next(iter(visualizations.values()))
            print("Visualization available (not showing in terminal)")
        else:
            print("Plotly not available - visualizations skipped")
        
        print("\nTesting real-time prediction...")
        sample_user = {
            'age': 30,
            'income': 50000,
            'purchases_count': 3,
            'total_spent': 500,
            'session_duration': 250,
            'page_views': 8,
            'days_since_last_visit': 5,
            'gender': 'F',
            'location': 'NYC',
            'device_type': 'mobile',
            'subscription_type': 'free',
            'articles_read': 4,
            'videos_watched': 2,
            'cart_abandonment_rate': 0.2
        }
        
        prediction = system.predict_segment(sample_user)
        print(f"User assigned to segment: {prediction.get('segment', 'unknown')}")
        if 'segment_name' in prediction:
            print(f"Segment name: {prediction['segment_name']}")
        print(f"Risk scores: {prediction.get('risk_scores', {})}")
        if 'recommendations' in prediction:
            print("Top recommendations:")
            for rec in prediction['recommendations']:
                print(f"- {rec}")
        
        print("\nSaving models and profiles...")
        system.save_models('./segmentation_models')
    
    return system, results

if __name__ == "__main__":
    system, results = run_demo()
    print("\nDemo completed successfully!")