# app.py
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os

# Set page configuration
st.set_page_config(
    page_title="Property Recommendation System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# LOAD MODEL
@st.cache_data
def load_data():
    """Load all necessary data files"""
    
    # Load the model
    model = xgb.XGBRanker()
    model_path = "../model/xgb_ranking_model.ubj"
    
    # Check if model file exists
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    
    model.load_model(model_path)
    
    # Load datasets
    subjects_path = "../data/cleaned/subjects_cleaned.csv"
    candidates_path = "../data/model_ready/candidates_pair_model_ready.csv"
    recommendations_path = "../data/results/top3_candidates_recommendations.csv"
    
    # Check if data files exist
    for path, name in [(subjects_path, 'Subjects'), (candidates_path, 'Candidates'), (recommendations_path, 'Recommendations')]:
        if not os.path.exists(path):
            st.error(f"{name} file not found at: {path}")
            st.stop()
    
    subjects_df = pd.read_csv(subjects_path)
    candidates_df = pd.read_csv(candidates_path)
    recommendations_df = pd.read_csv(recommendations_path)
    
    return model, subjects_df, candidates_df, recommendations_df

# Feature columns used in the model
feature_columns = [
    'gla_diff', 'lot_size_diff', 'bedroom_diff', 'bathroom_diff',
    'room_count_diff', 'same_property_type', 'same_storey_type',
    'sold_recently_90'
]

# SHAP
@st.cache_resource
def create_shap_explainer(_model, sample_data):
    """Create SHAP explainer for the model"""
    # Convert XGBoost booster to SKLearn-like interface for SHAP
    explainer = shap.TreeExplainer(_model)
    return explainer

def get_shap_explanation(model, property_features, explainer):
    """Generate SHAP values for a single property"""
    # Calculate SHAP values
    shap_values = explainer.shap_values(property_features)
    return shap_values

def create_feature_importance_plot(shap_values, features_df, feature_names):
    """Create an interactive feature importance plot"""
    # Get mean absolute SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    # Create dataframe for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_shap
    }).sort_values('Importance', ascending=True)
    
    # Create horizontal bar plot
    fig = px.bar(
        importance_df, 
        x='Importance', 
        y='Feature',
        orientation='h',
        title='Feature Importance for Property Selection',
        color='Importance',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Mean |SHAP value|",
        yaxis_title="Features"
    )
    
    return fig

# CHART
def create_property_comparison_chart(subject_data, recommended_properties):
    """Create a radar chart comparing properties"""
    
    categories = ['GLA', 'Bedrooms', 'Bathrooms', 'Lot Size', 'Room Count']
    
    fig = go.Figure()
    
    # Add subject property
    fig.add_trace(go.Scatterpolar(
        r=[100, 100, 100, 100, 100],  # Normalized to 100 for subject
        theta=categories,
        fill='toself',
        name='Subject Property',
        line_color='red'
    ))
    
    # Add recommended properties
    colors = ['blue', 'green', 'orange']
    for idx, (_, prop) in enumerate(recommended_properties.iterrows()):
        if idx < 3:  # Top 3 only
            # Calculate relative values (as percentage of subject)
            values = [
                (prop['gla_clean'] / subject_data['gla_clean'] * 100) if subject_data['gla_clean'] > 0 else 100,
                (prop['bedrooms_clean'] / subject_data['bedrooms_clean'] * 100) if subject_data['bedrooms_clean'] > 0 else 100,
                (prop['bathrooms_clean'] / subject_data['bathrooms_clean'] * 100) if subject_data['bathrooms_clean'] > 0 else 100,
                (prop['lot_size_clean'] / subject_data['lot_size_clean'] * 100) if subject_data['lot_size_clean'] > 0 else 100,
                (prop['room_count_clean'] / subject_data['room_count_clean'] * 100) if subject_data['room_count_clean'] > 0 else 100
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=f'Recommendation #{idx+1}',
                line_color=colors[idx],
                opacity=0.6
            ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 150]
            )),
        showlegend=True,
        title="Property Comparison (Subject = 100%)"
    )
    
    return fig  # <-- Fixed: Added fig

# NLP 
def generate_explanation(property_rec, shap_df, subject_data):
    """Generate a natural language explanation for why a property was selected"""
    
    # Get top 3 most impactful features
    top_features = shap_df.head(3)
    
    explanation = f"This property was ranked #{property_rec['rank']} because:\n\n"
    
    # Analyze each important feature
    for _, feature in top_features.iterrows():
        feature_name = feature['Feature'].replace('_', ' ').title()
        
        if 'diff' in feature['Feature']:
            # Handle difference features
            if abs(feature['Value']) < 100:
                explanation += f"• The {feature_name.replace(' Diff', '')} is very similar to the subject property "
                explanation += f"(difference of only {abs(feature['Value']):.1f})\n"
            else:
                explanation += f"• The {feature_name.replace(' Diff', '')} differs by {abs(feature['Value']):.1f}, "
                explanation += f"which {'positively' if feature['SHAP Impact'] > 0 else 'negatively'} impacts the match\n"
        
        elif feature['Feature'] == 'same_property_type':
            if feature['Value'] == 1:
                explanation += f"• It's the same property type as the subject ({subject_data['property_type_clean']})\n"
            else:
                explanation += f"• It's a different property type, which reduces the match score\n"
        
        elif feature['Feature'] == 'same_storey_type':
            if feature['Value'] == 1:
                explanation += f"• It has the same story type as the subject ({subject_data['stories_clean']})\n"
            else:
                explanation += f"• It has a different story type, affecting comparability\n"
        
        elif feature['Feature'] == 'sold_recently_90':
            if feature['Value'] == 1:
                explanation += f"• It was sold within the last 90 days, providing current market value\n"
            else:
                explanation += f"• It wasn't sold recently, which may affect price relevance\n"
    
    # Add overall match score interpretation
    if property_rec['pred'] > 0.5:
        explanation += f"\nOverall, this is an excellent comparable with a high match score of {property_rec['pred']:.3f}."
    elif property_rec['pred'] > 0.3:
        explanation += f"\nThis is a good comparable with a moderate match score of {property_rec['pred']:.3f}."
    else:
        explanation += f"\nThis property has some similarities but a lower match score of {property_rec['pred']:.3f}."
    
    return explanation

# APP 
def main():
    # Title
    st.markdown('<h1 class="main-header">🏠 Property Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Comparable Property Analysis with Explainability</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading model and data...'):
        model, subjects_df, candidates_df, recommendations_df = load_data()
    
    # Sidebar for property selection
    with st.sidebar:
        st.header("Select Subject Property")
        
        # Get unique order IDs
        order_ids = sorted(subjects_df['orderID'].unique())
        selected_order_id = st.selectbox(
            "Order ID",
            order_ids,
            help="Select the subject property you want to analyze"
        )
        
        # Display subject property details
        subject_data = subjects_df[subjects_df['orderID'] == selected_order_id].iloc[0]
        
        st.subheader("Subject Property Details")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("GLA (sqft)", f"{subject_data['gla_clean']:,.0f}")
            st.metric("Bedrooms", f"{subject_data['bedrooms_clean']:.0f}")
            st.metric("Property Type", subject_data['property_type_clean'])
        
        with col2:
            st.metric("Lot Size (sqft)", f"{subject_data['lot_size_clean']:,.0f}")
            st.metric("Bathrooms", f"{subject_data['bathrooms_clean']:.1f}")
            st.metric("Stories", subject_data['stories_clean'])
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📊 Recommendations", "🔍 Explanations", "📈 Analytics"])
    
    # Get recommendations for selected property
    property_recommendations = recommendations_df[recommendations_df['orderID'] == selected_order_id].sort_values('rank')
    
    with tab1:
        st.header("Top 3 Recommended Comparable Properties")
        
        if len(property_recommendations) > 0:
            # Display recommendations in columns
            cols = st.columns(3)
            
            for idx, (_, rec) in enumerate(property_recommendations.iterrows()):
                with cols[idx]:
                    st.markdown(f"### 🏆 Rank #{rec['rank']}")
                    
                    # Property details in a nice card format
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    
                    # Display key metrics
                    st.metric("Match Score", f"{rec['pred']:.3f}")
                    st.metric("GLA", f"{rec['gla_clean']:,.0f} sqft")
                    st.metric("Bedrooms", f"{rec['bedrooms_clean']:.0f}")
                    st.metric("Bathrooms", f"{rec['bathrooms_clean']:.1f}")
                    
                    # Show differences
                    st.caption("Differences from Subject:")
                    st.text(f"GLA: {rec['gla_diff']:+,.0f} sqft")
                    st.text(f"Bedrooms: {rec['bedroom_diff']:+.0f}")
                    st.text(f"Bathrooms: {rec['bathroom_diff']:+.1f}")
                    
                    # Recent sale indicator
                    if rec['sold_recently_90'] == 1:
                        st.success("✅ Sold within 90 days")
                    elif rec['sold_recently_180'] == 1:
                        st.info("📅 Sold within 180 days")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Property comparison chart
            st.subheader("Visual Property Comparison")
            comparison_fig = create_property_comparison_chart(subject_data, property_recommendations)
            st.plotly_chart(comparison_fig, use_container_width=True)
        else:
            st.warning("No recommendations found for this property.")
    
    with tab2:
        st.header("Why These Properties Were Selected")
        
        if len(property_recommendations) > 0:
            # Create SHAP explainer
            sample_data = property_recommendations[feature_columns].head(10)
            explainer = create_shap_explainer(model, sample_data)
            
            # Get SHAP values for recommendations
            shap_values = get_shap_explanation(
                model, 
                property_recommendations[feature_columns].values,
                explainer
            )
            
            # Feature importance plot
            st.subheader("Overall Feature Importance")
            importance_fig = create_feature_importance_plot(
                shap_values, 
                property_recommendations[feature_columns],
                feature_columns
            )
            st.plotly_chart(importance_fig, use_container_width=True)
            
            # Individual property explanations
            st.subheader("Individual Property Explanations")
            
            for idx, (_, rec) in enumerate(property_recommendations.iterrows()):
                with st.expander(f"Explanation for Rank #{rec['rank']} Property"):
                    # Get SHAP values for this specific property
                    property_shap = shap_values[idx]
                    
                    # Create a dataframe for this property's SHAP values
                    shap_df = pd.DataFrame({
                        'Feature': feature_columns,
                        'Value': rec[feature_columns].values,
                        'SHAP Impact': property_shap
                    }).sort_values('SHAP Impact', key=abs, ascending=False)
                    
                    # Display explanation
                    st.write("**Key factors for this recommendation:**")
                    
                    for _, row in shap_df.iterrows():
                        impact = "positive" if row['SHAP Impact'] > 0 else "negative"
                        st.write(f"- **{row['Feature']}** (value: {row['Value']:.2f}) has a {impact} impact of {abs(row['SHAP Impact']):.3f}")
                    
                    # Generate natural language explanation
                    explanation = generate_explanation(rec, shap_df, subject_data)
                    st.info(explanation)
    
    with tab3:
        st.header("Model Analytics")
        
        # Model performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Type", "XGBoost Ranking")
            st.metric("Features Used", len(feature_columns))
        
        with col2:
            st.metric("Training Accuracy", "94.44%")
            st.metric("Test Hit Rate", "94.44%")
        
        with col3:
            st.metric("Total Properties", f"{len(candidates_df):,}")
            st.metric("Total Subjects", f"{len(subjects_df):,}")
        
        # Feature statistics
        st.subheader("Feature Distribution Analysis")
        
        # Select feature to analyze
        selected_feature = st.selectbox("Select feature to analyze", feature_columns)
        
        # Create distribution plot
        fig = px.histogram(
            candidates_df[candidates_df[selected_feature].notna()],
            x=selected_feature,
            nbins=50,
            title=f"Distribution of {selected_feature}",
            labels={selected_feature: selected_feature.replace('_', ' ').title()}
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    # Configure Streamlit
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    # Run the main app
    main()