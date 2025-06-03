import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import plotly.express as px
from datetime import datetime
import os

# Set page configuration
st.set_page_config(
    page_title="Property Recommendation System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS for minimal dark style ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp { background-color: #0a0a0a; color: #e5e7eb; }
    .main-header-title { font-size: 3rem; font-weight: 600; color: #fff; text-align: center; margin-bottom: 0.5rem; letter-spacing: -0.02em; }
    .sub-header-subtitle { font-size: 1.5rem; color: #6b7280; text-align: center; margin-bottom: 3rem; font-weight: 300; }
    .property-card { background: #1a1a1a; border: 1px solid #262626; padding: 2rem; border-radius: 16px; height: 100%; transition: all 0.3s ease; display: flex; flex-direction: column; margin-bottom: 1.7rem;}
    .property-card:hover { border-color: #404040; transform: translateY(-2px);}
    .rank-number { font-size: 3rem; font-weight: 700; color: #fff; line-height: 1; margin-bottom: 0.5rem; }
    .match-score { font-size: 1.8rem; font-weight: 600; color: #10b981; margin-bottom: 0.25rem; }
    .match-label { font-size: 0.875rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 1.5rem; }
    .property-stats { display: flex; justify-content: space-between; padding-top: 1.5rem; border-top: 1px solid #262626; margin-top: auto;}
    .stat-block { text-align: center; }
    .stat-value { font-size: 1.25rem; font-weight: 600; color: #fff; }
    .stat-label { font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.25rem; }
    .insight-card { background: #1a1a1a; border: 1px solid #262626; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; transition: all 0.2s ease; }
    .insight-positive { border-left: 3px solid #10b981; }
    .insight-neutral { border-left: 3px solid #f59e0b; }
    .insight-negative { border-left: 3px solid #ef4444; }
    .insight-title { font-size: 0.9rem; font-weight: 600; color: #fff; margin-bottom: 0.25rem; }
    .insight-description { font-size: 0.8rem; color: #9ca3af; }
    .section-header { font-size: 1.5rem; font-weight: 600; color: #fff; margin: 2.5rem 0 1.5rem 0; display: flex; align-items: center; gap: 0.75rem; }
    .section-icon { font-size: 1.25rem; }
    .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-top: 2rem; }
    .stat-card { background: #1a1a1a; border: 1px solid #262626; padding: 1.5rem; border-radius: 12px; text-align: center; }
    .stat-card-value { font-size: 2rem; font-weight: 700; color: #fff; margin-bottom: 0.25rem; }
    .stat-card-label { font-size: 0.875rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }
    div[data-testid="stMetric"] { background-color: #1a1a1a; border: 1px solid #262626; padding: 1rem; border-radius: 8px;}
    div[data-testid="stMetric"] label { color: #6b7280;}
    div[data-testid="stMetric"] div { color: #fff;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data_and_model():
    model = xgb.XGBRanker()
    model_path = "../model/xgb_ranking_model.ubj"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    model.load_model(model_path)
    subjects_path = "../data/cleaned/subjects_cleaned.csv"
    candidates_path = "../data/model_ready/candidates_pair_model_ready.csv"
    recommendations_path = "../data/results/top3_candidates_recommendations.csv"
    for path, name in [(subjects_path, 'Subjects'), (candidates_path, 'Candidates'), (recommendations_path, 'Recommendations')]:
        if not os.path.exists(path):
            st.error(f"{name} file not found at: {path}")
            st.stop()
    subjects_df = pd.read_csv(subjects_path)
    candidates_df = pd.read_csv(candidates_path)
    recommendations_df = pd.read_csv(recommendations_path)
    return model, subjects_df, candidates_df, recommendations_df

feature_columns = [
    'gla_diff', 'lot_size_diff', 'bedroom_diff', 'bathroom_diff',
    'room_count_diff', 'same_property_type', 'same_storey_type',
    'sold_recently_90'
]

@st.cache_resource
def create_shap_explainer(_model):
    explainer = shap.TreeExplainer(_model)
    return explainer

def get_shap_values_for_recommendations(explainer, recommendations_features_df):
    shap_values = explainer.shap_values(recommendations_features_df)
    return shap_values

def create_shap_feature_importance_plot(shap_values_matrix, features_df_columns):
    import plotly.express as px
    mean_abs_shap = np.abs(shap_values_matrix).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': features_df_columns,
        'Importance': mean_abs_shap
    }).sort_values('Importance', ascending=True)
    fig = px.bar(
        importance_df, 
        x='Importance', 
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale=px.colors.sequential.Blues_r
    )
    fig.update_layout(
        height=400 + len(features_df_columns) * 10,
        xaxis_title="Mean Absolute SHAP Value",
        yaxis_title="Features",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e5e7eb',
        coloraxis_showscale=False
    )
    return fig

def generate_property_insights(rec, subject_data):
    insights = []
    if pd.notna(subject_data['gla_clean']) and subject_data['gla_clean'] > 0 and pd.notna(rec.get('gla_diff')):
        gla_diff_pct = (rec['gla_diff'] / subject_data['gla_clean']) * 100
        if abs(gla_diff_pct) < 10: insights.append(("Excellent Size Match", f"Only {abs(gla_diff_pct):.1f}% difference in living area.", "positive"))
        elif abs(gla_diff_pct) < 20: insights.append(("Good Size Match", f"{abs(gla_diff_pct):.1f}% difference in living area.", "neutral"))
        else: insights.append(("Size Variance", f"{abs(gla_diff_pct):.1f}% difference in living area.", "negative"))
    else: insights.append(("Size Comparison", "Living area data unavailable or subject GLA is zero.", "neutral"))
    if pd.notna(rec.get('same_property_type')):
        insights.append(("Same Property Type" if rec['same_property_type'] == 1 else "Different Property Type", 
                         "Matches subject's property category." if rec['same_property_type'] == 1 else "Differs from subject's property category.",
                         "positive" if rec['same_property_type'] == 1 else "negative"))
    if pd.notna(rec.get('sold_recently_90')) and rec['sold_recently_90'] == 1:
        insights.append(("Recent Sale", "Sold within 90 days - highly relevant pricing.", "positive"))
    elif pd.notna(rec.get('sold_recently_180')) and rec.get('sold_recently_180') == 1:
        insights.append(("Fairly Recent Sale", "Sold within 180 days - good pricing indicator.", "neutral"))
    elif pd.notna(rec.get('sold_recently_90')): insights.append(("Older Sale", "Sale date may impact price relevance.", "negative"))
    if pd.notna(rec.get('room_count_diff')):
        if rec['room_count_diff'] == 0: insights.append(("Perfect Room Match", "Identical number of rooms.", "positive"))
        elif abs(rec['room_count_diff']) == 1: insights.append(("Similar Layout", f"Only {abs(rec['room_count_diff']):.0f} room difference.", "neutral"))
    return insights

def generate_nlp_explanation_from_shap(property_rec, shap_df_for_property, subject_data):
    top_features = shap_df_for_property.head(3)
    explanation_points = []
    for _, feature_row in top_features.iterrows():
        feature_name_display = feature_row['Feature'].replace('_', ' ').title()
        actual_value = feature_row['Value']
        shap_impact = feature_row['SHAP Impact']
        if 'Diff' in feature_name_display:
            base_feature_name = feature_name_display.replace(' Diff', '')
            if abs(actual_value) < 0.05 * (subject_data.get(base_feature_name.lower().replace(' ','_') + '_clean', 1) or 1):
                 explanation_points.append(f"The {base_feature_name} is very similar to the subject property.")
            else:
                explanation_points.append(f"The {base_feature_name} difference of {actual_value:.1f} significantly {'contributed to' if shap_impact > 0 else 'detracted from'} the match score.")
        elif feature_row['Feature'] == 'same_property_type':
            explanation_points.append(f"Being the {'same' if actual_value == 1 else 'a different'} property type ({subject_data['property_type_clean'] if actual_value == 1 else 'other'}) {'positively' if actual_value == 1 else 'negatively'} impacted the score.")
        elif feature_row['Feature'] == 'same_storey_type':
             explanation_points.append(f"Having the {'same' if actual_value == 1 else 'a different'} story type ({subject_data['stories_clean'] if actual_value == 1 else 'other'}) impacted comparability.")
        elif feature_row['Feature'] == 'sold_recently_90':
            explanation_points.append(f"It {'was sold recently (within 90 days)' if actual_value == 1 else 'was not sold very recently'}, which is a key factor for price relevance.")
        else:
            explanation_points.append(f"The characteristic '{feature_name_display}' (value: {actual_value:.2f}) had a notable {'positive' if shap_impact > 0 else 'negative'} influence on the ranking.")
    pred_score = property_rec['pred']
    if pred_score > 0.5: overall = f"Overall, an excellent comparable with a high match score ({pred_score:.3f})."
    elif pred_score > 0.3: overall = f"A good comparable with a moderate match score ({pred_score:.3f})."
    else: overall = f"Some similarities, but a lower match score ({pred_score:.3f})."
    return f"This property ranks #{property_rec['rank']}.\nKey contributing factors:\n• " + "\n• ".join(explanation_points) + f"\n\n{overall}"

# --- SHAP Pill Renderer ---
def render_shap_factors(shap_df_for_property):
    def factor_style(val):
        if val > 0.01:
            return "background: linear-gradient(90deg, #22d3ee33, #16a34aff); color: #16a34a; font-weight:600;"
        elif val < -0.01:
            return "background: linear-gradient(90deg, #fca5a533, #ef4444ff); color: #ef4444; font-weight:600;"
        else:
            return "background: linear-gradient(90deg, #fde68a33, #f59e0bff); color: #f59e0b; font-weight:600;"
    html = '<div style="display:flex;flex-wrap:wrap;gap:0.75rem;margin-bottom:1.5rem;">'
    for _, row in shap_df_for_property.head(5).iterrows():
        feature = row["Feature"].replace("_", " ").title()
        value = row["Value"]
        impact = row["SHAP Impact"]
        pill = f'''
            <span style="padding:0.5rem 1.2rem;border-radius:999px;
                {factor_style(impact)}
                margin-right:0.4rem;display:inline-flex;align-items:center;box-shadow:0 2px 8px #2223;">
                <span style="font-weight:600;margin-right:0.4em;">{feature}</span>
                <span style="font-size:0.95em;opacity:0.7;">{value:.2f}</span>
                <span style="margin-left:0.7em;font-size:0.96em;">{'+' if impact>0 else ''}{impact:.3f}</span>
            </span>
        '''
        html += pill
    html += '</div>'
    return html

def pretty_summary(text):
    points = [pt.strip() for pt in text.replace("\n", " ").split("•") if pt.strip()]
    html = "<div style='background: linear-gradient(90deg,#0f766e33,#27272a 80%);border-radius:14px;padding:1.2rem 1.5rem;margin:1rem 0;color:#fff;font-size:1.07rem;box-shadow:0 1px 10px #1113;border:1px solid #2224;'>"
    for pt in points:
        html += f"<div style='margin:0.35em 0;line-height:1.7;'>{pt}</div>"
    html += "</div>"
    return html

def generate_professional_narrative(rec, subject_data, shap_df):
    """Generate a professional appraisal narrative for a comparable property"""
    
    # Start with property overview
    narrative_parts = []
    
    # Opening statement
    property_type = rec.get('property_type_clean', 'property')
    narrative_parts.append(f"""
    <p style="color: #e5e7eb; line-height: 1.8; margin-bottom: 1rem;">
        This <strong>{property_type}</strong> presents as a {classify_match_quality(rec.get('pred', 0))} comparable for the subject property. 
        The property features <strong>{rec.get('gla_clean', 0):,.0f} square feet</strong> of living area 
        with <strong>{rec.get('bedrooms_clean', 0):.0f} bedrooms</strong> and <strong>{rec.get('bathrooms_clean', 0):.1f} bathrooms</strong>.
    </p>
    """)
    
    # Size comparison analysis
    gla_diff_pct = abs(rec.get('gla_diff', 0) / subject_data.get('gla_clean', 1) * 100) if subject_data.get('gla_clean', 0) > 0 else 0
    size_analysis = analyze_size_comparison(rec, subject_data, gla_diff_pct)
    narrative_parts.append(f'<p style="color: #e5e7eb; line-height: 1.8; margin-bottom: 1rem;">{size_analysis}</p>')
    
    # Location and market conditions
    market_analysis = analyze_market_conditions(rec)
    narrative_parts.append(f'<p style="color: #e5e7eb; line-height: 1.8; margin-bottom: 1rem;">{market_analysis}</p>')
    
    # Key adjustments needed
    adjustments = analyze_adjustments(rec, subject_data, shap_df)
    narrative_parts.append(f"""
    <p style="color: #e5e7eb; line-height: 1.8; margin-bottom: 1rem;">
        <strong>Key Considerations:</strong><br/>
        {adjustments}
    </p>
    """)
    
    # Professional conclusion
    conclusion = generate_conclusion(rec, subject_data)
    narrative_parts.append(f"""
    <p style="color: #e5e7eb; line-height: 1.8; margin-bottom: 0;">
        <strong>Overall Assessment:</strong> {conclusion}
    </p>
    """)
    
    return ''.join(narrative_parts)

def classify_match_quality(pred_score):
    """Classify the quality of the match based on prediction score"""
    if pred_score > 0.6:
        return "highly relevant"
    elif pred_score > 0.4:
        return "good"
    elif pred_score > 0.2:
        return "moderate"
    else:
        return "marginal"

def analyze_size_comparison(rec, subject_data, gla_diff_pct):
    """Generate professional size comparison analysis"""
    if gla_diff_pct < 5:
        return f"The living area is nearly identical to the subject property, with only a {gla_diff_pct:.1f}% variance, requiring minimal adjustment for size differences."
    elif gla_diff_pct < 10:
        return f"The {gla_diff_pct:.1f}% difference in living area represents a minor variance that should be considered in the final valuation. This difference is within acceptable parameters for a reliable comparable."
    elif gla_diff_pct < 20:
        direction = "larger" if rec.get('gla_clean', 0) > subject_data.get('gla_clean', 0) else "smaller"
        return f"This property is {gla_diff_pct:.1f}% {direction} than the subject, requiring a moderate adjustment. The size difference may impact the property's utility and market appeal differently than the subject."
    else:
        direction = "larger" if rec.get('gla_clean', 0) > subject_data.get('gla_clean', 0) else "smaller"
        return f"With a {gla_diff_pct:.1f}% size variance ({direction}), this property requires significant adjustment. The substantial difference in living area suggests this comparable should be weighted accordingly in the final analysis."

def analyze_market_conditions(rec):
    """Analyze market conditions based on sale recency"""
    if rec.get('sold_recently_90', 0) == 1:
        return "The property sold within the last 90 days, providing excellent market timing relevance. No significant time adjustment is required, and the sale price reflects current market conditions."
    elif rec.get('sold_recently_180', 0) == 1:
        return "Having sold within the last 180 days, this comparable offers good market timing. Minor time adjustments may be warranted depending on local market trends during this period."
    else:
        return "The sale date of this property may require time adjustment to reflect current market conditions. Market appreciation or depreciation since the sale should be carefully considered."

def analyze_adjustments(rec, subject_data, shap_df):
    """Analyze key adjustments needed"""
    adjustments = []
    
    # Property type adjustment
    if rec.get('same_property_type', 0) == 0:
        adjustments.append(f"• Property type difference ({rec.get('property_type_clean', 'N/A')} vs. {subject_data.get('property_type_clean', 'N/A')}) may require adjustment for market preference")
    
    # Room count adjustments
    bed_diff = rec.get('bedroom_diff', 0)
    bath_diff = rec.get('bathroom_diff', 0)
    
    if abs(bed_diff) > 0:
        adjustments.append(f"• Bedroom count difference ({bed_diff:+.0f}) impacts functional utility")
    
    if abs(bath_diff) > 0.5:
        adjustments.append(f"• Bathroom count variance ({bath_diff:+.1f}) affects market appeal")
    
    # Lot size if significant
    if pd.notna(rec.get('lot_size_diff')) and abs(rec.get('lot_size_diff', 0)) > 2000:
        lot_direction = "larger" if rec.get('lot_size_diff', 0) > 0 else "smaller"
        adjustments.append(f"• Lot size is {abs(rec.get('lot_size_diff', 0)):,.0f} sqft {lot_direction}, impacting land value")
    
    if not adjustments:
        adjustments.append("• Minimal adjustments required due to high similarity")
    
    return "<br/>".join(adjustments)

def generate_conclusion(rec, subject_data):
    """Generate professional conclusion"""
    match_score = (rec.get('pred', 0) + 1) / 2 * 100
    
    if match_score > 80:
        return "This property represents an excellent comparable with minimal adjustments required. The high degree of similarity in key characteristics makes this a reliable indicator of market value."
    elif match_score > 65:
        return "This is a good comparable that provides solid market evidence. While some adjustments are necessary, the overall similarity supports its use in the valuation analysis."
    elif match_score > 50:
        return "This property serves as a useful comparable with moderate adjustments required. The differences should be carefully considered when weighting this sale in the final value conclusion."
    else:
        return "While this property has some similarities to the subject, significant adjustments are required. It should be used with caution and given less weight in the final analysis."

def main():
    st.markdown('<h1 class="main-header-title">🏠 Property Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header-subtitle">AI-Powered Comparable Property Analysis with Explainability</p>', unsafe_allow_html=True)
    model, subjects_df, candidates_df, recommendations_df = load_data_and_model()
    with st.sidebar:
        st.markdown("### <span class='section-icon'>🎯</span> Property Selection", unsafe_allow_html=True)
        order_ids = sorted(subjects_df['orderID'].unique())
        selected_order_id = st.selectbox("Select Property ID", order_ids, format_func=lambda x: f"Property #{x}")
        subject_data = subjects_df[subjects_df['orderID'] == selected_order_id].iloc[0]
        st.markdown("### <span class='section-icon'>📊</span> Subject Property Overview", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("GLA (sqft)", f"{subject_data.get('gla_clean', 0):,.0f}")
            st.metric("Bedrooms", f"{subject_data.get('bedrooms_clean', 0):.0f}")
        with col2:
            st.metric("Lot Size (sqft)", f"{subject_data.get('lot_size_clean', 0):,.0f}")
            st.metric("Bathrooms", f"{subject_data.get('bathrooms_clean', 0):.1f}")
        st.metric("Property Type", subject_data.get('property_type_clean', 'N/A'))
        st.metric("Stories", subject_data.get('stories_clean', 'N/A'))
    property_recommendations = recommendations_df[recommendations_df['orderID'] == selected_order_id].sort_values('rank').head(3)

    # Calculate SHAP values ONCE, to use in Recommendations
    recommendations_features_df = property_recommendations[feature_columns]
    explainer = create_shap_explainer(model)
    shap_values_matrix = get_shap_values_for_recommendations(explainer, recommendations_features_df)

    tab1, tab2, tab3 = st.tabs(["📊 Recommendations", "🔍 Explanations", "📈 Analytics"])

    with tab1:
    # Subject property ID display
        st.markdown(
            f'<div style="font-size:1.2rem; color:#a1a1aa; margin-bottom:0.3rem;">Subject Property ID: <span style="color:#fff;font-weight:600;">{selected_order_id}</span></div>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="section-header"><span class="section-icon">🏆</span> Top Comparable Properties</div>', unsafe_allow_html=True)
        if not property_recommendations.empty:
            cols = st.columns(len(property_recommendations))
            for idx, (_, rec) in enumerate(property_recommendations.iterrows()):
                with cols[idx]:
                    match_percentage = (rec.get('pred',0) + 1) / 2 * 100 
                    candidate_id = rec.get('id', 'N/A')
                    try:
                        candidate_id_str = f"{int(candidate_id)}" if candidate_id != 'N/A' and pd.notna(candidate_id) else "N/A"
                    except Exception:
                        candidate_id_str = str(candidate_id)
                    st.markdown(f"""<div class="property-card">
                        <div class="rank-number">#{rec.get('rank','?')}</div>
                        <div style="font-size:0.95rem; color:#a1a1aa; margin-bottom:0.5rem;">
                            <span style="color:#fff;font-weight:600;">ID:</span> {candidate_id_str}
                        </div>
                        <div class="match-score">{match_percentage:.1f}%</div>
                        <div class="match-label">Match Score</div>
                        <div class="property-stats">
                            <div class="stat-block"><div class="stat-value">{rec.get('gla_clean', 0):,.0f}</div><div class="stat-label">SQ FT</div></div>
                            <div class="stat-block"><div class="stat-value">{rec.get('bedrooms_clean', 0):.0f}</div><div class="stat-label">BEDS</div></div>
                            <div class="stat-block"><div class="stat-value">{rec.get('bathrooms_clean', 0):.1f}</div><div class="stat-label">BATHS</div></div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                    st.markdown("<div style='height:1.1rem;'></div>", unsafe_allow_html=True)
                    with st.expander("View Detailed Insights", expanded=False):
                        st.markdown('<div style="padding:0.8rem 0;">', unsafe_allow_html=True)
                        insights = generate_property_insights(rec, subject_data)
                        if insights:
                            for title, desc, sentiment in insights:
                                st.markdown(f'<div class="insight-card insight-{sentiment}"><div class="insight-title">{title}</div><div class="insight-description">{desc}</div></div>', unsafe_allow_html=True)
                        else:
                            st.write("No specific insights generated for this property.")
                        st.markdown('</div>', unsafe_allow_html=True)
            
                    # Predicted Price & Avg Price per SqFt for Top 3
            if 'close_price' in property_recommendations:
                avg_pred_price = property_recommendations['close_price'].mean()
                avg_price_per_sqft = (property_recommendations['close_price'] / property_recommendations['gla_clean']).mean()
                
                st.markdown('<div class="section-header" style="margin-top: 2.5rem;"><span class="section-icon">💰</span> Predicted Market Value (Top 3 Avg)</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-card-value">${avg_pred_price:,.0f}</div>
                        <div class="stat-card-label">Avg Predicted Price</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-card-value">${avg_price_per_sqft:,.0f} <span style="font-size:1.1rem;">/sqft</span></div>
                        <div class="stat-card-label">Avg Price per SqFt</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No closing price data available for recommendations.")

             # ======= MOVE THE SHAP SECTION HERE ==========
            st.markdown('<div class="section-header" style="margin-top: 3rem;"><span class="section-icon">🔍</span> AI Decision Factors (SHAP Explanations)</div>', unsafe_allow_html=True)
            st.markdown("Based on the average impact of features across the top recommendations for this subject property.")
            importance_fig = create_shap_feature_importance_plot(shap_values_matrix, feature_columns)
            st.plotly_chart(importance_fig, use_container_width=True)
            # ==============================================

            st.markdown('<div class="section-header" style="margin-top: 3rem;"><span class="section-icon">📊</span> Quick Statistics for Recommendations</div>', unsafe_allow_html=True)
            avg_match_pct = (property_recommendations['pred'].mean() + 1) / 2 * 100 if not property_recommendations['pred'].empty else 0
            recent_sales = property_recommendations['sold_recently_90'].sum() if 'sold_recently_90' in property_recommendations else 0
            same_type = property_recommendations['same_property_type'].sum() if 'same_property_type' in property_recommendations else 0
            avg_gla_diff = property_recommendations['gla_diff'].mean() if 'gla_diff' in property_recommendations and not property_recommendations['gla_diff'].empty else 0

            st.markdown(f"""
            <div class="stats-grid">
                <div class="stat-card"><div class="stat-card-value">{avg_match_pct:.1f}%</div><div class="stat-card-label">Avg Match</div></div>
                <div class="stat-card"><div class="stat-card-value">{recent_sales}</div><div class="stat-card-label">Recent Sales</div></div>
                <div class="stat-card"><div class="stat-card-value">{same_type}</div><div class="stat-card-label">Same Type</div></div>
                <div class="stat-card"><div class="stat-card-value">{abs(avg_gla_diff):,.0f}</div><div class="stat-card-label">Avg Size Diff</div></div>
            </div>""", unsafe_allow_html=True)

        

        else:
            st.warning("No recommendations found for this property.")


    with tab2:
        st.markdown('<div class="section-header"><span class="section-icon">📝</span> Professional Appraisal Analysis</div>', unsafe_allow_html=True)
        
        if not property_recommendations.empty:
            # Add subject property summary first
            st.markdown(f"""
            <div style="background: #1a1a1a; border: 1px solid #262626; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;">
                <h3 style="color: #fff; margin-bottom: 1rem;">Subject Property Summary</h3>
                <p style="color: #9ca3af; line-height: 1.8;">
                    The subject property (ID: {selected_order_id}) is a <strong style="color: #fff;">{subject_data.get('property_type_clean', 'N/A')}</strong> 
                    with <strong style="color: #fff;">{subject_data.get('gla_clean', 0):,.0f} square feet</strong> of living area, 
                    <strong style="color: #fff;">{subject_data.get('bedrooms_clean', 0):.0f} bedrooms</strong>, and 
                    <strong style="color: #fff;">{subject_data.get('bathrooms_clean', 0):.1f} bathrooms</strong>. 
                    The property sits on a <strong style="color: #fff;">{subject_data.get('lot_size_clean', 0):,.0f} square foot</strong> lot 
                    and is configured as a <strong style="color: #fff;">{subject_data.get('stories_clean', 'N/A')}</strong> design.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Process each comparable
            for idx, (_, rec) in enumerate(property_recommendations.iterrows()):
                # Get SHAP values for this specific property
                shap_values_for_prop = shap_values_matrix[idx]
                shap_df = pd.DataFrame({
                    'Feature': feature_columns,
                    'Value': recommendations_features_df.iloc[idx].values,
                    'SHAP Impact': shap_values_for_prop
                }).sort_values('SHAP Impact', key=abs, ascending=False)
                
                # Generate professional narrative
                narrative = generate_professional_narrative(rec, subject_data, shap_df)
                
                # Property header
                st.markdown(f"""
                <div style="background: #1a1a1a; border: 1px solid #262626; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
                        <div>
                            <h3 style="color: #fff; margin: 0;">Comparable #{rec.get('rank', '?')}</h3>
                            <p style="color: #6b7280; margin: 0.25rem 0;">Property ID: {int(rec.get('id', 0)) if pd.notna(rec.get('id')) else 'N/A'}</p>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 1.5rem; font-weight: 700; color: #10b981;">{((rec.get('pred', 0) + 1) / 2 * 100):.1f}%</div>
                            <div style="color: #6b7280; font-size: 0.875rem;">Match Score</div>
                        </div>
                    </div>
                    
                    <div style="border-top: 1px solid #262626; padding-top: 1rem;">
                        {narrative}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add expandable detailed metrics
                with st.expander("View Detailed Metrics", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Living Area", f"{rec.get('gla_clean', 0):,.0f} sqft", 
                                f"{rec.get('gla_diff', 0):+,.0f} sqft")
                        st.metric("Bedrooms", f"{rec.get('bedrooms_clean', 0):.0f}", 
                                f"{rec.get('bedroom_diff', 0):+.0f}")
                    with col2:
                        st.metric("Bathrooms", f"{rec.get('bathrooms_clean', 0):.1f}", 
                                f"{rec.get('bathroom_diff', 0):+.1f}")
                        st.metric("Lot Size", f"{rec.get('lot_size_clean', 0):,.0f} sqft", 
                                f"{rec.get('lot_size_diff', 0):+,.0f} sqft")
                    with col3:
                        st.metric("Property Type", rec.get('property_type_clean', 'N/A'),
                                "✓ Match" if rec.get('same_property_type', 0) == 1 else "✗ Different")
                        st.metric("Sale Recency", 
                                "< 90 days" if rec.get('sold_recently_90', 0) == 1 else "> 90 days",
                                "✓ Recent" if rec.get('sold_recently_90', 0) == 1 else "")
        else:
            st.warning("No recommendations found for this property.")


    with tab3:
        st.markdown('<div class="section-header"><span class="section-icon">📈</span> Model & Data Analytics</div>', unsafe_allow_html=True)
        st.markdown("#### Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Type", "XGBoost Ranker")
            st.metric("Features Used", len(feature_columns))
        with col2:
            st.metric("Training Accuracy (Example)", "94.44%") 
            st.metric("Test Hit Rate (Example)", "94.44%")
        st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)
        st.markdown("#### Data Overview")
        col1a, col2a = st.columns(2)
        with col1a:
             st.metric("Total Properties in Dataset", f"{len(candidates_df):,}")
        with col2a:
             st.metric("Total Subject Properties", f"{len(subjects_df):,}")
        st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)
        st.markdown("#### Feature Distribution Analysis")
        st.write("Distribution of feature values in the candidates dataset.")
        selected_feature = st.selectbox("Select feature to analyze", feature_columns, key="analytics_feature_select")
        fig_dist = px.histogram(
            candidates_df[candidates_df[selected_feature].notna()],
            x=selected_feature,
            nbins=50,
            labels={selected_feature: selected_feature.replace('_', ' ').title()}
        )
        fig_dist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#e5e7eb',
            bargap=0.1
        )
        st.plotly_chart(fig_dist, use_container_width=True)

if __name__ == "__main__":
    main()

