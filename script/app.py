import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import plotly.express as px
import os
import openai



OPENAI_API_KEY = " " #put your API key

FEATURE_COLUMNS = [
    'gla_diff', 'lot_size_diff', 'bedroom_diff', 'bathroom_diff',
    'room_count_diff', 'same_property_type', 'same_storey_type',
    'sold_recently_90'
]

# Mapping for friendly feature names on y-axis
FRIENDLY_NAMES = {
    'gla_diff': 'GLA Difference',
    'lot_size_diff': 'Lot Size Difference',
    'bedroom_diff': 'Bedroom Difference',
    'bathroom_diff': 'Bathroom Difference',
    'room_count_diff': 'Room Count Difference',
    'same_property_type': 'Same Property Type',
    'same_storey_type': 'Same Storey Type',
    'sold_recently_90': 'Sold Within 90 Days'
}

st.set_page_config(
    page_title="Property Recommendation System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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


def chatgpt_summary_for_recommendations(subject_data, recommendations, summary_stats, shap_summary, api_key=None):
    """
    Generate a holistic summary of the recommended comparable properties.
    """
    # Build context text
    subject_str = (
        f"Subject property: {subject_data['gla_clean']:.0f} sqft, {subject_data['bedrooms_clean']} beds, "
        f"{subject_data['bathrooms_clean']} baths, {subject_data['property_type_clean']}, "
        f"sale date: {subject_data.get('effective_date_clean', 'N/A')}"
    )
    # Summarize top comps in text
    comp_lines = []
    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
        comp_lines.append(
            f"#{i}: ID {row['id']}, {row['gla_clean']:.0f} sqft, {row['bedrooms_clean']} beds, "
            f"{row['bathrooms_clean']} baths, match score {((row['pred'] + 1) / 2 * 100):.1f}%"
        )
    comps_str = "; ".join(comp_lines)
    # Stats summary
    stats_str = (
        f"Average predicted price: ${summary_stats['avg_pred_price']:,.0f}. "
        f"Avg match: {summary_stats['avg_match_pct']:.1f}%. "
        f"Avg size diff: {summary_stats['avg_gla_diff']:.0f} sqft."
    )

    prompt = (
        f"You are an expert real estate AI assistant. Given the following data, write a concise, insightful summary (4-6 sentences) for a human reader.\n"
        f"{subject_str}\n"
        f"Top comparable properties: {comps_str}\n"
        f"{stats_str}\n"
        f"Key decision factors: {shap_summary}\n"
        "Summarize the overall comparability, highlight key similarities and differences, and note any stand-out properties or factors."
    )

    # If NO API key is provided, show explicit reason:
    if not api_key or api_key.strip() == "":
        return (
            st.warning("🛑 **OpenAI summary not available. Please provide an OpenAI API key to enable this feature.**\n\n")
        )

    # Try calling OpenAI, fallback if error
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=180
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return (
            st.warning(f"OpenAI API call failed: {str(e)}.")
        )



def chatgpt_summarize_shap(importance_df, n=5, api_key=None):
    """
    Use OpenAI GPT to summarize top SHAP features (for openai>=1.0).
    Falls back to a rule-based summary if API fails or no key is provided.
    """
    top = importance_df.sort_values('Importance', ascending=False).head(n)
    lines = [f"{row['Feature']} (score {row['Importance']:.3f})" for _, row in top.iterrows()]
    features_str = "; ".join(lines)
    
    # If no API key provided, use fallback
    if not api_key or api_key == '':
        return generate_fallback_summary(top)
    
    try:
        client = openai.OpenAI(api_key=api_key)
        prompt = (
            f"Given the following features and their importance scores from a real estate comparable property AI model, "
            f"write a short summary (3-4 sentences) in natural language, explaining what features most drive the comp selection. "
            f"Features: {features_str}"
        )
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=120
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        # If API call fails, return fallback summary
        st.warning(f"OpenAI API call failed: {str(e)}. Using fallback summary.")
        return generate_fallback_summary(top)
    

def generate_fallback_summary(top_features_df):
    """
    Generate a rule-based summary when OpenAI API is not available.
    """
    if len(top_features_df) == 0:
        return "No feature importance data available."
    
    # Get top 3 features
    top_3 = top_features_df.head(3)
    
    feature_descriptions = {
        'GLA Difference': 'property size similarity',
        'Lot Size Difference': 'lot size comparison',
        'Bedroom Difference': 'bedroom count matching',
        'Bathroom Difference': 'bathroom count similarity',
        'Room Count Difference': 'overall room layout',
        'Same Property Type': 'property type matching',
        'Same Storey Type': 'building style similarity',
        'Sold Within 90 Days': 'recent sale timing'
    }
    
    # Build summary
    summary_parts = []
    for idx, (_, row) in enumerate(top_3.iterrows()):
        feature_name = row['Feature']
        importance = row['Importance']
        desc = feature_descriptions.get(feature_name, feature_name.lower())
        
        if idx == 0:
            summary_parts.append(f"The most important factor in selecting comparable properties is {desc} (importance: {importance:.2f})")
        elif idx == 1:
            summary_parts.append(f"followed by {desc}")
        else:
            summary_parts.append(f"and {desc}")
    
    summary = ", ".join(summary_parts) + "."
    summary += " These features significantly influence how well a property matches the subject property for valuation purposes."
    
    return summary



@st.cache_resource
def create_shap_explainer(_model):
    explainer = shap.TreeExplainer(_model)
    return explainer

def create_shap_feature_importance_plot(shap_values_matrix, features_df_columns):
    import plotly.graph_objects as go
    mean_abs_shap = np.abs(shap_values_matrix).mean(axis=0)
    friendly_labels = [FRIENDLY_NAMES.get(f, f) for f in features_df_columns]
    importance_df = pd.DataFrame({
        'Feature': friendly_labels,
        'Importance': mean_abs_shap
    }).sort_values('Importance', ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=importance_df["Importance"],
        y=importance_df["Feature"],
        orientation='h',
        marker=dict(
            color='#10b981',  # Your exact match score color
            line=dict(color='#13f3af', width=2),  # Subtle glow effect (optional)
        ),
        hoverlabel=dict(bgcolor="#0a0a0a", font_color="#e5e7eb"),
    ))

    fig.update_layout(
        height=400 + len(features_df_columns) * 10,
        xaxis_title="Mean Absolute SHAP Value",
        yaxis_title="Feature",
        plot_bgcolor='#0a0a0a',
        paper_bgcolor='#0a0a0a',
        font_color='#e5e7eb',
        bargap=0.24,
        margin=dict(l=80, r=40, t=25, b=40),
        xaxis=dict(showgrid=False, zeroline=False, linecolor='#262626'),
        yaxis=dict(showgrid=False, linecolor='#262626'),
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

def main():
    st.markdown('<h1 class="main-header-title">🏠 Property Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header-subtitle">AI-Powered Comparable Property Analysis with Explainability</p>', unsafe_allow_html=True)

    api_key = OPENAI_API_KEY
    
    # Optional: Allow users to input API key through sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings", unsafe_allow_html=True)
        with st.expander("OpenAI Configuration", expanded=False):
            user_api_key = st.text_input(
                "OpenAI API Key (optional)", 
                value="",
                type="password",
                help="Enter your OpenAI API key for AI-generated summaries. Leave blank to use fallback summaries."
            )
            if user_api_key:
                api_key = user_api_key
            
            if not api_key or api_key == '':
                st.info("No API key provided. Using rule-based summaries.")
            else:
                st.success("API key configured ✓")

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

    # Calculate SHAP values ONCE for Recommendations
    recommendations_features_df = property_recommendations[FEATURE_COLUMNS]
    explainer = create_shap_explainer(model)
    shap_values_matrix = explainer.shap_values(recommendations_features_df)

    tab1, tab2 = st.tabs(["📊 Recommendations", "📈 Analytics"])

    with tab1:
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

            # Quick Statistics for Recommendations
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




            # ==== SHAP Feature Importance Bar Chart ====
            st.markdown('<div class="section-header" style="margin-top: 3rem;"><span class="section-icon">🔍</span> Key AI Decision Factors (Feature Importance)</div>', unsafe_allow_html=True)
            st.markdown("These are the most important features influencing the comp ranking for this subject property.")

            # --- Build importance_df for both plot and summary ---
            mean_abs_shap = np.abs(shap_values_matrix).mean(axis=0)
            friendly_labels = [FRIENDLY_NAMES.get(f, f) for f in FEATURE_COLUMNS]
            importance_df = pd.DataFrame({
                'Feature': friendly_labels,
                'Importance': mean_abs_shap
            }).sort_values('Importance', ascending=False)

            # --- Plot SHAP feature importance chart (bars are already green) ---
            importance_fig = create_shap_feature_importance_plot(shap_values_matrix, FEATURE_COLUMNS)
            st.plotly_chart(importance_fig, use_container_width=True)

            # --- Get AI-written summary from OpenAI and display it below chart ---
            summary_text = chatgpt_summarize_shap(importance_df, n=3, api_key=api_key)
            if api_key and api_key != '':
                summary_label = "AI Summary"
                icon = "🤖"
            else:
                summary_label = "Summary"
                icon = "📊"

            st.markdown(
                f"<div style='margin-top:1.2em; padding: 1rem; background: #1a1a1a; border: 1px solid #262626; border-radius: 8px;'>"
                f"<div style='color:#10b981; font-weight: 600; margin-bottom: 0.5rem;'>{icon} {summary_label}:</div>"
                f"<div style='color:#e5e7eb; font-size:1.08rem;'>{summary_text}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

                        # --- Full executive summary of all recommendations ---
            summary_stats = {
                'avg_pred_price': avg_pred_price,
                'avg_match_pct': avg_match_pct,
                'avg_gla_diff': avg_gla_diff,
            }

            full_summary = chatgpt_summary_for_recommendations(
                subject_data,
                property_recommendations,
                summary_stats,
                summary_text,   # reuse SHAP summary string
                api_key=api_key
            )

            st.markdown(
                f"<div style='margin-top:1.2em; padding: 1.2rem; background: #18181b; border: 1.5px solid #10b981; border-radius: 10px;'>"
                f"<div style='color:#10b981; font-weight: 700; margin-bottom: 0.5rem; font-size:1.1rem;'>📝 Full Recommendation Summary:</div>"
                f"<div style='color:#e5e7eb; font-size:1.07rem;'>{full_summary}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.warning("No recommendations found for this property.")

    with tab2:
        st.markdown('<div class="section-header"><span class="section-icon">📈</span> Model & Data Analytics</div>', unsafe_allow_html=True)
        st.markdown("#### Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Type", "XGBoost Ranker")
            st.metric("Features Used", len(FEATURE_COLUMNS))
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
        selected_feature = st.selectbox("Select feature to analyze", FEATURE_COLUMNS, key="analytics_feature_select")
        # Use friendly feature name for the label
        feature_label = FRIENDLY_NAMES.get(selected_feature, selected_feature)
        fig_dist = px.histogram(
            candidates_df[candidates_df[selected_feature].notna()],
            x=selected_feature,
            nbins=50,
            labels={selected_feature: feature_label}
        )
        fig_dist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#e5e7eb',
            bargap=0.1
        )
        st.plotly_chart(fig_dist, use_container_width=True)

if __name__ == "__main__":
    main()
