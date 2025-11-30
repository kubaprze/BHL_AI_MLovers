"""
Eco-IT Hardware Recommender - Streamlit Frontend (Dark Mode, No Sidebar)
Run: streamlit run app.py
"""

import streamlit as st
import requests
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Eco Flow",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide sidebar
)

# ============ DARK MODE CSS ============
st.markdown("""
    <style>
    /* Force dark mode on everything */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0E1117 !important;
        color: #E6EDF3 !important;
    }
    
    [data-testid="stHeader"] {
        background-color: #0E1117 !important;
    }
    
    [data-testid="stToolbar"] {
        background-color: #0E1117 !important;
    }
    
    .stApp {
        background-color: #0E1117 !important;
    }
    
    .main {
        background-color: #0E1117 !important;
    }
    
    /* Hide sidebar completely */
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    .sidebar {
        display: none !important;
    }
    
    /* Text colors */
    p, label, span, div {
        color: #E6EDF3 !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #00D9FF !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #161B22;
        border-radius: 8px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: 1px solid rgba(0, 217, 255, 0.2);
        color: #E6EDF3;
        border-radius: 6px;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 217, 255, 0.15);
        border-color: #00D9FF;
        color: #00D9FF;
    }
    
    /* Text areas and inputs */
    .stTextArea textarea,
    .stTextInput input,
    textarea,
    input {
        background-color: #161B22 !important;
        color: #E6EDF3 !important;
        border: 1px solid rgba(0, 217, 255, 0.2) !important;
    }
    
    .stTextArea textarea::placeholder,
    .stTextInput input::placeholder {
        color: #666666 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00D9FF 0%, #0084D4 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 217, 255, 0.4) !important;
    }
    
    /* Metrics */
    .stMetric {
        background-color: #161B22 !important;
        border: 1px solid rgba(0, 217, 255, 0.1);
        border-radius: 8px;
        padding: 15px;
    }
    
    [data-testid="stMetricValue"] {
        color: #00D9FF !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #E6EDF3 !important;
    }
    
    /* Alert boxes */
    .stAlert {
        background-color: #161B22 !important;
        border: 1px solid rgba(0, 217, 255, 0.2) !important;
        color: #E6EDF3 !important;
    }
    
    .stSuccess {
        background-color: rgba(0, 217, 0, 0.1) !important;
        border: 1px solid rgba(0, 217, 0, 0.3) !important;
        color: #00D900 !important;
    }
    
    .stWarning {
        background-color: rgba(255, 193, 0, 0.1) !important;
        border: 1px solid rgba(255, 193, 0, 0.3) !important;
        color: #FFC100 !important;
    }
    
    .stError {
        background-color: rgba(255, 0, 0, 0.1) !important;
        border: 1px solid rgba(255, 0, 0, 0.3) !important;
        color: #FF0000 !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(0, 217, 255, 0.1) !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        background-color: #161B22 !important;
    }
    
    [role="table"] {
        background-color: #161B22 !important;
        color: #E6EDF3 !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #161B22;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #30363D;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #484F58;
    }
    </style>
    """, unsafe_allow_html=True)



# ============ CONSTANTS ============
API_URL = "http://localhost:8000"

# ============ HEADER ============
left_co, cent_co, right_co = st.columns(3)
with cent_co:
   st.image("img/Logo_bez_tla.png") 



st.markdown("---")

# ============ MAIN CONTENT ============

st.markdown("## 📝 Tell Us Your Needs")

col_input, col_btn = st.columns([4, 1])

with col_input:
    user_input = st.text_area(
        "What hardware do you need?",
        height=150,
        placeholder="""Examples:
        • I need a server for EU data center, energy efficiency is critical
        • Looking for workplace computers, low carbon footprint is important
        • I need high-performance laptops for AI workload, no budget constraints
        • Green IT initiative, most sustainable hardware available""",
        label_visibility="collapsed"
    )

with col_btn:
    st.markdown("")
    st.markdown("")
    submit_button = st.button("🔍 Recommend", use_container_width=True, type="primary")

# ============ RECOMMENDATION LOGIC ============
if submit_button and user_input.strip():
    with st.spinner("🤖 AI analyzing..."):
        try:
            response = requests.post(
                f"{API_URL}/recommend",
                json={"requirements": user_input},
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.result = result
                st.success("✅ Recommendations ready!")
            else:
                st.error(f"❌ API Error: {response.status_code}")
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend. Is FastAPI running on :8000?")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

elif submit_button and not user_input.strip():
    st.warning("📝 Please describe your hardware needs!")

# ============ DISPLAY RESULTS ============
if 'result' in st.session_state:
    result = st.session_state.result
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Requirements",
        "⭐ Recommendations",
        "📊 Comparison",
        "💾 Export"
    ])

    
    
    # TAB 1: Requirements
    with tab1:
        st.markdown("## Extracted Requirements")
        req = result['extracted_requirements']
        
        cols = st.columns(4)
        with cols[0]:
            st.metric("Product type", req['product_type'].title())
        with cols[1]:
            st.metric("Priority", req['priority'].title())
        with cols[2]:
            st.metric("Budget", req['budget'].title())
        # with cols[3]:
        #     st.metric("Region", req['region'].upper())
        
        # st.divider()
        # st.markdown("### Your Input")
        # st.info(result['user_input'])
    
    # TAB 2: Recommendations
    with tab2:
        st.markdown("## ⭐ Top Recommendations")
        recs = result['recommendations']
        
        rec_tabs = st.tabs([f"#{i+1}: {r['manufacturer']}" for i, r in enumerate(recs)])
        
        for idx, (tab, hw) in enumerate(zip(rec_tabs, recs)):
            with tab:
                # #  debug:
                # st.write(f"Debug - Keys in hw: {list(hw.keys())}")
                # st.write(f"Price value: {hw.get('price_euros')}")
                # st.write(f"Storage value: {hw.get('hard_drive')}")

                # Header with name and price
                col_header_left, col_header_right = st.columns([4, 1])
                
                with col_header_left:
                    st.markdown(f"### {hw['manufacturer']} {hw['name']}")
                    #st.markdown(f"_{hw['category']}_")
                
                with col_header_right:
                    price = hw.get('price_euros', 0)
                    if price:
                        st.markdown(f"### €{price:.0f}")
                    else:
                        st.markdown("### N/A")
                
                st.divider()
                
                # SCORES SECTION - 4 columns: Environmental, Performance, Cost, Overall
                st.markdown("#### 📊 Scores")
                score_col1, score_col2, score_col3, score_col4 = st.columns(4)
                
                # Calculate individual scores for display
                env_score = (100 - (hw['gwp_total'] / 20)) * 0.4 + (100 - (hw['yearly_tec'] / 5)) * 0.3
                env_score = min(100, max(0, env_score))
                
                perf_score = 50  # Simplified - would need to recalculate from requirements
                
                price_score = hw.get('price_euros', 0)
                budget_target = 1500  # Default medium
                price_ratio = price_score / budget_target if budget_target > 0 else 1
                price_score = min(100, max(0, 100 / price_ratio if price_ratio > 0 else 100))
                
                overall_score = hw['score']
                
                with score_col1:
                    st.metric(
                        "🌍 Environmental",
                        f"{env_score:.0f}",
                        delta="Score",
                        delta_color="off"
                    )
                
                with score_col2:
                    st.metric(
                        "⚡ Performance",
                        f"{perf_score:.0f}",
                        delta="Score",
                        delta_color="off"
                    )
                
                with score_col3:
                    st.metric(
                        "💰 Cost Value",
                        f"{price_score:.0f}",
                        delta="Score",
                        delta_color="off"
                    )
                
                with score_col4:
                    color = "green" if overall_score >= 80 else "blue" if overall_score >= 60 else "orange"
                    
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=overall_score,
                        title={'text': "Overall"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 40], 'color': "rgba(255, 0, 0, 0.2)"},
                                {'range': [40, 70], 'color': "rgba(255, 200, 0, 0.2)"},
                                {'range': [70, 100], 'color': "rgba(0, 200, 0, 0.2)"},
                            ],
                        }
                    ))
                    fig_gauge.update_layout(
                        height=150, 
                        margin=dict(l=10, r=10, t=50, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(22,27,34,0.5)",
                        font=dict(color="#E6EDF3", size=12)
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{idx}_score")

                    

                # REASONING
                st.markdown("#### 💡 Why This Recommendation?")
                st.info(hw['reasoning'])
                
                st.divider()
                
                # SPECS SECTION
                st.markdown("#### 🖥️ Specifications")
                specs_col1, specs_col2, specs_col3 = st.columns(3)
                
                with specs_col1:
                    mem = hw.get('memory')
                    if mem and mem not in ['N/A', '', 0, '0', None]:
                        try:
                            mem_display = f"{int(float(mem))} GB"
                        except:
                            mem_display = "N/A"
                    else:
                        mem_display = "N/A"
                    st.metric("Memory", mem_display)
                
                with specs_col2:
                    cpus = hw.get('number_cpu')
                    if cpus and cpus not in ['N/A', '', 0, '0', None]:
                        try:
                            cpus_display = f"{int(float(cpus))} cores"
                        except:
                            cpus_display = "N/A"
                    else:
                        cpus_display = "N/A"
                    st.metric("CPUs", cpus_display)
                
                with specs_col3:
                    drive = hw.get('hard_drive')
                    if drive:
                        drive_str = str(drive).strip().lower()
                        if (drive_str and 
                            drive_str not in ['nan', '0', '0 ssd', 'no hard drive', 'none', ''] and
                            ('gb' in drive_str or 'tb' in drive_str or 'ssd' in drive_str or 'hdd' in drive_str)):
                            drive_display = str(hw.get('hard_drive')).strip()
                        else:
                            drive_display = "N/A"
                    else:
                        drive_display = "N/A"
                    st.metric("Storage", drive_display)
                
                st.divider()
                
                # ENVIRONMENTAL IMPACT SECTION
                st.markdown("#### 🌱 Environmental Impact")
                env_col1, env_col2, env_col3, env_col4 = st.columns(4)
                
                with env_col1:
                    st.metric("GWP Total", f"{hw['gwp_total']:.0f} kgCO₂eq")

                with env_col2:
                    st.metric("Yearly Energy", f"{hw['yearly_tec']:.0f} kWh")

                with env_col3:
                    mfg_ratio = hw.get('gwp_manufacturing_ratio', 0)
                    use_ratio = hw.get('gwp_use_ratio', 0)
                    st.metric("Manufacturing", f"{mfg_ratio*100:.0f}%", delta=f"Use: {use_ratio*100:.0f}%", delta_color="off")

                with env_col4:
                    lifecycle = hw['gwp_total'] + (hw['yearly_tec'] * hw['lifetime'] / 1000)
                    st.metric("Lifecycle", f"{lifecycle:.0f} kgCO₂eq")

                
                st.divider()
                
                # BREAKDOWN CHARTS
                st.markdown("#### 📈 Impact Breakdown")
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    st.markdown("**Manufacturing vs Use Phase**")
                    breakdown = {
                        "Phase": ["Manufacturing", "Use"],
                        "Impact": [hw['gwp_manufacturing_ratio'] * 100, hw['gwp_use_ratio'] * 100]
                    }
                    fig_pie = px.pie(breakdown, names="Phase", values="Impact", 
                                    color_discrete_sequence=["#FF6B6B", "#4ECDC4"])
                    fig_pie.update_layout(
                        height=250,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(22,27,34,0.5)",
                        font=dict(color="#E6EDF3"),
                        margin=dict(l=10, r=10, t=10, b=10)
                    )
                    st.plotly_chart(fig_pie, use_container_width=True, key=f"rec_pie_{idx}_breakdown")
                
                with chart_col2:
                    st.markdown("**Score Composition**")
                    scores_breakdown = {
                        "Component": ["Environmental", "Performance", "Cost"],
                        "Score": [env_score, perf_score, price_score]
                    }
                    fig_bar = px.bar(scores_breakdown, x="Component", y="Score", 
                                    color="Component",
                                    color_discrete_sequence=["#32B899", "#5B9BD5", "#FFC000"],
                                    range_y=[0, 100])
                    fig_bar.update_layout(
                        height=250,
                        showlegend=False,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(22,27,34,0.5)",
                        font=dict(color="#E6EDF3"),
                        margin=dict(l=40, r=10, t=10, b=30),
                        yaxis_title="",
                        xaxis_title=""
                    )
                    st.plotly_chart(fig_bar, use_container_width=True, key=f"rec_bar_{idx}_scores")
                
                st.divider()

                
                st.markdown("#### 🌍 Metrics")
                sus_col1, sus_col2, sus_col3, = st.columns(3)
                
                with sus_col1:
                    st.metric("GWP Total", f"{hw['gwp_total']:.0f} kgCO₂eq")
                
                with sus_col2:
                    st.metric("Yearly Energy", f"{hw['yearly_tec']:.0f} kWh")
                
                with sus_col3:
                    st.metric("Lifetime", f"{hw['lifetime']} years")    
                
                # with sus_col4:
                #     lifecycle = hw['gwp_total'] + (hw['yearly_tec'] * hw['lifetime'] / 1000)
                #     st.metric("Lifecycle Impact", f"{lifecycle:.0f} kgCO₂eq")
                
                # st.divider()    
                
                # st.markdown("#### 📊 Breakdown")
                # breakdown = {
                #     "Phase": ["Manufacturing", "Use"],
                #     "Ratio": [hw['gwp_manufacturing_ratio'] * 100, hw['gwp_use_ratio'] * 100]
                # }   
                # fig_pie = px.pie(breakdown, names="Phase", values="Ratio", 
                #                 color_discrete_sequence=["#FF6B6B", "#4ECDC4"])
                # fig_pie.update_layout(
                #     paper_bgcolor="rgba(0,0,0,0)",
                #     plot_bgcolor="rgba(22,27,34,0.5)",
                #     font=dict(color="#E6EDF3")
                # )
                # st.plotly_chart(fig_gauge, use_container_width=True, key=f"rec_gauge_{idx}_overall")
                
                # st.markdown("**Score Composition**")
                # scores_breakdown = {
                #     "Component": ["Environmental", "Performance", "Cost"],
                #     "Score": [env_score, perf_score, price_score]
                # }
                # fig_bar = px.bar(scores_breakdown, x="Component", y="Score", 
                #                 color="Component",
                #                 color_discrete_sequence=["#32B899", "#5B9BD5", "#FFC000"],
                #                 range_y=[0, 100])
                # fig_bar.update_layout(
                #     height=250,
                #     showlegend=False,
                #     paper_bgcolor="rgba(0,0,0,0)",
                #     plot_bgcolor="rgba(22,27,34,0.5)",
                #     font=dict(color="#E6EDF3"),
                #     margin=dict(l=40, r=10, t=10, b=30),
                #     yaxis_title="",
                #     xaxis_title=""
                # )
                # st.plotly_chart(fig_bar, use_container_width=True, key=f"rec_bar_{idx}_scores")
    
    # TAB 3: Comparison
    with tab3:
        st.markdown("## Comparison")
        comp = result['comparison']
        recs = result['recommendations']
        
        st.markdown("### 📈 Summary")
        sum_col1, sum_col2, sum_col3 = st.columns(3)
        
        with sum_col1:
            st.metric("Avg GWP", f"{comp['average_gwp']:.0f} kgCO₂eq")
        with sum_col2:
            st.metric("Avg Energy", f"{comp['average_yearly_tec']:.0f} kWh")
        with sum_col3:
            st.metric("Winner", recs[0]['manufacturer'])
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### GWP Total")
            gwp_df = pd.DataFrame(comp['gwp_comparison'])
            fig_gwp = px.bar(gwp_df, x='name', y='gwp', color='gwp', 
                            color_continuous_scale='RdYlGn_r',
                            labels={'gwp': 'GWP (kgCO₂eq)', 'name': ''})
            fig_gwp.update_layout(
                height=400, 
                showlegend=False, 
                xaxis_tickangle=-45,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(22,27,34,0.5)",
                font=dict(color="#E6EDF3"),
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(88,166,255,0.1)"),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(88,166,255,0.1)")
            )
            st.plotly_chart(fig_gwp, use_container_width=True, key="gwp_chart")
        
        with col2:
            st.markdown("### Yearly Energy")
            tec_df = pd.DataFrame(comp['energy_comparison'])
            fig_tec = px.bar(tec_df, x='name', y='yearly_tec', color='yearly_tec',
                            color_continuous_scale='Blues',
                            labels={'yearly_tec': 'Energy (kWh)', 'name': ''})
            fig_tec.update_layout(
                height=400, 
                showlegend=False, 
                xaxis_tickangle=-45,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(22,27,34,0.5)",
                font=dict(color="#E6EDF3"),
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(88,166,255,0.1)"),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(88,166,255,0.1)")
            )
            st.plotly_chart(fig_tec, use_container_width=True, key="tec_chart")
        
        st.divider()
        
        st.markdown("### Lifecycle Impact")
        lifecycle_df = pd.DataFrame(comp['lifecycle_cost'])
        fig_lifecycle = px.bar(lifecycle_df, x='name', y='total_emissions', 
                              color='total_emissions', color_continuous_scale='YlOrRd',
                              labels={'total_emissions': 'Total (kgCO₂eq)', 'name': ''})
        fig_lifecycle.update_layout(
            height=400, 
            showlegend=False, 
            xaxis_tickangle=-45,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(22,27,34,0.5)",
            font=dict(color="#E6EDF3"),
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(88,166,255,0.1)"),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(88,166,255,0.1)")
        )
        st.plotly_chart(fig_lifecycle, use_container_width=True, key="lifecycle_chart")
        
        st.divider()
        
        st.markdown("### Table")
        table_data = []
        for i, hw in enumerate(recs):
            table_data.append({
                'Rank': i+1,
                'Manufacturer': hw['manufacturer'],
                'Model': hw['name'],
                'GWP': f"{hw['gwp_total']:.0f}",
                'Energy': f"{hw['yearly_tec']:.0f}",
                'Lifetime': hw['lifetime'],
                'Score': f"{hw['score']:.0f}"
            })
        
        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True, hide_index=True)
    
    # TAB 4: Export
    with tab4:
        st.markdown("## Export Results")
        
        json_str = json.dumps(result, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Download as JSON",
            data=json_str,
            file_name=f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        table_data = []
        for hw in recs:
            table_data.append({
                'Manufacturer': hw['manufacturer'],
                'Model': hw['name'],
                'GWP_kgCO2': hw['gwp_total'],
                'Energy_kWh': hw['yearly_tec'],
                'Lifetime_y': hw['lifetime'],
                'Score': hw['score']
            })
        
        df_export = pd.DataFrame(table_data)
        csv_str = df_export.to_csv(index=False)
        
        st.download_button(
            label="📥 Download as CSV",
            data=csv_str,
            file_name=f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        st.divider()
        st.dataframe(df_export, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #58A6FF; font-size: 0.9em;'>
    🌍 Eco-Flow Hardware Recommender v1.0 
    <br/>
    Made with ❤️ for sustainable IT infrastructure
</div>
""", unsafe_allow_html=True)
