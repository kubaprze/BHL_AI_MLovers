"""
Eco-IT Hardware Recommender - Streamlit Frontend
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
    page_title="Eco-IT Hardware Recommender",
    page_icon="🌍",
    layout="wide",  
    initial_sidebar_state="expanded"
)

# ============ CONSTANTS ============
API_URL = "http://localhost:8000"

# ============ STYLING ============
st.markdown("""
    <style>
    .main {
        padding: 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ============ HEADER ============
st.markdown("""
# 🌍 Eco-IT Hardware Recommender
### AI-powered sustainable hardware recommendations
""")

st.markdown("---")

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("### 📌 About This Tool")
    st.markdown("""
    This tool uses:
    - **LLM** (Local Ollama) to understand your needs
    - **ML** to rank hardware by sustainability
    - **Real data** from manufacturer carbon reports
    
    **Key Metrics:**
    - 🌍 GWP (Global Warming Potential)
    - ⚡ Yearly Energy Consumption (TEC)
    - 🏭 Manufacturing vs Use Phase
    - 🌐 Regional Energy Mix Impact
    """)
    
    st.divider()
    
    st.markdown("### 🚀 How It Works")
    st.markdown("""
    1. **Describe** your hardware needs
    2. **AI extracts** requirements automatically
    3. **ML ranks** products by priorities
    4. **Compare** sustainability metrics
    5. **Choose** wisely
    """)
    
    st.divider()
    
    st.markdown("### 🔧 API Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        st.success(f"✅ API Online ({health['hardware_count']} products)")
        if health.get('llm_available'):
            st.success("✅ LLM Available")
        else:
            st.warning("⚠️ LLM Offline (fallback mode)")
    except:
        st.error("❌ API Offline")
    
    st.divider()
    
    st.markdown("### 📊 Quick Stats")
    try:
        stats = requests.get(f"{API_URL}/hardware/stats", timeout=2).json()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Products", stats['total_products'])
        with col2:
            st.metric("Avg GWP", f"{stats['avg_gwp']:.0f}")
    except:
        pass

# ============ MAIN CONTENT ============

# Input section
st.markdown("## 📝 Tell Us Your Needs")

col_input, col_btn = st.columns([4, 1])

with col_input:
    user_input = st.text_area(
        "What hardware do you need?",
        height=100,
        placeholder="""Examples:
• I need a server for a small data center in Europe, energy efficiency is critical
• Looking for workplace computers for our office, low carbon footprint is important
• High-performance servers for AI workload, US-based, no budget constraints
• Small form factor servers for edge computing, Asia region, cost-effective""",
        label_visibility="collapsed"
    )

with col_btn:
    st.markdown("")
    submit_button = st.button("🔍 Recommend", use_container_width=True, type="primary", key="recommend_btn")

# ============ RECOMMENDATION LOGIC ============
if submit_button and user_input.strip():
    with st.spinner("🤖 AI is analyzing your needs..."):
        try:
            # Call backend API
            response = requests.post(
                f"{API_URL}/recommend",
                json={"requirements": user_input},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Store in session state for use across tabs
                st.session_state.result = result
                st.success("✅ Recommendations ready!")
            else:
                st.error(f"❌ API Error: {response.status_code}")
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend API. Make sure FastAPI is running on http://localhost:8000")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

elif submit_button and not user_input.strip():
    st.warning("📝 Please describe your hardware needs!")

# ============ DISPLAY RESULTS ============
if 'result' in st.session_state:
    result = st.session_state.result
    
    st.markdown("---")
    
    # ============ TAB 1: REQUIREMENTS ============
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Requirements",
        "⭐ Recommendations",
        "📊 Comparison",
        "💾 Export"
    ])
    
    with tab1:
        st.markdown("## Extracted Requirements")
        
        req = result['extracted_requirements']
        
        cols = st.columns(4)
        
        with cols[0]:
            st.metric("Use Case", req['use_case'].title())
        
        with cols[1]:
            st.metric("Priority", req['priority'].title())
        
        with cols[2]:
            st.metric("Budget", req['budget'].title())
        
        with cols[3]:
            st.metric("Region", req['region'].upper())
        
        st.divider()
        
        st.markdown("### Your Input")
        st.info(result['user_input'])
    
        # TAB 2: Recommendations
    with tab2:
        st.markdown("## Top Recommendations")
        recs = result['recommendations']
        
        rec_tabs = st.tabs([f"#{i+1}: {r['manufacturer']}" for i, r in enumerate(recs)])
        
        for idx, (tab, hw) in enumerate(zip(rec_tabs, recs)):
            with tab:
                col_info, col_score = st.columns([3, 1])
                
                with col_info:
                    st.markdown(f"### {hw['manufacturer']} {hw['name']}")
                    st.markdown(f"**Category:** {hw['category']} | **Region:** {hw['use_location']}")
                
                with col_score:
                    score = hw['score']
                    color = "green" if score >= 80 else "blue" if score >= 60 else "orange"
                    
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=score,
                        title={'text': "Score"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 40], 'color': "#ffcccc"},
                                {'range': [40, 70], 'color': "#ffffcc"},
                                {'range': [70, 100], 'color': "#ccffcc"},
                            ],
                        }
                    ))
                    fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{idx}")
                
                st.divider()
                
                st.markdown("#### 🖥️ Specs")
                specs_col1, specs_col2, specs_col3 = st.columns(3)
                
                with specs_col1:
                    mem = hw['memory']
                    if isinstance(mem, str) and mem != 'N/A':
                        mem = f"{mem} GB"
                    elif mem != 'N/A':
                        mem = f"{mem} GB"
                    st.metric("Memory", mem if mem != 'N/A' else "N/A")
                
                with specs_col2:
                    cpus = hw['number_cpu']
                    st.metric("CPUs", cpus if cpus != 'N/A' else "N/A")
                
                with specs_col3:
                    height = hw['height']
                    if height != 'N/A':
                        height = f"{height} U"
                    st.metric("Height", height if height != 'N/A' else "N/A")
                
                st.divider()
                
                st.markdown("#### 🌍 Metrics")
                sus_col1, sus_col2, sus_col3, sus_col4 = st.columns(4)
                
                with sus_col1:
                    st.metric("GWP Total", f"{hw['gwp_total']:.0f} kgCO₂eq")
                
                with sus_col2:
                    st.metric("Yearly Energy", f"{hw['yearly_tec']:.0f} kWh")
                
                with sus_col3:
                    st.metric("Lifetime", f"{hw['lifetime']} years")
                
                with sus_col4:
                    lifecycle = hw['gwp_total'] + (hw['yearly_tec'] * hw['lifetime'] / 1000)
                    st.metric("Lifecycle Impact", f"{lifecycle:.0f} kgCO₂eq")
                
                st.divider()
                
                st.markdown("#### 📊 Breakdown")
                breakdown = {
                    "Phase": ["Manufacturing", "Use"],
                    "Ratio": [hw['gwp_manufacturing_ratio'] * 100, hw['gwp_use_ratio'] * 100]
                }
                fig_pie = px.pie(breakdown, names="Phase", values="Ratio", 
                                color_discrete_sequence=["#FF6B6B", "#4ECDC4"])
                st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{idx}")
                
                st.divider()
                st.markdown("#### 💡 Why?")
                st.info(hw['reasoning'])

    
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
            fig_gwp.update_layout(height=400, showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig_gwp, use_container_width=True, key="gwp_chart")
        
        with col2:
            st.markdown("### Yearly Energy")
            tec_df = pd.DataFrame(comp['energy_comparison'])
            fig_tec = px.bar(tec_df, x='name', y='yearly_tec', color='yearly_tec',
                            color_continuous_scale='Blues',
                            labels={'yearly_tec': 'Energy (kWh)', 'name': ''})
            fig_tec.update_layout(height=400, showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig_tec, use_container_width=True, key="tec_chart")
        
        st.divider()
        
        st.markdown("### Lifecycle Impact")
        lifecycle_df = pd.DataFrame(comp['lifecycle_cost'])
        fig_lifecycle = px.bar(lifecycle_df, x='name', y='total_emissions', 
                              color='total_emissions', color_continuous_scale='YlOrRd',
                              labels={'total_emissions': 'Total (kgCO₂eq)', 'name': ''})
        fig_lifecycle.update_layout(height=400, showlegend=False, xaxis_tickangle=-45)
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

    
    # ============ TAB 4: EXPORT ============
    with tab4:
        st.markdown("## Export Results")
        
        st.markdown("### 📥 Download Options")
        
        # JSON export
        json_str = json.dumps(result, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Download as JSON",
            data=json_str,
            file_name=f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        # CSV export
        table_data = []
        for hw in recs:
            table_data.append({
                'Manufacturer': hw['manufacturer'],
                'Model': hw['name'],
                'Category': hw['category'],
                'GWP_Total_kgCO2eq': hw['gwp_total'],
                'Yearly_Energy_kWh': hw['yearly_tec'],
                'Lifetime_Years': hw['lifetime'],
                'Memory_GB': hw['memory'],
                'CPUs': hw['number_cpu'],
                'Height_U': hw['height'],
                'Region': hw['use_location'],
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
        
        st.markdown("### Preview")
        st.dataframe(df_export, use_container_width=True, hide_index=True)

# ============ FOOTER ============
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    🌍 Eco-IT Hardware Recommender v1.0 | Built with Streamlit & FastAPI
    <br/>
    Made with ❤️ for sustainable IT infrastructure
</div>
""", unsafe_allow_html=True)
