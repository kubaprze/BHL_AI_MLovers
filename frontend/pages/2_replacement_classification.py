import streamlit as st

st.set_page_config(
    page_title="New Function",
    page_icon="🆕",
    layout="wide"
)

# Your dark mode CSS (same as main app)
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0E1117 !important;
        color: #E6EDF3 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #00D9FF !important;
    }
    </style>
""", unsafe_allow_html=True)

# Your page content here
st.markdown("# 🆕 New Function")
st.markdown("## Add your content here")

# Example content
st.write("This is a separate page!")

if st.button("Click me"):
    st.success("Button clicked!")
