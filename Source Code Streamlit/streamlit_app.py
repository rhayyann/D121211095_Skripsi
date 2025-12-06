import streamlit as st
st.set_page_config(layout="wide")

# CORRECTED Custom CSS for the sidebar selected item
st.markdown("""
<style>
div[data-testid="stSidebarNavItems"] a[aria-current="page"] {
    background-color: #008940;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# The rest of your app code remains the same
st.logo("elements/logo-kalla-toyota.png", size="large")

# st.title("🚗 Vehicle Maintenance Dashboard")
# st.write("Welcome to the Maintenance Dashboard app. Use the sidebar to navigate to different pages.")

maintenance_page = st.Page("maintenance.py", title="Maintenance", icon=":material/build:")
diagnostic_page = st.Page("diagnostic.py", title="Diagnostic", icon=":material/car_crash:")

pg = st.navigation([maintenance_page, diagnostic_page])
pg.run()