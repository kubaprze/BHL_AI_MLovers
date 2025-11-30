import streamlit as st
import requests

API_URL = "http://localhost:8000/issues"  # adjust if needed

st.title("System wspomagania wymiany sprzętu")

# ------------------- DEVICES ----------------------
st.header("Add Device")
with st.form("add_device_form"):
    dev_name = st.text_input("Device name")
    submitted = st.form_submit_button("Add device")
    if submitted:
        if dev_name:
            r = requests.post(f"{API_URL}/add_device/{dev_name}")
            if r.status_code == 200:
                st.success("Device added!")
            else:
                st.error(f"Error: {r.text}")
        else:
            st.warning("Enter device name")

st.header("List Devices")
if st.button("Refresh device list"):
    r = requests.get(f"{API_URL}/get_devices")
    if r.status_code == 200:
        st.table(r.json())
    else:
        st.error(f"Error: {r.text}")

# ------------------- ISSUES ----------------------
st.header("Add Issue")
with st.form("add_issue_form"):
    #device_id = st.text_input("Device ID")
    ids = []
    labels = []
    try:
        r = requests.get(f"{API_URL}/get_devices")
        if r.status_code == 200:
            #ids = [krotka["id"] for krotka in r.json()]
            #labels = [krotka["device_name"] for krotka in r.json()]
            options = {krotka["id"]:krotka["device_name"] for krotka in r.json()}
        else:
            st.error(f"Error: {r.text}")
    except Exception as e:
        st.error(f"Error: {e}")
    device_id = st.selectbox(label="Choose device",options=list(options.keys()), format_func=options.get)
    description = st.text_area("Issue description")
    submitted_issue = st.form_submit_button("Submit issue")
    if submitted_issue:
        if device_id and description:
            payload = {
                "device_id": str(device_id),
                "description": description
            }
            r = requests.post(f"{API_URL}/add_issue", json=payload)
            if r.status_code == 200:
                st.success("Issue submitted!")
            else:
                st.error(f"Error: {r.text}")
        else:
            st.warning("Fill all fields")

# ------------------- WORST ISSUES ----------------------
st.header("Worst Issues (Top 5)")
if st.button("Load worst issues"):
    r = requests.get(f"{API_URL}/get_worst_issues")
    if r.status_code == 200:
        #st.json(r.json())
        st.table(r.json())
    else:
        st.error(f"Error: {r.text}")
