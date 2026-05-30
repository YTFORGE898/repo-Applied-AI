import os

import requests
import streamlit as st


API_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")


st.set_page_config(page_title="Startup Success Predictor", page_icon=":chart_with_upwards_trend:")
st.title("Startup Success Predictor")
st.caption("Small demo frontend for the FastAPI backend.")

with st.form("startup_form"):
    main_category = st.text_input("Main category", value="Software")
    country_code = st.text_input("Country code", value="USA")
    state_code = st.text_input("State code", value="CA")
    region = st.text_input("Region", value="SF Bay Area")
    city = st.text_input("City", value="San Francisco")
    funding_total_usd = st.number_input("Funding total USD", min_value=0.0, value=5000000.0)
    funding_rounds = st.number_input("Funding rounds", min_value=0, value=2, step=1)
    founded_at = st.text_input("Founded at", value="2010-01-01")
    first_funding_at = st.text_input("First funding at", value="2011-06-01")
    last_funding_at = st.text_input("Last funding at", value="2013-09-15")
    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "main_category": main_category,
        "country_code": country_code,
        "state_code": state_code,
        "region": region,
        "city": city,
        "funding_total_usd": funding_total_usd,
        "funding_rounds": int(funding_rounds),
        "founded_at": founded_at,
        "first_funding_at": first_funding_at,
        "last_funding_at": last_funding_at,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach FastAPI backend at {API_URL}: {exc}")
    else:
        st.subheader("Prediction")
        st.write(f"Label: `{result['predicted_label']}`")
        st.metric("Success probability", f"{result['success_probability']:.2%}")
        st.metric("Failure probability", f"{result['failure_probability']:.2%}")
        st.json(result)
