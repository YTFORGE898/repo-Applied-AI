import os
from datetime import date

import requests
import streamlit as st


API_URL = os.getenv(
    "FASTAPI_URL",
    "http://api:8000"
)


st.set_page_config(
    page_title="Startup Success Predictor"
)

st.title("Startup Success Predictor")

st.write(
    """
    Enter startup information below and the model will
    estimate the probability of success.
    """
)

with st.form("startup_form"):
    category_list = st.text_input("Category list", value="Software")
    country_code = st.text_input("Country code", value="USA")
    state_code = st.text_input("State code", value="CA")
    region = st.text_input("Region", value="SF Bay Area")
    city = st.text_input("City", value="San Francisco")
    funding_total_usd = st.number_input("Funding total USD", min_value=0.0, value=5000000.0)
    funding_rounds = st.number_input("Funding rounds", min_value=0, value=2, step=1)
    founded_at = st.date_input("Founded at", value=date(2010, 1, 1))
    first_funding_at = st.date_input("First funding at", value=date(2011, 6, 1))
    last_funding_at = st.date_input("Last Funding Date", value=date(2013, 9, 15))
    submitted = st.form_submit_button("Predict")

if submitted:
    errors = []
    if not category_list.strip():
        errors.append("Category list is required.")

    if not country_code.strip():
        errors.append("Country code is required.")

    if not state_code.strip():
        errors.append("State code is required.")

    if not region.strip():
        errors.append("Region is required.")

    if not city.strip():
        errors.append("City is required.")

    if first_funding_at < founded_at:
        errors.append("First funding date cannot be before founded date.")

    if last_funding_at < first_funding_at:
        errors.append("Last funding date cannot be before first funding date.")

    if errors:
        for error in errors:
            st.error(error)
    else:
        payload = {
            "category_list": category_list,
            "country_code": country_code,
            "state_code": state_code,
            "region": region,
            "city": city,
            "funding_total_usd": float(funding_total_usd),
            "funding_rounds": int(funding_rounds),
            "founded_at": founded_at.isoformat(),
            "first_funding_at": first_funding_at.isoformat(),
            "last_funding_at": last_funding_at.isoformat(),
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)

            if response.status_code != 200:
                detail = response.json().get("detail", "Unknown API error")
                st.error(detail)
            else:
                result = response.json()
                st.success("Prediction completed successfully.")
                st.subheader("Prediction")
                st.write(f"Predicted Label: **{result['predicted_label']}**")
                st.metric("Success Probability", f"{result['success_probability']:.2%}")
                st.metric("Failure Probability", f"{result['failure_probability']:.2%}")
                st.json(result)

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI backend.")

        except requests.exceptions.Timeout:
            st.error("The prediction request timed out.")

        except Exception as exc:
            st.error(f"Unexpected error: {str(exc)}")