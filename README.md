# Startup Success Prediction API

## Overview

Our API predicts whether a startup business is likely to become succesful based on its category of business, location, funding history and company age.

For training we considered a startup a success if it was acquired or IPO. We considered a startup a failure if it was closed.

The model was trained using CatBoost and a dataset containing the information mentioned before.

The API automatically performs the required preprocessing and feature engineering.

# Architecture

1. The user enters the data for every feature on the Streamlit frontend.

2. Streamlit sends a JSON request to the API.

3. FastAPI validates the request.

4. The preprocessing is used on the data.

5. The catboost model predicts the success of the startup.

6. The API returns a JSON response.

7. Streamlit displays the result.

# Activating the API

We use Docker for containerization and Streamlit to handle input and output.

To activate the API, open a terminal and navigate to the folder you have saved this GitHub repository in. Then, perform the following commands:

This command will navigate you to the folder the API is in.
```bash
cd deployment
```

This command will activate the API, which you can access via the URLs noted in the following section. To deactivate the API, you can press CTRL+C in the terminal that you activated the API from.
```bash
docker compose up --build
```

This command will make sure that the API is deactivated. Run this for safety after doing CTRL+C.
```bash
docker compose down
```

# Accessing the API

After activating the API as described in the previous section, these URLs will lead to the webpages that allow for usage of the API.

This is the most important URL. It leads to the Streamlit frontend, from where the relevant data can be input and a prediction can be received.
```text
http://localhost:8501
```

This URL is where the API itself is hosted.
```text
http://localhost:8000
```

This URL contains automatically generated documentation for the endpoints and schemas.
```text
http://localhost:8000/docs
```

# Endpoints

## POST /predict

Predicts whether a startup is likely to succeed. For most users this will be the only relevant endpoint.

### Input JSON

```json
{
    "main_category": string,
    "country_code": string,
    "state_code": string,
    "region": string,
    "city": string,
    "funding_total_usd": float,
    "funding_rounds": int,
    "founded_at": datetime,
    "first_funding_at": datetime,
    "last_funding_at": datetime
}
```

### Usage explanation

The information for these fields is input from the Streamlit URL. There will be input boxes for each of the features. funding_total_usd and funding_rounds can either be manually input or incremented with the plus and minus symbols on their input boxes. founded_at, first_funded_at and last_funded_at are input using a date selector. The other features are input manually. The datetime features do not support any dates past 2016. Invalid inputs will be caught by the frontend and displayed to the user.

main_category is the main category of business of a startup, such as Software, Games, Fashion, Analytics, Advertising etc.

country_code is the 3-letter ISO country code of the country a startup is located in, such as USA for the United States of America, FRA for France, NLD for the Netherlands, CHN for China etc.

state_code is the code of the state in which a startup is located. This can be either two letters or an integer, depending on the country. For example: IL for Illinois in the USA, or 22 for Qinghai in China.

region is the region in which the startup is located. For startups located in larger cities this can be the city itself. For startups in smaller cities, it can be the surrounding area, such as the SF Bay Area.

city is the city in which the startup is located.

funding_total_usd is the total amount of funding in US Dollars that the startup has received.

funding_rounds is the amount of times that a startup has received funding.

founded_at is the date on which the startup was founded.

first_funded_at is the date on which the startup first received funding.

last_funded_at is the date on which the startup last received funding.

### Response JSON

```json
{
    "prediction": int,
    "predicted_label": string,
    "success_probability": float,
    "failure_probability": float
}
```

prediction returns 0 for a failure and 1 for a success.
predicted_label returns "failure" or "success" and is what is displayed to the user.
success_probability returns a value from 0 to 1 representing the probability that the startup will be successful.
failure_probability returns a value from 0 to 1 representing the probability that the startup will fail.

# GET /

Displays basic information about the API.

## Response JSON

```json
{
    "message": "Startup Success Prediction API",
    "docs_url": "/docs",
    "predict_url" "/predict"
}
```

# GET /health

Returns the health of the API

## Response JSON

```json
{
    "health": "ok"
}
```