from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from catboost import CatBoostClassifier
from catboost import Pool

import pandas as pd

from app.preprocessing import preprocess_input

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")

model = CatBoostClassifier()
model.load_model("app/model/best_catboost_optuna_model.cbm")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request
        }
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,

    permalink: str = Form(...),
    name: str = Form(...),
    homepage_url: str = Form(...),

    category_list: str = Form(...),
    funding_total_usd: str = Form(...),

    country_code: str = Form(...),
    state_code: str = Form(...),
    region: str = Form(...),
    city: str = Form(...),

    funding_rounds: str = Form(...),

    founded_at: str = Form(...),
    first_funding_at: str = Form(...),
    last_funding_at: str = Form(...)
):

    raw_input = {
        "permalink": permalink,
        "name": name,
        "homepage_url": homepage_url,
        "category_list": category_list,
        "funding_total_usd": funding_total_usd,
        "country_code": country_code,
        "state_code": state_code,
        "region": region,
        "city": city,
        "funding_rounds": funding_rounds,
        "founded_at": founded_at,
        "first_funding_at": first_funding_at,
        "last_funding_at": last_funding_at
    }

    df = preprocess_input(raw_input)

    cat_feature_names = [
        "country_code",
        "state_code",
        "region",
        "city",
        "main_category"
    ]

    pool = Pool(
        df,
        cat_features=cat_feature_names
    )

    prediction = model.predict(pool)[0]

    probability = model.predict_proba(pool)[0][1]

    result = "Successful" if prediction == 1 else "Unsuccessful"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "prediction": result,
            "probability": round(probability * 100, 2)
        }
    )