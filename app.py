from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference import predict_startup


app = FastAPI(
    title="Startup Success Prediction API",
    description="Predict whether a startup is likely to succeed based on funding and location data.",
    version="1.0.0",
)


class StartupInput(BaseModel):
    main_category: str = Field(..., examples=["Software"])
    country_code: str = Field(..., examples=["USA"])
    state_code: str = Field(..., examples=["CA"])
    region: str = Field(..., examples=["SF Bay Area"])
    city: str = Field(..., examples=["San Francisco"])
    funding_total_usd: float = Field(..., ge=0)
    funding_rounds: int = Field(..., ge=0)
    founded_at: str = Field(..., examples=["2010-01-01"])
    first_funding_at: str = Field(..., examples=["2011-06-01"])
    last_funding_at: str = Field(..., examples=["2013-09-15"])


class PredictionResponse(BaseModel):
    prediction: int
    predicted_label: str
    success_probability: float
    failure_probability: float


@app.get("/")
def root() -> dict:
    return {
        "message": "Startup Success Prediction API",
        "docs_url": "/docs",
        "predict_url": "/predict",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: StartupInput) -> PredictionResponse:
    try:
        result = predict_startup(payload.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return PredictionResponse(**result)
