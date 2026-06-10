from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from datetime import datetime

from inference import predict_startup


app = FastAPI(
    title="Startup Success Prediction API",
    description="Predict whether a startup is likely to succeed based on funding and location data.",
    version="1.0.0",
)


class StartupInput(BaseModel):
    category_list: str = Field(..., min_length=1)
    country_code: str = Field(..., min_length=1)
    state_code: str = Field(..., min_length=1)
    region: str = Field(..., min_length=1)
    city: str = Field(..., min_length=1)
    funding_total_usd: float = Field(..., ge=0)
    funding_rounds: int = Field(..., ge=0)
    founded_at: str
    first_funding_at: str
    last_funding_at: str
    @validator(
        "founded_at",
        "first_funding_at",
        "last_funding_at"
    )
    def validate_date(cls, value):
        try:
            datetime.strptime(value, "%Y-%m-%d")
            return value
        except ValueError:
            raise ValueError(
                "Dates must use YYYY-MM-DD format"
            )


class PredictionResponse(BaseModel):
    prediction: int
    predicted_label: str
    success_probability: float
    failure_probability: float


@app.get("/")
def root():
    return {
        "message": "Startup Success Prediction API",
        "docs_url": "/docs",
        "predict_url": "/predict"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: StartupInput):
    founded = datetime.strptime(payload.founded_at, "%Y-%m-%d")
    first_funding = datetime.strptime(payload.first_funding_at, "%Y-%m-%d")
    last_funding = datetime.strptime(payload.last_funding_at, "%Y-%m-%d")

    if first_funding < founded:
        raise HTTPException(status_code=400, detail="First funding date cannot be before founded date.")
    
    if last_funding < first_funding:
        raise HTTPException(status_code=400, detail="Last funding date cannot be before first funding date.")

    try:
        result = predict_startup(payload.model_dump())
        return PredictionResponse(**result)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}")