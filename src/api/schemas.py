from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float] = Field(min_length=1)


class PredictResponse(BaseModel):
    prediction: int
    score: float | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_uri: str | None = None


class ReloadResponse(BaseModel):
    status: str
    model_uri: str