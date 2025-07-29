from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# Setup OpenTelemetry
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(CloudTraceSpanExporter()))
tracer = trace.get_tracer(__name__)

app = FastAPI(title="Iris Classifier API")
model = joblib.load("model.joblib")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict/")
def predict_species(data: IrisInput):
    with tracer.start_as_current_span("model_prediction") as span:
        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)[0]
        span.set_attribute("predicted_class", prediction)
        return {"predicted_class": prediction}