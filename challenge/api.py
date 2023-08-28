import fastapi
import pandas as pd
from model import DelayModel

app = fastapi.FastAPI()

delay_predictor = DelayModel()

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(data:dict) -> dict:
    try:
        #Convert incoming JSON data to a DataFrame
        incoming_data = pd.DataFrame(data)

        #Preprocess the incoming data to match the features used during training
        new_features, _ = delay_predictor.preprocess(incoming_data, target_column=None)

        #Make predictions using the DelayModel
        predictions = delay_predictor.predict(new_features)

        return {
            "predictions": predictions
        }
    except Exception as e:
        return {
            "error": str(e)
        }