from fastapi import FastAPI, HTTPException
import uvicorn
import numpy as np
import torch
import multiprocessing
from model import BinaryClassifier
multiprocessing.set_start_method('spawn')

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or the URL where your React app is hosted
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to load the model (you might want to do this when starting the server)
def load_model():
    model_path = 'startup_predictor.pt'
    input_size = 34  # Replace with actual input size used during training
    hidden_layers = [5, 4]  # Replace with actual hidden layer configuration used during training
    model = BinaryClassifier(input_size, hidden_layers)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

model = load_model()

@app.post("/predict/")
async def predict(startup_data: dict):
    # Convert startup_data to the appropriate format for your model
    # For example, if you expect a NumPy array:
    startup_data = np.array([list(startup_data.values())])
    
    # Make prediction
    with torch.no_grad():
        inputs = torch.tensor(startup_data).float()
        outputs = model(inputs)
        probabilities = outputs.numpy() * 100  # Convert to percentage

    return {"probability": probabilities.tolist()}

# Run the API with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
