import os
from typing import List

import torch
from fastapi import FastAPI, HTTPException, Request, Header
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from model_architecture.hf import LogBertForSequenceClassification
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable is required")

# Rate limit configuration from environment variables
RATE_LIMIT_HEALTH = os.getenv("RATE_LIMIT_HEALTH", "60/minute")
RATE_LIMIT_FORWARD = os.getenv("RATE_LIMIT_FORWARD", "100/minute")

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class ForwardRequest(BaseModel):
    input_ids: List[List[int]]
    device_ids: List[List[int]]

class ForwardResponse(BaseModel):
    logits: List[List[float]]

@app.get("/health")
@limiter.limit(RATE_LIMIT_HEALTH)
async def health(request: Request):
    return {"status": "ok"}

@app.on_event("startup")
async def startup_event():
    model_dir = os.getenv("MODEL_DIR")
    if not model_dir:
        raise RuntimeError("MODEL_DIR env is required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LogBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    model.to(device)
    app.state.model = model
    app.state.device = device
    
@app.post("/forward", response_model=ForwardResponse)
@limiter.limit(RATE_LIMIT_FORWARD)
async def forward(request: Request, req: ForwardRequest, x_api_key: str = Header(None)):
    # Verify API key
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=503, detail="Model not loaded")
    device = app.state.device
    if len(req.input_ids) != len(req.device_ids):
        raise HTTPException(status_code=400, detail="input_ids and device_ids must have same batch size")
    input_ids = torch.tensor(req.input_ids, dtype=torch.long, device=device)
    device_ids = torch.tensor(req.device_ids, dtype=torch.long, device=device)
    with torch.no_grad():
        out = app.state.model(input_ids=input_ids, device_ids=device_ids)
        logits = out.logits.detach().cpu().tolist()
    return ForwardResponse(logits=logits)
