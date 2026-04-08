import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List
import uvicorn

from src.algorithms.autocorrelation import AutocorrelationTracker
from src.algorithms.state_space import StateSpaceTracker

app = FastAPI(title="Beat Tracking API")

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_header(request, call_next):
    response = await call_next(request)
    if "static" in request.url.path:
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return RedirectResponse(url="/static/index.html")

class AnalysisResult(BaseModel):
    algorithm: str
    tempo: float
    beat_times: List[float]

@app.post("/analyze/{algorithm}", response_model=AnalysisResult)
async def analyze_audio(algorithm: str, file: UploadFile = File(...)):
    """
    Analyze an uploaded audio file using the selected algorithm.
    """
    if algorithm not in ["autocorrelation", "state_space"]:
        raise HTTPException(
            status_code=400, 
            detail="Invalid algorithm. Choose 'autocorrelation' or 'state_space'. DBN is disabled."
        )
    
    # Save the file to a temporary location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] if file.filename else ".wav") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")

    try:
        # Initialize the appropriate tracker
        if algorithm == "autocorrelation":
            tracker = AutocorrelationTracker()
        else:
            tracker = StateSpaceTracker()
        
        # Run prediction
        tempo, beat_times = tracker.predict(tmp_path)
        
        return AnalysisResult(
            algorithm=algorithm,
            tempo=float(tempo),
            beat_times=beat_times.tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
