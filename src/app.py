from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from contextlib import asynccontextmanager
from pdf_export import generate_pdf
from pipeline import CrisisDetectionPipeline
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# --- 1. INITIALIZE FASTAPI APP WITH LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models
    load_models()
    yield
    # Shutdown: Cleanup (if needed)

app = FastAPI(
    title="MindGuard AI",
    description="Mental Health Crisis Detection System",
    lifespan=lifespan
)

# --- 2. PIPELINE & HISTORY ---
PIPELINE = None
PERF_REPORT = None
ANALYSIS_HISTORY = []


def load_models():
    global PIPELINE, PERF_REPORT
    try:
        PIPELINE = CrisisDetectionPipeline()   # Loads all models internally
        try:
            with open(os.path.join(ROOT_DIR, "outputs", "models", "model_report.txt"), "r") as f:
                PERF_REPORT = f.read()
        except FileNotFoundError:
            PERF_REPORT = "Run baseline.py to generate report."
        print("[OK] Pipeline loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load pipeline: {str(e)}")
        raise


# --- 3. PYDANTIC MODELS WITH VALIDATION ---
class AnalysisRequest(BaseModel):
    user_input: str
    age: int
    gender: str
    height: int
    weight: int
    sleep: int
    stress: int
    severity: int
    mood_score: int
    emotion: str
    lifestyle: str
    diagnosis: str
    medication: str
    therapy_type: str
    treatment_duration: int
    treatment_progress: int
    physical_activity: float
    
    @field_validator('user_input')
    @classmethod
    def validate_user_input(cls, v):
        if not v or not v.strip():
            raise ValueError('User input cannot be empty')
        if len(v) > 5000:
            raise ValueError('User input cannot exceed 5000 characters')
        return v
    
    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v < 1 or v > 120:
            raise ValueError('Age must be between 1 and 120')
        return v
    
    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        if v not in ['Male', 'Female', 'Other']:
            raise ValueError('Gender must be Male, Female, or Other')
        return v
    
    @field_validator('height')
    @classmethod
    def validate_height(cls, v):
        if v < 100 or v > 250:
            raise ValueError('Height must be between 100-250 cm')
        return v
    
    @field_validator('weight')
    @classmethod
    def validate_weight(cls, v):
        if v < 30 or v > 500:
            raise ValueError('Weight must be between 30-500 kg')
        return v
    
    @field_validator('sleep')
    @classmethod
    def validate_sleep(cls, v):
        if v < 0 or v > 12:
            raise ValueError('Sleep must be between 0-12 hours')
        return v
    
    @field_validator('stress')
    @classmethod
    def validate_stress(cls, v):
        if v < 0 or v > 10:
            raise ValueError('Stress level must be between 0-10')
        return v
    
    @field_validator('severity')
    @classmethod
    def validate_severity(cls, v):
        if v < 1 or v > 10:
            raise ValueError('Symptom severity must be between 1-10')
        return v
    
    @field_validator('mood_score')
    @classmethod
    def validate_mood_score(cls, v):
        if v < 1 or v > 10:
            raise ValueError('Mood score must be between 1-10')
        return v
    
    @field_validator('emotion')
    @classmethod
    def validate_emotion(cls, v):
        valid = ['Neutral', 'Happy', 'Sad', 'Anxious', 'Angry', 'Depressed']
        if v not in valid:
            raise ValueError(f'Emotion must be one of: {", ".join(valid)}')
        return v
    
    @field_validator('lifestyle')
    @classmethod
    def validate_lifestyle(cls, v):
        valid = ['Sedentary', 'Moderately Active', 'Active', 'Very Active']
        if v not in valid:
            raise ValueError(f'Lifestyle must be one of: {", ".join(valid)}')
        return v
    
    @field_validator('diagnosis')
    @classmethod
    def validate_diagnosis(cls, v):
        valid = ['Major Depressive Disorder', 'Panic Disorder', 'Generalized Anxiety', 
                 'Bipolar Disorder', 'Other', 'None']
        if v not in valid:
            raise ValueError(f'Diagnosis must be one of: {", ".join(valid)}')
        return v
    
    @field_validator('medication')
    @classmethod
    def validate_medication(cls, v):
        valid = ['Mood Stabilizers', 'Antipsychotics', 'SSRIs', 'Anxiolytics',
                 'Antidepressants', 'Benzodiazepines', 'Other', 'None']
        if v not in valid:
            raise ValueError(f'Medication must be one of: {", ".join(valid)}')
        return v
    
    @field_validator('therapy_type')
    @classmethod
    def validate_therapy_type(cls, v):
        valid = ['Interpersonal Therapy', 'Mindfulness-Based Therapy',
                 'Cognitive Behavioral Therapy', 'Dialectical Behavioral Therapy',
                 'Other', 'None']
        if v not in valid:
            raise ValueError(f'Therapy type must be one of: {", ".join(valid)}')
        return v
    
    @field_validator('treatment_duration')
    @classmethod
    def validate_treatment_duration(cls, v):
        if v < 0 or v > 10000:
            raise ValueError('Treatment duration must be between 0-10000 days')
        return v
    
    @field_validator('treatment_progress')
    @classmethod
    def validate_treatment_progress(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Treatment progress must be between 0-100%')
        return v
    
    @field_validator('physical_activity')
    @classmethod
    def validate_physical_activity(cls, v):
        if v < 0 or v > 24:
            raise ValueError('Physical activity must be between 0-24 hours/week')
        return v


class AnalysisResult(BaseModel):
    user_input: str
    age: int
    gender: str
    height: int
    weight: int
    sleep: int
    stress: int
    severity: int
    mood_score: int
    emotion: str
    lifestyle: str
    diagnosis: str
    medication: str
    therapy_type: str
    treatment_duration: int
    treatment_progress: int
    physical_activity: float
    confidence: float
    text_risk_score: float
    clinical_outcome: str
    risk_level: str
    recommendation: str
    timestamp: str
    language_detected: str = "english"


# --- 4. API ENDPOINTS ---

@app.get("/")
async def root():
    return FileResponse(os.path.join(ROOT_DIR, "static", "index.html"), media_type="text/html")


@app.get("/api/models-info")
async def get_models_info():
    return {"report": PERF_REPORT}


@app.post("/api/analyze")
async def analyze(request: AnalysisRequest):
    try:
        if not request.user_input.strip():
            raise HTTPException(status_code=400, detail="User input cannot be empty")

        if not PIPELINE:
            raise HTTPException(status_code=500, detail="Pipeline not loaded")

        # Use pipeline — handles text + demographics + combined score
        output = PIPELINE.predict_full_report(
            text_input           = request.user_input,
            age                  = request.age,
            gender               = request.gender,
            diagnosis            = request.diagnosis,
            sleep                = request.sleep,
            stress               = request.stress,
            severity             = request.severity,
            mood_score           = request.mood_score,
            emotion              = request.emotion,
            medication           = request.medication,
            therapy_type         = request.therapy_type,
            treatment_duration   = request.treatment_duration,
            treatment_progress   = request.treatment_progress,
            physical_activity    = request.physical_activity
        )

        result = {
            "user_input"         : request.user_input,
            "age"                : request.age,
            "gender"             : request.gender,
            "height"             : request.height,
            "weight"             : request.weight,
            "sleep"              : request.sleep,
            "stress"             : request.stress,
            "severity"           : request.severity,
            "mood_score"         : request.mood_score,
            "emotion"            : request.emotion,
            "lifestyle"          : request.lifestyle,
            "diagnosis"          : request.diagnosis,
            "medication"         : request.medication,
            "therapy_type"       : request.therapy_type,
            "treatment_duration" : request.treatment_duration,
            "treatment_progress" : request.treatment_progress,
            "physical_activity"  : request.physical_activity,
            "confidence"         : round(output["Combined Score"] * 100, 2),
            "text_risk_score"    : round(output["Text Risk Score"] * 100, 2),
            "clinical_outcome"   : output["Clinical Outcome"],
            "risk_level"         : output["Risk Level"],
            "recommendation"     : output["Recommendation"],
            "timestamp"       : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "language_detected": output.get("Language Detected", "english"),
        }

        ANALYSIS_HISTORY.insert(0, result)
        ANALYSIS_HISTORY[:] = ANALYSIS_HISTORY[:25]

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/api/history")
async def get_history():
    return {"history": ANALYSIS_HISTORY}


@app.post("/api/delete-history/{index}")
async def delete_history(index: int):
    try:
        if 0 <= index < len(ANALYSIS_HISTORY):
            ANALYSIS_HISTORY.pop(index)
            return {"success": True, "message": "Item deleted"}
        else:
            raise HTTPException(status_code=404, detail="History item not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@app.post("/api/clear-history")
async def clear_history():
    ANALYSIS_HISTORY.clear()
    return {"success": True, "message": "History cleared"}


@app.post("/api/generate-pdf")
async def generate_pdf_endpoint(result: AnalysisResult):
    try:
        pdf_buffer = generate_pdf(
            result.user_input,
            result.age,
            result.gender,
            result.height,
            result.weight,
            result.sleep,
            result.lifestyle,
            result.confidence / 100,
            result.risk_level,
            result.recommendation
        )

        pdf_buffer.seek(0)
        filename = f"MindGuard_Report_{result.timestamp.replace(' ', '_').replace(':', '-')}.pdf"

        return StreamingResponse(
            iter([pdf_buffer.getvalue()]),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


# --- 5. MOUNT STATIC FILES ---
app.mount("/", StaticFiles(directory=os.path.join(ROOT_DIR, "static"), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)