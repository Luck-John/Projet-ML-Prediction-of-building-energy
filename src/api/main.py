"""
FastAPI Endpoint - Building Energy Consumption Prediction API
Uses the generic PredictionService for maximum flexibility
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from src.models.prediction_service import PredictionService

app = FastAPI(
    title="Building Energy Prediction API",
    description="Predict building energy consumption for Seattle non-residential buildings",
    version="1.0"
)

# Initialize services (one for each scenario)
services = {
    True: PredictionService(use_energy_star=True),
    False: PredictionService(use_energy_star=False)
}


# ============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# ============================================================================

class BuildingRecord(BaseModel):
    """Input building record for single prediction"""
    PrimaryPropertyType: str
    BuildingType: str
    PropertyGFATotal: float
    YearBuilt: int
    Latitude: float
    Longitude: float
    Neighborhood: str
    LargestPropertyUseType: str
    ListOfAllPropertyUseTypes: str
    ENERGYSTARScore: Optional[float] = None


class PredictionResponse(BaseModel):
    """Response for single prediction"""
    prediction_kbtu: float
    prediction_log: float
    model_type: str
    unit: str


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    models_available: Dict[str, bool]
    message: str


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - Basic info"""
    return {
        "name": "Building Energy Prediction API",
        "version": "1.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "predict_single": "/predict",
            "model_info": "/model-info"
        }
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """Check API health and model availability"""
    return {
        "status": "operational",
        "models_available": {
            "with_energy_star": True,
            "without_energy_star": True
        },
        "message": "All models loaded and ready"
    }


@app.get("/model-info", tags=["Info"])
async def model_info(use_energy_star: bool = Query(True)):
    """Get information about a specific model"""
    try:
        service = services[use_energy_star]
        info = service.get_model_info()
        return {
            "success": True,
            "model_info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    record: BuildingRecord,
    use_energy_star: bool = Query(True, description="Use model with ENERGY STAR Score")
):
    """
    Predict energy consumption for a single building
    
    **Parameters:**
    - `use_energy_star`: Whether to use the model trained with ENERGY_STAR Score
    - `record`: Building information (see schema)
    
    **Returns:**
    - `prediction_kbtu`: Predicted consumption in kBtu
    - `prediction_log`: Log-transformed prediction
    - `model_type`: Which scenario was used
    """
    try:
        service = services[use_energy_star]
        
        # Convert Pydantic model to dict
        record_dict = record.dict(exclude_none=True)
        
        # Get prediction
        result = service.predict_single(record_dict)
        
        return result
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")


@app.post("/predict-batch", tags=["Prediction"])
async def predict_batch(
    records: List[BuildingRecord],
    use_energy_star: bool = Query(True)
):
    """
    Predict energy consumption for multiple buildings
    
    **Parameters:**
    - `records`: List of building records
    - `use_energy_star`: Use model with ENERGY STAR Score
    
    **Returns:**
    - List of predictions with input data
    """
    try:
        service = services[use_energy_star]
        
        # Convert to list of dicts
        records_list = [r.dict(exclude_none=True) for r in records]
        
        # Get predictions
        results_df = service.predict_batch(records_list)
        
        # Convert to list of dicts for JSON response
        return {
            "success": True,
            "count": len(results_df),
            "predictions": results_df[['prediction_kbtu', 'prediction_log', 'model_type']].to_dict('records')
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {e}")


@app.get("/required-columns", tags=["Info"])
async def required_columns():
    """Get list of required columns for predictions"""
    try:
        service = services[True]
        cols = service.get_required_columns()
        return {
            "required_columns": cols,
            "total": len(cols)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("🚀 Building Energy Prediction API Starting...")
    print("✅ Models loaded:")
    print("   - With ENERGY STAR Score (MAPE=0.4041)")
    print("   - Without ENERGY STAR Score (MAPE=0.4950)")
    print("📚 Documentation available at /docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🛑 Shutting down API")


# ============================================================================
# CUSTOM EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc)
        }
    )


# ============================================================================
# EXAMPLES FOR TESTING (Interactive Swagger UI at /docs)
# ============================================================================

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
