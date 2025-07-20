"""
NETHRA Backend - FastAPI Application with Complete Feature Set
Behavior-Based Continuous Authentication for Mobile Banking
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import uuid
import hashlib
import os
from pathlib import Path

# Import database
from database.database import db_manager
from database.models import UserModel, SessionModel, TrustScoreModel, MirageModel, SecurityModel

# Import our services
from services.ai_interface import AIInterface
from services.trust_service import TrustService
from services.behavioral_analyzer import BehavioralAnalyzer
from services.tamper_detection import TamperDetector
from services.mirage_controller import MirageController
from services.encryption_service import EncryptionService
from services.monitoring_service import MonitoringService
from services.cache_service import CacheService
from services.rate_limiter import rate_limiter, start_rate_limiter_cleanup

# Import middleware
from middleware.rate_limit_middleware import RateLimitMiddleware

# Import API routers
from api.endpoints import router as api_router
from api.auth_endpoints import router as auth_router
from api.trust_endpoints import router as trust_router
from api.monitoring_endpoints import router as monitoring_router

# Import utilities
from utils.security_utils import SecurityUtils
from utils.privacy_utils import PrivacyUtils
from utils.performance_utils import PerformanceUtils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
ai_interface = None
trust_service = None
behavioral_analyzer = None
tamper_detector = None
mirage_controller = None
encryption_service = None
monitoring_service = None
cache_service = None

# Global memory store for services
memory_store = {
    "mirage_data": {},
    "active_mirages": {},
    "tamper_logs": {},
    "security_incidents": []
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global ai_interface, trust_service, behavioral_analyzer, tamper_detector, mirage_controller
    global encryption_service, monitoring_service, cache_service
    
    # Initialize database
    await db_manager.initialize_database()
    logger.info("Database initialized successfully")
    
    # Initialize encryption service
    encryption_service = EncryptionService()
    logger.info("Encryption service initialized")
    
    # Initialize cache service
    cache_service = CacheService()
    logger.info("Cache service initialized")
    
    # Initialize monitoring service
    monitoring_service = MonitoringService()
    logger.info("Monitoring service initialized")
    
    # Initialize AI interface
    model_path = Path(__file__).parent / "models" / "trust_model.tflite"
    ai_interface = AIInterface(model_path)
    logger.info("AI interface initialized")
    
    # Initialize services with dependencies
    trust_service = TrustService(ai_interface, db_manager, cache_service, encryption_service)
    behavioral_analyzer = BehavioralAnalyzer(ai_interface, db_manager, cache_service)
    tamper_detector = TamperDetector(memory_store)
    mirage_controller = MirageController(memory_store)
    
    # Start background tasks
    asyncio.create_task(start_rate_limiter_cleanup())
    
    logger.info("NETHRA Backend services initialized successfully")
    yield
    
    logger.info("NETHRA Backend services cleaned up")

# Create FastAPI app
app = FastAPI(
    title="NETHRA API",
    description="Behavior-Based Continuous Authentication for Mobile Banking",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure for production
)

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware)

# Security scheme
security = HTTPBearer()

# Request/Response models
class BehavioralData(BaseModel):
    """Behavioral data from mobile device"""
    user_id: str
    session_id: str
    timestamp: datetime
    touch_patterns: List[Dict] = Field(default_factory=list)
    swipe_patterns: List[Dict] = Field(default_factory=list)
    device_motion: Dict = Field(default_factory=dict)
    app_usage: Dict = Field(default_factory=dict)
    network_info: Dict = Field(default_factory=dict)
    location_context: Optional[Dict] = None

class TrustScoreResponse(BaseModel):
    """Trust score response"""
    trust_index: float = Field(..., ge=0, le=100)
    risk_level: str = Field(..., pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    behavioral_anomalies: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    session_valid: bool = True
    mirage_active: bool = False
    lockout_imminent: bool = False
    lockout_countdown: int = 0

class SessionStatus(BaseModel):
    """Session status response"""
    session_id: str
    user_id: str
    is_active: bool
    trust_index: float
    last_activity: datetime
    risk_factors: List[str] = Field(default_factory=list)
    mirage_active: bool = False
    is_locked: bool = False

class MirageResponse(BaseModel):
    """Mirage interface response"""
    mirage_id: str
    interface_type: str
    fake_elements: List[Dict] = Field(default_factory=list)
    cognitive_challenges: List[Dict] = Field(default_factory=list)
    duration_minutes: int = 30
    lockout_timer: int = 10

# Dependency injection
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate user token and return user info"""
    try:
        user_info = SecurityUtils.validate_token(credentials.credentials)
        return user_info
    except Exception as e:
        logger.error(f"Token validation failed: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid authentication token")

# Performance monitoring decorator
@PerformanceUtils.timing_decorator("behavioral_analysis")
async def analyze_behavioral_data_with_monitoring(data: BehavioralData, user_info: dict):
    """Analyze behavioral data with performance monitoring"""
    return await analyze_behavioral_data_core(data, user_info)

# Main API endpoints
@app.post("/api/v1/behavioral/analyze", response_model=TrustScoreResponse)
async def analyze_behavioral_data(
    data: BehavioralData,
    user_info: dict = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Analyze behavioral data and return trust score"""
    try:
        # Performance monitoring
        start_time = datetime.utcnow()
        
        if user_info['user_id'] != data.user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check if user exists and get baseline
        user_data = await UserModel.get_user(data.user_id)
        is_new_user = user_data is None
        
        if is_new_user:
            # Create new user with HIGH initial trust (better safe than sorry)
            await UserModel.create_user(data.user_id, data.app_usage or {})
            initial_trust = 95.0  # HIGH TRUST for new users
            logger.info(f"New user {data.user_id} created with 95% trust")
        else:
            # Existing user - analyze against baseline (LOWER trust for anomalies)
            trust_score = await behavioral_analyzer.analyze_behavior(data)
            initial_trust = trust_score.trust_index
            logger.info(f"Existing user {data.user_id} analyzed: {initial_trust}% trust")
        
        # Detect tampering
        tamper_detected = await tamper_detector.detect_tampering(data)
        
        if tamper_detected:
            initial_trust = 0.0
            risk_level = "CRITICAL"
            recommended_actions = ["IMMEDIATE_LOCKDOWN"]
            
            background_tasks.add_task(
                SecurityModel.log_security_incident,
                user_info['user_id'],
                "TAMPER_DETECTED",
                "CRITICAL",
                data.dict()
            )
        else:
            risk_level = "LOW" if initial_trust > 80 else "MEDIUM" if initial_trust > 50 else "HIGH"
            recommended_actions = []
        
        # Mirage activation logic (TRUST < 50%)
        mirage_active = False
        lockout_imminent = False
        lockout_countdown = 0
        
        if initial_trust < 50 and not tamper_detected:
            mirage_response = await mirage_controller.activate_mirage(
                data.user_id,
                data.session_id,
                initial_trust
            )
            mirage_active = True
            recommended_actions.append("MIRAGE_ACTIVATED")
            
            # Set lockout timer (10 seconds after mirage activation)
            lockout_imminent = True
            lockout_countdown = 10
            
            # Schedule automatic lockout after 10 seconds
            background_tasks.add_task(
                trust_service.schedule_lockout,
                data.session_id,
                10  # seconds - APPLICATION GETS LOCKED OUT
            )
            
            logger.warning(f"Mirage activated for {data.user_id}, lockout in 10 seconds")
        
        # Update session trust
        await trust_service.update_session_trust(
            data.session_id,
            initial_trust,
            risk_level
        )
        
        # Performance monitoring
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        await monitoring_service.record_metric("behavioral_analysis_time", processing_time)
        
        return TrustScoreResponse(
            trust_index=initial_trust,
            risk_level=risk_level,
            behavioral_anomalies=[] if is_new_user else trust_score.behavioral_anomalies,
            recommended_actions=recommended_actions,
            session_valid=initial_trust > 40,
            mirage_active=mirage_active,
            lockout_imminent=lockout_imminent,
            lockout_countdown=lockout_countdown
        )
        
    except Exception as e:
        logger.error(f"Behavioral analysis failed: {str(e)}")
        await monitoring_service.record_error("behavioral_analysis_error")
        raise HTTPException(status_code=500, detail="Analysis failed")

@app.get("/api/v1/session/{session_id}/status", response_model=SessionStatus)
async def get_session_status(
    session_id: str,
    user_info: dict = Depends(get_current_user)
):
    """Get current session status and trust level"""
    try:
        session_data = await SessionModel.get_session(session_id)
        
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session_data['user_id'] != user_info['user_id']:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check if session is locked
        is_locked = await trust_service.is_session_locked(session_id)
        
        return SessionStatus(
            **session_data, 
            risk_factors=[],
            is_locked=is_locked
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session status retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session status")

@app.post("/api/v1/session/{session_id}/lock")
async def lock_session(
    session_id: str,
    user_info: dict = Depends(get_current_user)
):
    """Lock session immediately - APPLICATION GETS LOCKED OUT"""
    try:
        success = await trust_service.lock_session(session_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        logger.warning(f"Session {session_id} LOCKED OUT by request")
        
        return {"message": "Session locked successfully", "locked": True}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session lock failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to lock session")

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "services": {
            "ai_interface": ai_interface is not None,
            "trust_service": trust_service is not None,
            "behavioral_analyzer": behavioral_analyzer is not None,
            "tamper_detector": tamper_detector is not None,
            "mirage_controller": mirage_controller is not None,
            "encryption_service": encryption_service is not None,
            "monitoring_service": monitoring_service is not None,
            "cache_service": cache_service is not None,
            "database": "sqlite"
        }
    }

# Include additional routers
app.include_router(api_router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1/auth")
app.include_router(trust_router, prefix="/api/v1/trust")
app.include_router(monitoring_router, prefix="/api/v1/monitoring")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat(),
            "path": request.url.path
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )