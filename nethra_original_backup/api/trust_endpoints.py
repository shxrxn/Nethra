"""
Trust Management Endpoints - Trust scoring and behavioral analysis
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

from utils.security_utils import SecurityUtils

router = APIRouter()
security = HTTPBearer()
logger = logging.getLogger(__name__)

class TrustAnalysisRequest(BaseModel):
    session_id: str
    time_range: Optional[str] = "24h"

class ChallengeResponse(BaseModel):
    challenge_id: str
    user_response: str
    response_time: float

class MirageInteraction(BaseModel):
    mirage_id: str
    interaction_type: str
    element_id: str
    action: str
    response_time: float
    success: bool

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from token"""
    try:
        user_info = SecurityUtils.validate_token(credentials.credentials)
        return user_info
    except Exception as e:
        logger.error(f"Token validation failed: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid authentication token")

@router.get("/analysis/{session_id}")
async def get_trust_analysis(
    session_id: str,
    user_info: dict = Depends(get_current_user)
):
    """Get detailed trust analysis for a session"""
    try:
        # Get trust analytics
        from main import trust_service
        trust_analytics = await trust_service.get_trust_analytics(session_id)
        
        # Get behavioral insights
        from main import behavioral_analyzer
        behavioral_insights = await behavioral_analyzer.get_behavioral_insights(
            user_info['user_id'],
            session_id
        )
        
        # Get tamper report
        from main import tamper_detector
        tamper_report = await tamper_detector.get_tamper_report(
            user_info['user_id'],
            session_id
        )
        
        return {
            "session_id": session_id,
            "trust_analytics": trust_analytics,
            "behavioral_insights": behavioral_insights,
            "tamper_report": tamper_report,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trust analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Analysis failed")

@router.get("/history/{user_id}")
async def get_trust_history(
    user_id: str,
    limit: int = 50,
    user_info: dict = Depends(get_current_user)
):
    """Get trust history for user"""
    try:
        # Validate user access
        if user_info['user_id'] != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get tamper history
        from main import tamper_detector
        tamper_history = await tamper_detector.get_tamper_history(user_id, limit)
        
        # Get mirage analytics
        from main import mirage_controller
        mirage_analytics = await mirage_controller.get_mirage_analytics(user_id)
        
        return {
            "user_id": user_id,
            "tamper_history": tamper_history,
            "mirage_analytics": mirage_analytics,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get trust history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve history")

@router.post("/challenge/respond")
async def respond_to_challenge(
    response: ChallengeResponse,
    user_info: dict = Depends(get_current_user)
):
    """Respond to cognitive challenge"""
    try:
        # Validate and process challenge response
        # This is a simplified implementation
        
        # Calculate authenticity score based on response
        authenticity_score = 0.8  # Simplified scoring
        
        if response.response_time < 0.5:  # Too fast
            authenticity_score -= 0.3
        elif response.response_time > 30:  # Too slow
            authenticity_score -= 0.2
        
        # Determine if challenge passed
        challenge_passed = authenticity_score > 0.6
        
        # Log challenge response
        logger.info(f"Challenge response from {user_info['user_id']}: {challenge_passed}")
        
        return {
            "challenge_id": response.challenge_id,
            "passed": challenge_passed,
            "authenticity_score": authenticity_score,
            "next_action": "CONTINUE" if challenge_passed else "ADDITIONAL_VERIFICATION"
        }
        
    except Exception as e:
        logger.error(f"Challenge response failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Challenge response failed")

@router.post("/mirage/interact")
async def record_mirage_interaction(
    interaction: MirageInteraction,
    user_info: dict = Depends(get_current_user)
):
    """Record interaction with mirage interface"""
    try:
        # Record interaction
        from main import mirage_controller
        success = await mirage_controller.record_mirage_interaction(
            interaction.mirage_id,
            interaction.dict()
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Mirage not found")
        
        return {
            "mirage_id": interaction.mirage_id,
            "interaction_recorded": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Mirage interaction recording failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Interaction recording failed")

@router.get("/mirage/{mirage_id}/status")
async def get_mirage_status(
    mirage_id: str,
    user_info: dict = Depends(get_current_user)
):
    """Get mirage status"""
    try:
        from main import mirage_controller
        mirage_status = await mirage_controller.get_mirage_status(mirage_id)
        
        if not mirage_status:
            raise HTTPException(status_code=404, detail="Mirage not found")
        
        # Validate user access
        if mirage_status['user_id'] != user_info['user_id']:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return {
            "mirage_id": mirage_id,
            "status": mirage_status['status'],
            "type": mirage_status['mirage_type'],
            "created_at": mirage_status['created_at'],
            "expires_at": mirage_status['expires_at'],
            "interactions_count": len(mirage_status['interactions'])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get mirage status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get mirage status")

@router.post("/mirage/{mirage_id}/deactivate")
async def deactivate_mirage(
    mirage_id: str,
    user_info: dict = Depends(get_current_user)
):
    """Deactivate mirage interface"""
    try:
        # Verify mirage ownership
        from main import mirage_controller
        mirage_status = await mirage_controller.get_mirage_status(mirage_id)
        
        if not mirage_status:
            raise HTTPException(status_code=404, detail="Mirage not found")
        
        if mirage_status['user_id'] != user_info['user_id']:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Deactivate mirage
        success = await mirage_controller.deactivate_mirage(mirage_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to deactivate mirage")
        
        return {
            "mirage_id": mirage_id,
            "deactivated": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Mirage deactivation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Mirage deactivation failed")

@router.get("/baseline/{user_id}")
async def get_user_baseline(
    user_id: str,
    user_info: dict = Depends(get_current_user)
):
    """Get user's behavioral baseline"""
    try:
        # Validate user access
        if user_info['user_id'] != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get baseline from behavioral analyzer
        from main import behavioral_analyzer
        baseline = await behavioral_analyzer._get_user_baseline(user_id)
        
        return {
            "user_id": user_id,
            "baseline": baseline,
            "privacy_compliant": True,
            "last_updated": baseline.get('last_updated', datetime.utcnow().isoformat())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user baseline: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get baseline")

@router.get("/metrics/summary")
async def get_trust_metrics_summary(
    user_info: dict = Depends(get_current_user)
):
    """Get trust metrics summary"""
    try:
        # Get user's trust metrics
        user_id = user_info['user_id']
        
        # Get active sessions
        from main import trust_service
        active_sessions = await trust_service.get_user_sessions(user_id)
        
        # Calculate summary metrics
        total_sessions = len(active_sessions)
        average_trust = 0
        
        if total_sessions > 0:
            trust_scores = []
            for session_id in active_sessions:
                session_status = await trust_service.get_session_status(session_id)
                if session_status:
                    trust_scores.append(session_status['trust_index'])
            
            average_trust = sum(trust_scores) / len(trust_scores) if trust_scores else 0
        
        return {
            "user_id": user_id,
            "total_sessions": total_sessions,
            "average_trust_score": round(average_trust, 2),
            "trust_trend": "STABLE",  # Simplified
            "security_level": "HIGH" if average_trust > 80 else "MEDIUM" if average_trust > 60 else "LOW",
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get trust metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")