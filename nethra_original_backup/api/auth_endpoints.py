"""
Authentication Endpoints - User authentication and session management
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Dict, Optional
from datetime import datetime, timedelta
import logging

from utils.security_utils import SecurityUtils
from utils.privacy_utils import PrivacyUtils

router = APIRouter()
security = HTTPBearer()
logger = logging.getLogger(__name__)

class LoginRequest(BaseModel):
    user_id: str
    device_info: Dict
    biometric_data: Optional[Dict] = None

class LoginResponse(BaseModel):
    access_token: str
    session_id: str
    trust_index: float
    expires_at: datetime

class RefreshTokenRequest(BaseModel):
    refresh_token: str
    session_id: str

@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    background_tasks: BackgroundTasks
):
    """Authenticate user and create session"""
    try:
        # Validate user credentials (simplified for demo)
        if not request.user_id or len(request.user_id) < 3:
            raise HTTPException(status_code=400, detail="Invalid user ID")
        
        # Generate access token
        access_token = SecurityUtils.generate_token(
            request.user_id,
            expires_in=timedelta(hours=24)
        )
        
        # Create session
        from main import trust_service
        session_id = await trust_service.create_session(
            request.user_id,
            request.device_info
        )
        
        # Initial trust score (high for legitimate login)
        initial_trust = 90.0
        
        # Log login event
        background_tasks.add_task(
            SecurityUtils.log_security_incident,
            request.user_id,
            "USER_LOGIN",
            {
                "session_id": session_id,
                "device_info": request.device_info,
                "initial_trust": initial_trust
            }
        )
        
        return LoginResponse(
            access_token=access_token,
            session_id=session_id,
            trust_index=initial_trust,
            expires_at=datetime.utcnow() + timedelta(hours=24)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Login failed")

@router.post("/refresh")
async def refresh_token(
    request: RefreshTokenRequest,
    background_tasks: BackgroundTasks
):
    """Refresh access token"""
    try:
        # Validate refresh token (simplified for demo)
        user_info = SecurityUtils.validate_token(request.refresh_token)
        
        # Validate session
        from main import trust_service
        session_valid = await trust_service.validate_session(request.session_id)
        
        if not session_valid:
            raise HTTPException(status_code=401, detail="Invalid session")
        
        # Generate new access token
        new_token = SecurityUtils.generate_token(
            user_info['user_id'],
            expires_in=timedelta(hours=24)
        )
        
        # Log token refresh
        background_tasks.add_task(
            SecurityUtils.log_security_incident,
            user_info['user_id'],
            "TOKEN_REFRESH",
            {"session_id": request.session_id}
        )
        
        return {
            "access_token": new_token,
            "expires_at": datetime.utcnow() + timedelta(hours=24)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Token refresh failed")

@router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Logout user and terminate session"""
    try:
        # Validate token
        user_info = SecurityUtils.validate_token(credentials.credentials)
        
        # Get user sessions
        from main import trust_service
        user_sessions = await trust_service.get_user_sessions(user_info['user_id'])
        
        # Terminate all sessions
        for session_id in user_sessions:
            await trust_service.terminate_session(session_id)
        
        # Log logout event
        background_tasks.add_task(
            SecurityUtils.log_security_incident,
            user_info['user_id'],
            "USER_LOGOUT",
            {"terminated_sessions": len(user_sessions)}
        )
        
        return {"message": "Logout successful"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Logout failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Logout failed")

@router.get("/validate")
async def validate_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Validate access token"""
    try:
        user_info = SecurityUtils.validate_token(credentials.credentials)
        return {
            "valid": True,
            "user_id": user_info['user_id'],
            "expires_at": user_info['expires_at']
        }
        
    except Exception as e:
        logger.error(f"Token validation failed: {str(e)}")
        return {"valid": False}

@router.get("/user/profile")
async def get_user_profile(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get user profile information"""
    try:
        user_info = SecurityUtils.validate_token(credentials.credentials)
        
        # Get user's behavioral profile (simplified)
        return {
            "user_id": user_info['user_id'],
            "profile_created": datetime.utcnow().isoformat(),
            "trust_history": {
                "average_trust": 85.5,
                "sessions_count": 25,
                "anomalies_detected": 2
            },
            "security_settings": {
                "mirage_enabled": True,
                "tamper_detection": True,
                "privacy_mode": True
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user profile: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get profile")

@router.get("/sessions")
async def get_user_sessions(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get user's active sessions"""
    try:
        user_info = SecurityUtils.validate_token(credentials.credentials)
        
        # Get active sessions
        from main import trust_service
        active_sessions = await trust_service.get_user_sessions(user_info['user_id'])
        
        session_details = []
        for session_id in active_sessions:
            session_status = await trust_service.get_session_status(session_id)
            if session_status:
                session_details.append({
                    "session_id": session_id,
                    "trust_index": session_status['trust_index'],
                    "risk_level": session_status['risk_level'],
                    "last_activity": session_status['last_activity'].isoformat(),
                    "mirage_active": session_status['mirage_active']
                })
        
        return {
            "active_sessions": session_details,
            "total_count": len(session_details)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user sessions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get sessions")