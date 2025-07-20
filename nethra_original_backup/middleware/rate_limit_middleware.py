"""
Rate Limiting Middleware - Protects API endpoints from abuse
"""

import logging
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce rate limiting on API endpoints"""
    
    def __init__(self, app):
        super().__init__(app)
        self.rate_limiter = None
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting"""
        try:
            # Import rate limiter (avoid circular imports)
            if not self.rate_limiter:
                from services.rate_limiter import rate_limiter
                self.rate_limiter = rate_limiter
            
            # Extract user identifier
            user_id = await self._get_user_id(request)
            
            # Determine endpoint type for rate limiting
            endpoint_type = self._get_endpoint_type(request.url.path)
            
            # Check rate limit
            if not await self.rate_limiter.is_allowed(user_id, endpoint_type):
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "message": "Too many requests. Please try again later.",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            
            # Process request
            response = await call_next(request)
            
            # Record successful request
            await self.rate_limiter.record_request(user_id, endpoint_type)
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limiting middleware error: {str(e)}")
            # Continue processing on middleware error
            return await call_next(request)
    
    async def _get_user_id(self, request: Request) -> str:
        """Extract user ID from request"""
        try:
            # Try to get from Authorization header
            auth_header = request.headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                
                # Validate token and extract user ID
                from utils.security_utils import SecurityUtils
                user_info = SecurityUtils.validate_token(token)
                return user_info.get("user_id", "anonymous")
            
            # Fallback to IP address
            client_ip = request.client.host if request.client else "unknown"
            return f"ip_{client_ip}"
            
        except Exception as e:
            logger.error(f"Failed to extract user ID: {str(e)}")
            # Fallback to IP address
            client_ip = request.client.host if request.client else "unknown"
            return f"ip_{client_ip}"
    
    def _get_endpoint_type(self, path: str) -> str:
        """Determine endpoint type for rate limiting"""
        if "/auth/login" in path:
            return "login"
        elif "/auth/" in path:
            return "auth"
        elif "/behavioral/analyze" in path:
            return "behavioral"      # HIGH LIMITS!
        elif "/trust/" in path:
            return "trust"
        elif "/session/" in path:
            return "session"
        elif "/monitoring/" in path:
            return "monitoring"
        elif "/demo/" in path:
            return "demo"
        else:
            return "general"