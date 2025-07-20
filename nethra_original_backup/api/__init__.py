"""
API package initialization
"""

from .endpoints import router as api_router
from .auth_endpoints import router as auth_router
from .trust_endpoints import router as trust_router
from .monitoring_endpoints import router as monitoring_router

__all__ = [
    "api_router",
    "auth_router",
    "trust_router",
    "monitoring_router"
]