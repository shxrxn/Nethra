"""
Monitoring Endpoints - Performance and system monitoring APIs
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import PlainTextResponse
from fastapi.responses import HTMLResponse
from typing import Dict, List, Optional
from datetime import datetime
import logging

from utils.security_utils import SecurityUtils

router = APIRouter()
security = HTTPBearer()
logger = logging.getLogger(__name__)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from token"""
    try:
        user_info = SecurityUtils.validate_token(credentials.credentials)
        return user_info
    except Exception as e:
        logger.error(f"Token validation failed: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid authentication token")

@router.get("/metrics")
async def get_metrics():
    """Get system metrics (Prometheus format)"""
    try:
        from main import monitoring_service
        
        if not monitoring_service:
            raise HTTPException(status_code=503, detail="Monitoring service not available")
        
        prometheus_metrics = monitoring_service.export_prometheus_metrics()
        return PlainTextResponse(prometheus_metrics, media_type="text/plain")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

@router.get("/health/detailed")
async def get_detailed_health():
    """Get detailed health information"""
    try:
        from main import monitoring_service, cache_service, encryption_service
        
        health_data = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "monitoring": monitoring_service is not None,
                "cache": cache_service is not None,
                "encryption": encryption_service is not None
            },
            "system_info": {},
            "performance": {},
            "cache_stats": {}
        }
        
        if monitoring_service:
            health_data["system_info"] = monitoring_service.get_system_info()
            health_data["performance"] = monitoring_service.get_performance_metrics()
        
        if cache_service:
            health_data["cache_stats"] = cache_service.get_stats()
        
        return health_data
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")

@router.get("/performance")
async def get_performance_metrics(
    user_info: dict = Depends(get_current_user)
):
    """Get performance metrics"""
    try:
        from main import monitoring_service
        
        if not monitoring_service:
            raise HTTPException(status_code=503, detail="Monitoring service not available")
        
        performance_data = {
            "metrics_summary": monitoring_service.get_metrics_summary(),
            "counters": monitoring_service.get_counters(),
            "performance_metrics": monitoring_service.get_performance_metrics(),
            "system_info": monitoring_service.get_system_info()
        }
        
        return performance_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance metrics")

@router.get("/alerts")
async def get_alerts(
    limit: int = 50,
    user_info: dict = Depends(get_current_user)
):
    """Get system alerts"""
    try:
        from main import monitoring_service
        
        if not monitoring_service:
            raise HTTPException(status_code=503, detail="Monitoring service not available")
        
        alerts = monitoring_service.get_alerts(limit)
        
        return {
            "alerts": alerts,
            "total_count": len(alerts),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get alerts: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")

@router.get("/cache/stats")
async def get_cache_stats(
    user_info: dict = Depends(get_current_user)
):
    """Get cache statistics"""
    try:
        from main import cache_service
        
        if not cache_service:
            raise HTTPException(status_code=503, detail="Cache service not available")
        
        cache_stats = cache_service.get_stats()
        
        return {
            "cache_stats": cache_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cache stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cache statistics")

@router.post("/cache/clear")
async def clear_cache(
    user_info: dict = Depends(get_current_user)
):
    """Clear cache (admin only)"""
    try:
        from main import cache_service
        
        if not cache_service:
            raise HTTPException(status_code=503, detail="Cache service not available")
        
        await cache_service.clear()
        
        return {
            "message": "Cache cleared successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear cache: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")

@router.get("/dashboard")
async def get_monitoring_dashboard():
    """Get monitoring dashboard data"""
    try:
        from main import monitoring_service, cache_service
        
        dashboard_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_status": "healthy",
            "key_metrics": {},
            "recent_alerts": [],
            "performance_summary": {},
            "cache_summary": {}
        }
        
        if monitoring_service:
            # Key metrics
            metrics_summary = monitoring_service.get_metrics_summary()
            dashboard_data["key_metrics"] = {
                "cpu_usage": metrics_summary.get("system_cpu_usage", {}).get("latest", 0),
                "memory_usage": metrics_summary.get("system_memory_usage", {}).get("latest", 0),
                "avg_response_time": metrics_summary.get("behavioral_analysis_time", {}).get("avg", 0),
                "total_requests": monitoring_service.get_counters().get("total_requests", 0)
            }
            
            # Recent alerts
            dashboard_data["recent_alerts"] = monitoring_service.get_alerts(10)
            
            # Performance summary
            dashboard_data["performance_summary"] = monitoring_service.get_performance_metrics()
        
        if cache_service:
            dashboard_data["cache_summary"] = cache_service.get_stats()
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve dashboard data")

@router.get("/grafana/datasource")
async def grafana_datasource():
    """Grafana datasource endpoint"""
    try:
        from main import monitoring_service
        
        if not monitoring_service:
            raise HTTPException(status_code=503, detail="Monitoring service not available")
        
        # Return metrics in Grafana-compatible format
        metrics_data = []
        
        metrics_summary = monitoring_service.get_metrics_summary()
        for metric_name, data in metrics_summary.items():
            metrics_data.append({
                "target": metric_name,
                "datapoints": [[data.get("latest", 0), int(datetime.utcnow().timestamp() * 1000)]]
            })
        
        return metrics_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Grafana datasource failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Grafana datasource failed")

@router.get("/dashboard/html", response_class=HTMLResponse)
async def get_html_dashboard():
    """Get HTML dashboard with live graphs"""
    try:
        with open("templates/dashboard.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html><body>
        <h1>Dashboard Template Not Found</h1>
        <p>Please ensure templates/dashboard.html exists</p>
        </body></html>
        """)

@router.get("/rate-limit/stats")
async def get_rate_limit_stats(
    user_info: dict = Depends(get_current_user)
):
    """Get rate limiting statistics"""
    try:
        from services.rate_limiter import rate_limiter
        
        stats = rate_limiter.get_stats()
        
        return {
            "rate_limit_stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get rate limit stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve rate limit statistics")