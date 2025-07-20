"""
Database Models and Operations
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from database.database import db_manager

logger = logging.getLogger(__name__)

class UserModel:
    """User database operations"""
    
    @staticmethod
    async def create_user(user_id: str, device_info: Dict) -> bool:
        """Create a new user"""
        try:
            query = """
                INSERT OR REPLACE INTO users 
                (user_id, device_info, created_at, last_login)
                VALUES (?, ?, ?, ?)
            """
            params = (
                user_id,
                json.dumps(device_info),
                datetime.utcnow().isoformat(),
                datetime.utcnow().isoformat()
            )
            
            await db_manager.execute_insert(query, params)
            return True
            
        except Exception as e:
            logger.error(f"Failed to create user: {str(e)}")
            return False
    
    @staticmethod
    async def get_user(user_id: str) -> Optional[Dict]:
        """Get user by ID"""
        try:
            query = "SELECT * FROM users WHERE user_id = ?"
            result = await db_manager.execute_query(query, (user_id,))
            
            if result:
                row = result[0]
                return {
                    "user_id": row[0],
                    "created_at": row[1],
                    "last_login": row[2],
                    "device_info": json.loads(row[3]) if row[3] else {},
                    "behavioral_baseline": json.loads(row[4]) if row[4] else {},
                    "trust_profile": json.loads(row[5]) if row[5] else {},
                    "is_active": bool(row[6])
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user: {str(e)}")
            return None
    
    @staticmethod
    async def update_user_baseline(user_id: str, baseline: Dict) -> bool:
        """Update user behavioral baseline"""
        try:
            query = """
                UPDATE users 
                SET behavioral_baseline = ?, last_login = ?
                WHERE user_id = ?
            """
            params = (
                json.dumps(baseline),
                datetime.utcnow().isoformat(),
                user_id
            )
            
            await db_manager.execute_update(query, params)
            return True
            
        except Exception as e:
            logger.error(f"Failed to update user baseline: {str(e)}")
            return False

class SessionModel:
    """Session database operations"""
    
    @staticmethod
    async def create_session(session_id: str, user_id: str, device_fingerprint: str = "") -> bool:
        """Create a new session"""
        try:
            query = """
                INSERT INTO sessions 
                (session_id, user_id, device_fingerprint, created_at, last_activity)
                VALUES (?, ?, ?, ?, ?)
            """
            params = (
                session_id,
                user_id,
                device_fingerprint,
                datetime.utcnow().isoformat(),
                datetime.utcnow().isoformat()
            )
            
            await db_manager.execute_insert(query, params)
            return True
            
        except Exception as e:
            logger.error(f"Failed to create session: {str(e)}")
            return False
    
    @staticmethod
    async def get_session(session_id: str) -> Optional[Dict]:
        """Get session by ID"""
        try:
            query = "SELECT * FROM sessions WHERE session_id = ?"
            result = await db_manager.execute_query(query, (session_id,))
            
            if result:
                row = result[0]
                return {
                    "session_id": row[0],
                    "user_id": row[1],
                    "created_at": datetime.fromisoformat(row[2]),
                    "last_activity": datetime.fromisoformat(row[3]),
                    "trust_index": float(row[4]),
                    "risk_level": row[5],
                    "is_active": bool(row[6]),
                    "device_fingerprint": row[7],
                    "mirage_active": bool(row[8])
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session: {str(e)}")
            return None
    
    @staticmethod
    async def update_session_trust(session_id: str, trust_score: float, risk_level: str) -> bool:
        """Update session trust score"""
        try:
            query = """
                UPDATE sessions 
                SET trust_index = ?, risk_level = ?, last_activity = ?
                WHERE session_id = ?
            """
            params = (
                trust_score,
                risk_level,
                datetime.utcnow().isoformat(),
                session_id
            )
            
            await db_manager.execute_update(query, params)
            return True
            
        except Exception as e:
            logger.error(f"Failed to update session trust: {str(e)}")
            return False
    
    @staticmethod
    async def get_user_sessions(user_id: str) -> List[str]:
        """Get all active sessions for user"""
        try:
            query = """
                SELECT session_id FROM sessions 
                WHERE user_id = ? AND is_active = 1
            """
            result = await db_manager.execute_query(query, (user_id,))
            return [row[0] for row in result]
            
        except Exception as e:
            logger.error(f"Failed to get user sessions: {str(e)}")
            return []
    
    @staticmethod
    async def terminate_session(session_id: str) -> bool:
        """Terminate a session"""
        try:
            query = "UPDATE sessions SET is_active = 0 WHERE session_id = ?"
            await db_manager.execute_update(query, (session_id,))
            return True
            
        except Exception as e:
            logger.error(f"Failed to terminate session: {str(e)}")
            return False

class TrustScoreModel:
    """Trust score database operations"""
    
    @staticmethod
    async def store_trust_score(user_id: str, session_id: str, trust_data: Dict) -> bool:
        """Store trust score analysis"""
        try:
            query = """
                INSERT INTO trust_scores 
                (user_id, session_id, trust_score, risk_level, ai_score, 
                 deviation_score, anomaly_count, features)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                user_id,
                session_id,
                trust_data.get('trust_score', 0),
                trust_data.get('risk_level', 'MEDIUM'),
                trust_data.get('ai_score', 0),
                trust_data.get('deviation_score', 0),
                len(trust_data.get('anomalies', [])),
                json.dumps(trust_data.get('features', []))
            )
            
            await db_manager.execute_insert(query, params)
            return True
            
        except Exception as e:
            logger.error(f"Failed to store trust score: {str(e)}")
            return False
    
    @staticmethod
    async def get_trust_history(session_id: str, limit: int = 100) -> List[Dict]:
        """Get trust score history for session"""
        try:
            query = """
                SELECT trust_score, risk_level, timestamp 
                FROM trust_scores 
                WHERE session_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            result = await db_manager.execute_query(query, (session_id, limit))
            
            return [
                {
                    "trust_score": row[0],
                    "risk_level": row[1],
                    "timestamp": row[2]
                }
                for row in result
            ]
            
        except Exception as e:
            logger.error(f"Failed to get trust history: {str(e)}")
            return []

class MirageModel:
    """Mirage session database operations"""
    
    @staticmethod
    async def create_mirage_session(mirage_data: Dict) -> bool:
        """Create mirage session"""
        try:
            query = """
                INSERT INTO mirage_sessions 
                (mirage_id, user_id, session_id, expires_at, mirage_type, 
                 trust_index, content, challenges)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                mirage_data['mirage_id'],
                mirage_data['user_id'],
                mirage_data['session_id'],
                mirage_data['expires_at'],
                mirage_data['mirage_type'],
                mirage_data['trust_index'],
                json.dumps(mirage_data['content']),
                json.dumps(mirage_data['challenges'])
            )
            
            await db_manager.execute_insert(query, params)
            return True
            
        except Exception as e:
            logger.error(f"Failed to create mirage session: {str(e)}")
            return False
    
    @staticmethod
    async def get_mirage_session(mirage_id: str) -> Optional[Dict]:
        """Get mirage session"""
        try:
            query = "SELECT * FROM mirage_sessions WHERE mirage_id = ?"
            result = await db_manager.execute_query(query, (mirage_id,))
            
            if result:
                row = result[0]
                return {
                    "mirage_id": row[0],
                    "user_id": row[1],
                    "session_id": row[2],
                    "created_at": row[3],
                    "expires_at": row[4],
                    "mirage_type": row[5],
                    "trust_index": row[6],
                    "content": json.loads(row[7]) if row[7] else [],
                    "challenges": json.loads(row[8]) if row[8] else [],
                    "interactions": json.loads(row[9]) if row[9] else [],
                    "status": row[10]
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get mirage session: {str(e)}")
            return None
    
    @staticmethod
    async def update_mirage_interactions(mirage_id: str, interactions: List[Dict]) -> bool:
        """Update mirage interactions"""
        try:
            query = """
                UPDATE mirage_sessions 
                SET interactions = ? 
                WHERE mirage_id = ?
            """
            params = (json.dumps(interactions), mirage_id)
            
            await db_manager.execute_update(query, params)
            return True
            
        except Exception as e:
            logger.error(f"Failed to update mirage interactions: {str(e)}")
            return False

class SecurityModel:
    """Security incidents and tamper logs"""
    
    @staticmethod
    async def log_security_incident(user_id: str, incident_type: str, severity: str, details: Dict) -> bool:
        """Log security incident"""
        try:
            query = """
                INSERT INTO security_incidents 
                (user_id, incident_type, severity, details)
                VALUES (?, ?, ?, ?)
            """
            params = (
                user_id,
                incident_type,
                severity,
                json.dumps(details)
            )
            
            await db_manager.execute_insert(query, params)
            return True
            
        except Exception as e:
            logger.error(f"Failed to log security incident: {str(e)}")
            return False
    
    @staticmethod
    async def store_tamper_log(user_id: str, session_id: str, tamper_data: Dict) -> bool:
        """Store tamper detection log"""
        try:
            query = """
                INSERT INTO tamper_logs 
                (user_id, session_id, tamper_score, severity, indicators,
                 device_integrity, app_integrity, network_integrity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                user_id,
                session_id,
                tamper_data.get('tamper_score', 0),
                tamper_data.get('severity', 'LOW'),
                json.dumps(tamper_data.get('indicators', [])),
                tamper_data.get('device_integrity', True),
                tamper_data.get('app_integrity', True),
                tamper_data.get('network_integrity', True)
            )
            
            await db_manager.execute_insert(query, params)
            return True
            
        except Exception as e:
            logger.error(f"Failed to store tamper log: {str(e)}")
            return False