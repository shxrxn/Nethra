"""
SQLite Database Configuration and Connection Management
"""

import sqlite3
import asyncio
import aiosqlite
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class DatabaseManager:
    """SQLite Database Manager for NETHRA"""
    
    def __init__(self, db_path: str = "nethra.db"):
        self.db_path = Path(db_path)
        self.connection_pool = {}
        
    async def initialize_database(self):
        """Initialize database with all required tables"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Enable foreign keys
                await db.execute("PRAGMA foreign_keys = ON")
                
                # Create tables
                await self._create_users_table(db)
                await self._create_sessions_table(db)
                await self._create_behavioral_data_table(db)
                await self._create_trust_scores_table(db)
                await self._create_anomalies_table(db)
                await self._create_mirage_sessions_table(db)
                await self._create_tamper_logs_table(db)
                await self._create_security_incidents_table(db)
                
                await db.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise
    
    async def _create_users_table(self, db):
        """Create users table"""
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                device_info TEXT,
                behavioral_baseline TEXT,
                trust_profile TEXT,
                is_active BOOLEAN DEFAULT 1
            )
        """)
    
    async def _create_sessions_table(self, db):
        """Create sessions table"""
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                trust_index REAL DEFAULT 100.0,
                risk_level TEXT DEFAULT 'LOW',
                is_active BOOLEAN DEFAULT 1,
                device_fingerprint TEXT,
                mirage_active BOOLEAN DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        """)
    
    async def _create_behavioral_data_table(self, db):
        """Create behavioral data table"""
        await db.execute("""
            CREATE TABLE IF NOT EXISTS behavioral_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                touch_patterns TEXT,
                swipe_patterns TEXT,
                device_motion TEXT,
                app_usage TEXT,
                network_info TEXT,
                location_context TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id),
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        """)
    
    async def _create_trust_scores_table(self, db):
        """Create trust scores table"""
        await db.execute("""
            CREATE TABLE IF NOT EXISTS trust_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                trust_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                ai_score REAL,
                deviation_score REAL,
                anomaly_count INTEGER DEFAULT 0,
                features TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id),
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        """)
    
    async def _create_anomalies_table(self, db):
        """Create anomalies table"""
        await db.execute("""
            CREATE TABLE IF NOT EXISTS anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                anomaly_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                confidence REAL NOT NULL,
                description TEXT,
                features_involved TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id),
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        """)
    
    async def _create_mirage_sessions_table(self, db):
        """Create mirage sessions table"""
        await db.execute("""
            CREATE TABLE IF NOT EXISTS mirage_sessions (
                mirage_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                mirage_type TEXT NOT NULL,
                trust_index REAL NOT NULL,
                content TEXT,
                challenges TEXT,
                interactions TEXT DEFAULT '[]',
                status TEXT DEFAULT 'ACTIVE',
                FOREIGN KEY (user_id) REFERENCES users (user_id),
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        """)
    
    async def _create_tamper_logs_table(self, db):
        """Create tamper logs table"""
        await db.execute("""
            CREATE TABLE IF NOT EXISTS tamper_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tamper_score INTEGER NOT NULL,
                severity TEXT NOT NULL,
                indicators TEXT,
                device_integrity BOOLEAN,
                app_integrity BOOLEAN,
                network_integrity BOOLEAN,
                FOREIGN KEY (user_id) REFERENCES users (user_id),
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        """)
    
    async def _create_security_incidents_table(self, db):
        """Create security incidents table"""
        await db.execute("""
            CREATE TABLE IF NOT EXISTS security_incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                incident_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                details TEXT,
                resolved BOOLEAN DEFAULT 0,
                resolution_notes TEXT
            )
        """)
    
    async def get_connection(self):
        """Get database connection"""
        return await aiosqlite.connect(self.db_path)
    
    async def execute_query(self, query: str, params: tuple = ()):
        """Execute a query and return results"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(query, params) as cursor:
                    return await cursor.fetchall()
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise
    
    async def execute_insert(self, query: str, params: tuple = ()):
        """Execute insert query and return last row id"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(query, params)
                await db.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Insert execution failed: {str(e)}")
            raise
    
    async def execute_update(self, query: str, params: tuple = ()):
        """Execute update query and return affected rows"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(query, params)
                await db.commit()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Update execution failed: {str(e)}")
            raise

# Global database manager instance
db_manager = DatabaseManager()