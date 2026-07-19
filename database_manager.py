import os
import json
import psycopg2
import psycopg2.extras
import numpy as np
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabaseManager:
    """Handles all PostgreSQL database connections and raw operations"""
    
    def __init__(self):
        """Initialize database connection parameters"""
        self.database_url = os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable is required")
        
        # Parse database URL for connection parameters
        self.db_params = self._parse_database_url(self.database_url)
        
        # Initialize connection
        self.conn = None
        self._init_database_connection()
    
    def _parse_database_url(self, database_url):
        """Parse PostgreSQL database URL into connection parameters"""
        try:
            parsed = urlparse(database_url)
            return {
                'host': parsed.hostname,
                'port': parsed.port or 5432,
                'database': parsed.path[1:],  # Remove leading slash
                'user': parsed.username,
                'password': parsed.password,
            }
        except Exception as e:
            raise ValueError(f"Invalid DATABASE_URL format: {e}")
    
    def _init_database_connection(self):
        """Initialize PostgreSQL database connection"""
        try:
            self.conn = psycopg2.connect(**self.db_params)
            self.conn.autocommit = False  # Enable transaction control
            print(f"[DATABASE] Successfully connected to PostgreSQL: {self.db_params['host']}:{self.db_params['port']}/{self.db_params['database']}")
        except Exception as e:
            print(f"[DATABASE ERROR] Failed to connect to PostgreSQL: {e}")
            raise ConnectionError(f"Cannot connect to database: {e}")
    
    def _ensure_connection(self):
        """Ensure database connection is alive, reconnect if necessary"""
        try:
            if self.conn.closed:
                print("[DATABASE] Connection closed, reconnecting...")
                self._init_database_connection()
            else:
                # Test connection with a simple query
                cursor = self.conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
        except Exception as e:
            print(f"[DATABASE] Connection test failed, reconnecting: {e}")
            self._init_database_connection()
    
    def _execute_query_with_retry(self, query, params=None, fetch_results=True, max_retries=2):
        """Execute a query with automatic retry and connection management"""
        for attempt in range(max_retries + 1):
            try:
                self._ensure_connection()
                
                cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                if fetch_results:
                    results = cursor.fetchall()
                    cursor.close()
                    return results
                else:
                    cursor.close()
                    self.conn.commit()
                    return None
                    
            except Exception as e:
                print(f"[DATABASE] Query attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    raise e
                else:
                    # Try to reconnect for next attempt
                    try:
                        self._init_database_connection()
                    except Exception:
                        continue
        
        raise Exception("Failed to execute query after all retry attempts")
    
    def _json_serializer(self, obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif isinstance(obj, bytes):
            return obj.decode('utf-8')
        else:
            return str(obj)
    
    def execute_query(self, query, params=None):
        """Public method to execute queries with results"""
        return self._execute_query_with_retry(query, params, fetch_results=True)
    
    def execute_command(self, query, params=None):
        """Public method to execute commands without results (INSERT, UPDATE, DELETE)"""
        return self._execute_query_with_retry(query, params, fetch_results=False)
    
    def get_cursor(self):
        """Get a database cursor for complex operations"""
        self._ensure_connection()
        return self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    def close_connection(self):
        """Close database connection"""
        if self.conn and not self.conn.closed:
            self.conn.close()
            print("[DATABASE] Connection closed")