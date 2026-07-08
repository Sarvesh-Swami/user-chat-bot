import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel

class QueryLog(BaseModel):
    """Model for individual query log entries"""
    timestamp: str
    user_query: str
    sql_generated: Optional[str]
    response: str
    execution_time_ms: Optional[float]
    status: str  # "success", "error", "warning"

class QueryHistoryManager:
    """Manages query history logging and retrieval"""
    
    def __init__(self, log_file_path: str = "logs/query_history.json"):
        self.log_file_path = log_file_path
        self._ensure_log_directory()
    
    def _ensure_log_directory(self):
        """Create logs directory if it doesn't exist"""
        log_dir = os.path.dirname(self.log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    def log_query(self, user_query: str, response: str, sql_generated: str = None, 
                  execution_time_ms: float = None, status: str = "success"):
        """Log a query and its response"""
        
        log_entry = QueryLog(
            timestamp=datetime.now().isoformat(),
            user_query=user_query,
            sql_generated=sql_generated,
            response=response,
            execution_time_ms=execution_time_ms,
            status=status
        )
        
        # Read existing logs
        logs = self._read_logs()
        
        # Add new log entry
        logs.append(log_entry.dict())
        
        # Keep only last 100 entries to prevent file from getting too large
        if len(logs) > 100:
            logs = logs[-100:]
        
        # Write back to file
        self._write_logs(logs)
    
    def _read_logs(self) -> List[Dict]:
        """Read existing logs from file"""
        if not os.path.exists(self.log_file_path):
            return []
        
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _write_logs(self, logs: List[Dict]):
        """Write logs to file"""
        try:
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing to log file: {e}")
    
    def get_recent_queries(self, limit: int = 10) -> List[QueryLog]:
        """Get recent query history"""
        logs = self._read_logs()
        
        # Return most recent entries first
        recent_logs = logs[-limit:] if len(logs) >= limit else logs
        recent_logs.reverse()  # Most recent first
        
        return [QueryLog(**log) for log in recent_logs]
    
    def get_query_stats(self) -> Dict:
        """Get basic statistics about queries"""
        logs = self._read_logs()
        
        if not logs:
            return {
                "total_queries": 0,
                "success_rate": 0,
                "avg_execution_time_ms": 0,
                "most_recent_query": None
            }
        
        total_queries = len(logs)
        successful_queries = len([log for log in logs if log.get("status") == "success"])
        success_rate = (successful_queries / total_queries) * 100 if total_queries > 0 else 0
        
        # Calculate average execution time for queries that have timing data
        execution_times = [log.get("execution_time_ms") for log in logs if log.get("execution_time_ms")]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        most_recent = logs[-1]["timestamp"] if logs else None
        
        return {
            "total_queries": total_queries,
            "success_rate": round(success_rate, 2),
            "avg_execution_time_ms": round(avg_execution_time, 2),
            "most_recent_query": most_recent
        }
    
    def search_queries(self, search_term: str, limit: int = 20) -> List[QueryLog]:
        """Search through query history"""
        logs = self._read_logs()
        
        matching_logs = []
        for log in logs:
            if (search_term.lower() in log.get("user_query", "").lower() or 
                search_term.lower() in log.get("response", "").lower()):
                matching_logs.append(QueryLog(**log))
        
        # Return most recent matches first, limited by the limit parameter
        return matching_logs[-limit:] if len(matching_logs) >= limit else matching_logs