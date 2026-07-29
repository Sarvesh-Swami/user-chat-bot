import uuid
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class SessionManager:
    """Manages conversation sessions and context for the chatbot"""
    
    def __init__(self, session_timeout_minutes: int = 30, max_history_turns: int = 10):
        self.sessions: Dict[str, Dict] = {}
        self.session_timeout_minutes = session_timeout_minutes
        self.max_history_turns = max_history_turns
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """Get existing session or create new one"""
        if session_id and session_id in self.sessions:
            # Update last activity
            self.sessions[session_id]["last_activity"] = datetime.now()
            return session_id
        
        # Create new session using the provided session_id if available, otherwise generate UUID
        new_session_id = session_id if session_id else str(uuid.uuid4())
        self.sessions[new_session_id] = {
            "created": datetime.now(),
            "last_activity": datetime.now(),
            "messages": [],
            "last_raw_data": None,  # Store the last query result data for chart generation
            "last_pdf_report": None  # Store the last generated PDF report info
        }
        return new_session_id
    
    def get_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session"""
        if session_id not in self.sessions:
            return []
        
        self._cleanup_expired_sessions()
        return self.sessions[session_id]["messages"]
    
    def format_history_for_llm(self, session_id: str) -> str:
        """Format conversation history for LLM prompt"""
        history = self.get_history(session_id)
        if not history:
            return ""
        
        formatted_lines = []
        for msg in history[-self.max_history_turns:]:  # Only recent turns
            formatted_lines.append(f"User: {msg['user_prompt']}")
            formatted_lines.append(f"Assistant: {msg['assistant_summary']}")
        
        return "\n".join(formatted_lines)
    
    def get_condensation_prompt(self, chat_history_str: str, current_query: str) -> str:
        """Generate prompt for query condensation"""
        return f"""Given the following conversation history, rewrite the current user question as a standalone question that contains all necessary context for a fleet management database query.

Previous conversation:
{chat_history_str}

Current user question: {current_query}

CRITICAL INSTRUCTIONS - Read Carefully:

1. **ONLY bind context when the user uses explicit pronoun references:**
   - Pronouns: "it", "its", "that", "this", "them", "those", "these"
   - Examples that SHOULD use context:
     * "show me its trips" → use vehicle from history
     * "what about that vehicle" → use vehicle from history
     * "tell me about them" → use vehicles from history

2. **DO NOT bind context for general queries without pronouns:**
   - General queries: "the trips", "all trips", "trips from this week", "vehicles", "the longest trip"
   - Examples that SHOULD NOT use context:
     * "show me the trips from this week" → return as-is (general query, no pronoun)
     * "what are all the vehicles" → return as-is (general query, no pronoun)
     * "longest trip this week" → return as-is (general query, no pronoun)

3. **The difference:**
   - "show me ITS trips this week" (pronoun "its") → bind to VHC1119 from history
   - "show me THE trips this week" (article "the", no pronoun) → DO NOT bind to VHC1119

4. **Test before rewriting:**
   - Does the query contain pronouns like "it", "its", "that", "them", "those"? → USE context
   - Does the query only have general articles like "the", "a", "all"? → DO NOT USE context

Return ONLY the rewritten standalone question, no explanation.

Standalone question:"""
    
    def add_turn(self, session_id: str, user_prompt: str, assistant_summary: str):
        """Add a conversation turn to the session"""
        if session_id not in self.sessions:
            self.get_or_create_session(session_id)
        
        turn = {
            "user_prompt": user_prompt,
            "assistant_summary": assistant_summary,
            "timestamp": datetime.now().isoformat()
        }
        
        self.sessions[session_id]["messages"].append(turn)
        self.sessions[session_id]["last_activity"] = datetime.now()
        
        # Limit history size
        if len(self.sessions[session_id]["messages"]) > self.max_history_turns * 2:
            # Keep only recent turns
            self.sessions[session_id]["messages"] = self.sessions[session_id]["messages"][-self.max_history_turns:]
    
    def clear_session(self, session_id: str):
        """Clear a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session_data in self.sessions.items():
            if current_time - session_data["last_activity"] > timedelta(minutes=self.session_timeout_minutes):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
    
    def get_session_count(self) -> int:
        """Get number of active sessions"""
        self._cleanup_expired_sessions()
        return len(self.sessions)
    
    def set_last_raw_data(self, session_id: str, raw_data: List[Dict]):
        """Store the last query result data for potential chart generation"""
        if session_id in self.sessions:
            self.sessions[session_id]["last_raw_data"] = raw_data
            self.sessions[session_id]["last_activity"] = datetime.now()
    
    def get_last_raw_data(self, session_id: str) -> List[Dict]:
        """Get the last stored raw data for chart generation"""
        if session_id in self.sessions:
            return self.sessions[session_id].get("last_raw_data", [])
        return []

    def set_last_pdf_report(self, session_id: str, pdf_info: dict):
        """Store the last generated PDF report info"""
        if session_id in self.sessions:
            self.sessions[session_id]["last_pdf_report"] = pdf_info
            self.sessions[session_id]["last_activity"] = datetime.now()

    def get_last_pdf_report(self, session_id: str) -> Optional[dict]:
        """Get the last generated PDF report info"""
        if session_id in self.sessions:
            return self.sessions[session_id].get("last_pdf_report")
        return None