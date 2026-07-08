import json
from fastapi import FastAPI, HTTPException
from llm_service import ChatbotService, QueryRequest
from query_history import QueryHistoryManager, QueryLog
import time
from typing import List

# Initialize the FastAPI app
app = FastAPI()

# Initialize the chatbot service and query history manager
bot_service = ChatbotService()
history_manager = QueryHistoryManager()

# Define a root route (GET request)
@app.get("/")
def read_root():
    return {"message": "Welcome to your FastAPI server!"}

# Define a route with a path parameter and a query parameter
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "query_param": q}  

# Add the chat endpoint
@app.post("/api/v1/chat")
def chat_endpoint(request: QueryRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    start_time = time.time()
    
    try:
        answer = bot_service.answer_user_query(request.prompt)
        execution_time_ms = (time.time() - start_time) * 1000
        
        # <-- 2. Update this block to parse the string safely -->
        try:
            # Convert the raw LLM JSON string into a native Python dict
            parsed_answer = json.loads(answer)
            response_data = {"response": parsed_answer}
        except json.JSONDecodeError:
            # Fallback if the LLM occasionally returns regular text instead of JSON
            response_data = {"response": answer}
        
        # Add execution time to response
        response_data["execution_time_ms"] = round(execution_time_ms, 2)
        
        # Log the successful query
        history_manager.log_query(
            user_query=request.prompt,
            response=answer,
            execution_time_ms=execution_time_ms,
            status="success"
        )
        
        return response_data
        
    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000
        error_message = f"Error processing query: {str(e)}"
        
        # Log the failed query
        history_manager.log_query(
            user_query=request.prompt,
            response=error_message,
            execution_time_ms=execution_time_ms,
            status="error"
        )
        
        raise HTTPException(status_code=500, detail=error_message)

# Query history endpoints
@app.get("/api/v1/history")
def get_query_history(limit: int = 10):
    """Get recent query history"""
    try:
        recent_queries = history_manager.get_recent_queries(limit)
        return {
            "status": "success",
            "count": len(recent_queries),
            "queries": [query.dict() for query in recent_queries]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")

@app.get("/api/v1/history/stats")
def get_query_stats():
    """Get query statistics and system health info"""
    try:
        stats = history_manager.get_query_stats()
        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")

@app.get("/api/v1/history/search")
def search_query_history(q: str, limit: int = 20):
    """Search through query history"""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty")
    
    try:
        matching_queries = history_manager.search_queries(q, limit)
        return {
            "status": "success",
            "search_term": q,
            "count": len(matching_queries),
            "queries": [query.dict() for query in matching_queries]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching history: {str(e)}")