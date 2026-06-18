from fastapi import FastAPI, HTTPException
from llm_service import ChatbotService, QueryRequest

# Initialize the FastAPI app
app = FastAPI()

# Initialize the chatbot service
bot_service = ChatbotService()

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
        
    answer = bot_service.answer_user_query(request.prompt)
    return {"response": answer}