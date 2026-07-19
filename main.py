import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from chatbot_service import ChatbotService, QueryRequest
from query_history import QueryHistoryManager, QueryLog
import time
from typing import List

# Initialize the FastAPI app
app = FastAPI(title="Fleet Management Chatbot API", version="1.0.0")

# Initialize the chatbot service and query history manager
bot_service = ChatbotService()
history_manager = QueryHistoryManager()

@app.get("/", response_class=HTMLResponse)
def chat_interface():
    """Serve the main chat interface"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🚛 Fleet Management Assistant</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            
            .chat-container {
                width: 90%;
                max-width: 800px;
                height: 90%;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 24px;
                margin-bottom: 5px;
            }
            
            .header p {
                opacity: 0.9;
                font-size: 14px;
            }
            
            .chat-messages {
                flex: 1;
                padding: 20px;
                overflow-y: auto;
                background: #f8f9fa;
            }
            
            .message {
                margin-bottom: 15px;
                display: flex;
                align-items: flex-start;
            }
            
            .message.user {
                justify-content: flex-end;
            }
            
            .message-content {
                max-width: 70%;
                padding: 12px 16px;
                border-radius: 15px;
                word-wrap: break-word;
            }
            
            .message.user .message-content {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-bottom-right-radius: 5px;
            }
            
            .message.bot .message-content {
                background: white;
                border: 1px solid #e0e0e0;
                border-bottom-left-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            
            .input-area {
                padding: 20px;
                background: white;
                border-top: 1px solid #e0e0e0;
            }
            
            .input-form {
                display: flex;
                gap: 10px;
                align-items: center;
            }
            
            .message-input {
                flex: 1;
                padding: 12px 16px;
                border: 2px solid #e0e0e0;
                border-radius: 25px;
                font-size: 14px;
                outline: none;
                transition: border-color 0.3s;
            }
            
            .message-input:focus {
                border-color: #667eea;
            }
            
            .send-button {
                padding: 12px 24px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-weight: 500;
                transition: transform 0.2s;
            }
            
            .send-button:hover {
                transform: translateY(-2px);
            }
            
            .send-button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            
            .loading {
                display: none;
                text-align: center;
                padding: 10px;
                color: #666;
                font-style: italic;
            }
            
            .temperature-control {
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 10px;
                font-size: 12px;
                color: #666;
            }
            
            .temperature-slider {
                width: 100px;
            }
            
            @media (max-width: 768px) {
                .chat-container {
                    width: 95%;
                    height: 95%;
                }
                
                .header {
                    padding: 15px;
                }
                
                .header h1 {
                    font-size: 20px;
                }
                
                .chat-messages {
                    padding: 15px;
                }
                
                .input-area {
                    padding: 15px;
                }
                
                .message-content {
                    max-width: 85%;
                }
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="header">
                <h1>🚛 Fleet Management Assistant</h1>
                <p>Ask questions about your fleet in natural language</p>
            </div>
            
            <div class="chat-messages" id="chatMessages">
                <div class="message bot">
                    <div class="message-content">
                        👋 Hello! I'm your Fleet Management Assistant. You can ask me about:
                        <br><br>
                        • <strong>Vehicle locations:</strong> "Where is vehicle ABC123?"
                        <br>
                        • <strong>Trip details:</strong> "Show trips for vehicle XYZ789"
                        <br>
                        • <strong>Fleet status:</strong> "How many vehicles are running?"
                        <br><br>
                        Just type your question naturally!
                    </div>
                </div>
            </div>
            
            <div class="input-area">
                <div class="temperature-control">
                    <label>Creativity:</label>
                    <input type="range" id="temperature" class="temperature-slider" min="0" max="1" step="0.1" value="0.3">
                    <span id="tempValue">0.3</span>
                </div>
                
                <form class="input-form" id="chatForm">
                    <input 
                        type="text" 
                        class="message-input" 
                        id="messageInput" 
                        placeholder="Ask about vehicles, trips, locations, or fleet status..."
                        required
                    >
                    <button type="submit" class="send-button" id="sendButton">
                        Send
                    </button>
                </form>
                
                <div class="loading" id="loading">
                    🤔 Thinking...
                </div>
            </div>
        </div>

        <script>
            const chatMessages = document.getElementById('chatMessages');
            const messageInput = document.getElementById('messageInput');
            const sendButton = document.getElementById('sendButton');
            const loading = document.getElementById('loading');
            const temperatureSlider = document.getElementById('temperature');
            const tempValue = document.getElementById('tempValue');

            // Generate a session ID for this chat session
            let sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);

            // Update temperature display
            temperatureSlider.addEventListener('input', function() {
                tempValue.textContent = this.value;
            });

            // Handle form submission
            document.getElementById('chatForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const message = messageInput.value.trim();
                if (!message) return;
                
                // Add user message to chat
                addMessage(message, 'user');
                
                // Clear input and show loading
                messageInput.value = '';
                setLoading(true);
                
                try {
                    // Call your API with session support for conversational memory
                    const response = await fetch('/api/v1/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            prompt: message,
                            temperature: parseFloat(temperatureSlider.value),
                            session_id: sessionId  // Include session ID for conversational memory
                        })
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        
                        let botResponse = 'No response received';
                        let isRichMedia = false;
                        let htmlSnippet = '';
                        
                        // Check if the response payload contains a visual media graph configuration
                        if (data.response && data.response.type === 'rich_media') {
                            botResponse = data.response.display_value;
                            htmlSnippet = data.response.html_content;
                            isRichMedia = true;
                        } else if (data.response && typeof data.response === 'object' && data.response.display_value) {
                            botResponse = data.response.display_value;
                        } else if (data.response && typeof data.response === 'string') {
                            botResponse = data.response;
                        }
                        
                        // Add processing metadata timing parameters
                        if (data.execution_time_ms) {
                            botResponse += `\n\n⏱️ ${data.execution_time_ms.toFixed(0)}ms`;
                        }
                        
                        // Send indicators to the rendering pipeline
                        addMessage(botResponse, 'bot', isRichMedia, htmlSnippet);
                    } else {
                        addMessage(`❌ Error: ${response.status} ${response.statusText}`, 'bot');
                    }
                } catch (error) {
                    addMessage(`❌ Network Error: ${error.message}`, 'bot');
                } finally {
                    setLoading(false);
                    messageInput.focus();
                }
            });

            function addMessage(content, type, isRichMedia = false, htmlSnippet = '') {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${type}`;
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                
                if (isRichMedia && type === 'bot') {
                    // Step A: Print the summary label text layout first
                    const textNode = document.createElement('div');
                    textNode.textContent = content;
                    contentDiv.appendChild(textNode);
                    
                    // Step B: Inject structural visualization card (HTML/CSS layout wrapper)
                    const chartWrapper = document.createElement('div');
                    chartWrapper.innerHTML = htmlSnippet;
                    contentDiv.appendChild(chartWrapper);
                    
                    // Step C: Force execute injected Chart JS script objects sequentially
                    setTimeout(() => {
                        const inlineScripts = chartWrapper.getElementsByTagName('script');
                        for (let oldScript of inlineScripts) {
                            const newScript = document.createElement('script');
                            if (oldScript.src) {
                                newScript.src = oldScript.src;
                            } else {
                                newScript.textContent = oldScript.textContent;
                            }
                            // Appending to document body evaluates the script execution runtime thread
                            document.body.appendChild(newScript).parentNode.removeChild(newScript);
                        }
                    }, 50);
                } else {
                    // Standard text rendering path for raw conversational strings
                    contentDiv.textContent = content;
                }
                
                messageDiv.appendChild(contentDiv);
                chatMessages.appendChild(messageDiv);
                
                // Keep the chat frame scrolled down
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            function setLoading(isLoading) {
                if (isLoading) {
                    loading.style.display = 'block';
                    sendButton.disabled = true;
                    messageInput.disabled = true;
                } else {
                    loading.style.display = 'none';
                    sendButton.disabled = false;
                    messageInput.disabled = false;
                }
            }

            // Focus input on load
            messageInput.focus();
        </script>
    </body>
    </html>
    """

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
        # Use the new execute_pipeline method with conversational memory support
        answer = bot_service.execute_pipeline(
            session_id=request.session_id,
            raw_user_prompt=request.prompt,
            temperature=request.temperature
        )
        execution_time_ms = (time.time() - start_time) * 1000
        
        # The answer is already a JSON string from the chatbot service
        # Parse it once and return as proper Python object for FastAPI
        try:
            parsed_answer = json.loads(answer)
            response_data = {
                "response": parsed_answer,
                "execution_time_ms": round(execution_time_ms, 2)
            }
        except json.JSONDecodeError:
            # Fallback for non-JSON responses
            response_data = {
                "response": {"display_value": answer},
                "execution_time_ms": round(execution_time_ms, 2)
            }
        
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

# Add some helpful endpoints for development
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Fleet Management Chatbot"}

@app.get("/info")
def service_info():
    """Service information endpoint"""
    return {
        "service": "Fleet Management Chatbot",
        "version": "1.0.0",
        "endpoints": {
            "ui": "/",
            "chat_api": "/api/v1/chat",
            "history": "/api/v1/history",
            "stats": "/api/v1/history/stats",
            "search": "/api/v1/history/search"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Fleet Management Chatbot Server...")
    print("📊 API Documentation: http://localhost:8001/docs")
    print("💬 Chat Interface: http://localhost:8001/")
    print("📈 Health Check: http://localhost:8001/health")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)