import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from chatbot_service import ChatbotService, QueryRequest
from query_history import QueryHistoryManager, QueryLog
import time
from typing import List

# Ensure local database is running before starting the app
from start_db import ensure_db_running

print("\n" + "="*70)
print("   FLEET MANAGEMENT CHATBOT - STARTING")
print("="*70)

# Auto-start the local PostgreSQL database
if not ensure_db_running():
    print("\n[STARTUP ERROR] Failed to start local database!")
    print("[STARTUP] Please run: python setup_local_db.py (one-time setup)")
    import sys
    sys.exit(1)

print("[STARTUP] ✓ Database ready")
print("="*70 + "\n")

# Initialize the FastAPI app
app = FastAPI(title="Fleet Management Chatbot API", version="1.0.0")

# Ensure the reports directory exists and mount it for static file serving
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")

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
            
            .typing-indicator-content {
                display: flex;
                align-items: center;
                gap: 6px;
                padding: 14px 18px;
                min-height: 42px;
            }
            
            .typing-dot {
                width: 8px;
                height: 8px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 50%;
                animation: typingPulse 1.4s infinite ease-in-out;
            }
            
            .typing-dot:nth-child(1) {
                animation-delay: 0s;
            }
            
            .typing-dot:nth-child(2) {
                animation-delay: 0.2s;
            }
            
            .typing-dot:nth-child(3) {
                animation-delay: 0.4s;
            }
            
            @keyframes typingPulse {
                0%, 60%, 100% {
                    transform: translateY(0);
                    opacity: 0.4;
                }
                30% {
                    transform: translateY(-6px);
                    opacity: 1;
                }
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
            </div>
        </div>

        <script>
            const chatMessages = document.getElementById('chatMessages');
            const messageInput = document.getElementById('messageInput');
            const sendButton = document.getElementById('sendButton');
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
                        let isPdfReport = false;
                        let pdfUrl = '';
                        let pdfLabel = '';
                        let isEmailSent = false;
                        let emailRecipient = '';
                        let emailSubject = '';
                        let emailMode = '';
                        let emailFilename = '';
                        
                        let isDetailPreview = false;
                        let detailPreviewData = null;
                        
                        // Check if the response payload contains a PDF report
                        if (data.response && data.response.type === 'pdf_report') {
                            isPdfReport = true;
                            botResponse = data.response.display_value || 'PDF report ready.';
                            pdfUrl   = data.response.url || '';
                            pdfLabel = data.response.filename || 'report.pdf';
                        // Check if the response is a detail preview (wide-record "all details" queries)
                        } else if (data.response && data.response.type === 'detail_preview') {
                            isDetailPreview = true;
                            detailPreviewData = data.response;
                        // Check if the response payload contains an email sent confirmation
                        } else if (data.response && data.response.type === 'email_sent') {
                            isEmailSent = true;
                            botResponse = data.response.display_value || 'Email dispatched successfully.';
                            emailRecipient = data.response.recipient || '';
                            emailSubject = data.response.subject || '';
                            emailMode = data.response.mode || '';
                            emailFilename = data.response.filename || '';
                        // Check if the response payload contains a visual media graph configuration
                        } else if (data.response && data.response.type === 'rich_media') {
                            botResponse = data.response.display_value;
                            htmlSnippet = data.response.html_content;
                            isRichMedia = true;
                        } else if (data.response && typeof data.response === 'object' && data.response.display_value) {
                            botResponse = data.response.display_value;
                        } else if (data.response && typeof data.response === 'string') {
                            botResponse = data.response;
                        }
                        
                        // Add processing metadata timing parameters
                        if (data.execution_time_ms && !isDetailPreview) {
                            botResponse += `\n\n⏱️ ${data.execution_time_ms.toFixed(0)}ms`;
                        }
                        
                        // Send indicators to the rendering pipeline
                        if (isDetailPreview) {
                            addDetailPreviewCard(detailPreviewData, data.execution_time_ms);
                        } else if (isPdfReport) {
                            addPdfCard(botResponse, pdfUrl, pdfLabel);
                        } else if (isEmailSent) {
                            addEmailCard(botResponse, emailRecipient, emailSubject, emailMode, emailFilename);
                        } else {
                            addMessage(botResponse, 'bot', isRichMedia, htmlSnippet);
                        }
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

            function addDetailPreviewCard(previewData, execTimeMs) {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message bot';

                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                contentDiv.style.cssText = 'max-width: 520px;';

                // ── Header label ──
                const entityName = previewData.entity_name || 'Record';
                const totalCols  = previewData.total_cols  || 0;

                const headerDiv = document.createElement('div');
                headerDiv.style.cssText = 'margin-bottom: 10px; font-size: 13px; color: #475569;';
                headerDiv.innerHTML = `Here's a <strong>summary</strong> for <strong style="color:#6366f1;">${entityName}</strong> (${totalCols} fields in total):`;
                contentDiv.appendChild(headerDiv);

                // ── Preview table card ──
                const tableCard = document.createElement('div');
                tableCard.style.cssText = `
                    background: linear-gradient(135deg, #f8f9ff 0%, #ede9fe 100%);
                    border: 1px solid #c4b5fd;
                    border-radius: 12px;
                    overflow: hidden;
                    margin-bottom: 12px;
                    box-shadow: 0 2px 8px rgba(99,102,241,0.08);
                `;

                const table = document.createElement('table');
                table.style.cssText = 'width: 100%; border-collapse: collapse; font-size: 13px;';

                const rows = previewData.preview_table || [];
                rows.forEach((row, idx) => {
                    const tr = document.createElement('tr');
                    tr.style.cssText = idx % 2 === 0
                        ? 'background: rgba(255,255,255,0.55);'
                        : 'background: rgba(237,233,254,0.45);';

                    const tdLabel = document.createElement('td');
                    tdLabel.style.cssText = `
                        padding: 8px 14px;
                        font-weight: 600;
                        color: #4c1d95;
                        width: 42%;
                        border-bottom: 1px solid rgba(196,181,253,0.3);
                        white-space: nowrap;
                    `;
                    tdLabel.textContent = row.label;

                    const tdValue = document.createElement('td');
                    tdValue.style.cssText = `
                        padding: 8px 14px;
                        color: #1e293b;
                        border-bottom: 1px solid rgba(196,181,253,0.3);
                        word-break: break-word;
                    `;
                    tdValue.textContent = row.value;

                    tr.appendChild(tdLabel);
                    tr.appendChild(tdValue);
                    table.appendChild(tr);
                });

                tableCard.appendChild(table);
                contentDiv.appendChild(tableCard);

                // ── PDF prompt bar ──
                const promptBar = document.createElement('div');
                promptBar.style.cssText = `
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    background: #f1f5f9;
                    border: 1px solid #e2e8f0;
                    border-radius: 10px;
                    padding: 10px 14px;
                `;

                const promptText = document.createElement('span');
                promptText.style.cssText = 'flex: 1; font-size: 13px; color: #475569;';
                promptText.textContent = '📄 Would you like a full PDF report with all details?';
                promptBar.appendChild(promptText);

                // Yes button
                const yesBtn = document.createElement('button');
                yesBtn.textContent = 'Yes, Generate PDF';
                yesBtn.style.cssText = `
                    padding: 7px 14px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-size: 12px;
                    font-weight: 600;
                    cursor: pointer;
                    white-space: nowrap;
                    transition: opacity 0.2s;
                `;
                yesBtn.onmouseover = () => yesBtn.style.opacity = '0.85';
                yesBtn.onmouseout  = () => yesBtn.style.opacity = '1';
                yesBtn.onclick = () => {
                    // Auto-submit "Yes, generate PDF" into the chat
                    const input = document.getElementById('messageInput');
                    if (input) {
                        input.value = 'Yes, generate PDF';
                        document.getElementById('chatForm').dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
                    }
                    // Disable buttons after click to avoid double submit
                    yesBtn.disabled = true;
                    noBtn.disabled  = true;
                    yesBtn.style.opacity = '0.5';
                    noBtn.style.opacity  = '0.5';
                };
                promptBar.appendChild(yesBtn);

                // No button
                const noBtn = document.createElement('button');
                noBtn.textContent = 'No thanks';
                noBtn.style.cssText = `
                    padding: 7px 12px;
                    background: transparent;
                    color: #64748b;
                    border: 1px solid #cbd5e1;
                    border-radius: 8px;
                    font-size: 12px;
                    font-weight: 500;
                    cursor: pointer;
                    white-space: nowrap;
                    transition: background 0.2s;
                `;
                noBtn.onmouseover = () => noBtn.style.background = '#f1f5f9';
                noBtn.onmouseout  = () => noBtn.style.background  = 'transparent';
                noBtn.onclick = () => {
                    promptBar.style.display = 'none';
                };
                promptBar.appendChild(noBtn);

                contentDiv.appendChild(promptBar);

                // ── Timing badge ──
                if (execTimeMs) {
                    const timingDiv = document.createElement('div');
                    timingDiv.style.cssText = 'font-size: 11px; color: #94a3b8; margin-top: 6px; text-align: right;';
                    timingDiv.textContent = `⏱️ ${execTimeMs.toFixed(0)}ms`;
                    contentDiv.appendChild(timingDiv);
                }

                messageDiv.appendChild(contentDiv);
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            function addPdfCard(summaryText, pdfUrl, filename) {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message bot';

                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';

                // Summary text
                const textNode = document.createElement('div');
                textNode.textContent = summaryText;
                textNode.style.marginBottom = '12px';
                contentDiv.appendChild(textNode);

                // PDF download card
                const card = document.createElement('div');
                card.style.cssText = `
                    background: linear-gradient(135deg, #f8f9ff 0%, #ede9fe 100%);
                    border: 1px solid #c4b5fd;
                    border-radius: 12px;
                    padding: 16px;
                    display: flex;
                    align-items: center;
                    gap: 14px;
                    margin-top: 4px;
                `;

                // PDF icon
                const icon = document.createElement('div');
                icon.style.cssText = `
                    width: 44px; height: 44px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 10px;
                    display: flex; align-items: center; justify-content: center;
                    font-size: 22px; flex-shrink: 0;
                `;
                icon.textContent = '📄';

                // Info block
                const info = document.createElement('div');
                info.style.flex = '1';

                const title = document.createElement('div');
                title.style.cssText = 'font-weight: 600; font-size: 14px; color: #1e293b;';
                title.textContent = 'Fleet Report Ready';

                const sub = document.createElement('div');
                sub.style.cssText = 'font-size: 11px; color: #64748b; margin-top: 2px;';
                sub.textContent = filename;

                info.appendChild(title);
                info.appendChild(sub);

                // Download button
                const btn = document.createElement('a');
                btn.href = pdfUrl;
                btn.target = '_blank';
                btn.download = filename;
                btn.style.cssText = `
                    padding: 8px 16px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border-radius: 8px;
                    font-size: 13px;
                    font-weight: 600;
                    text-decoration: none;
                    white-space: nowrap;
                    transition: opacity 0.2s;
                `;
                btn.textContent = '⬇ Download PDF';
                btn.onmouseover = () => btn.style.opacity = '0.85';
                btn.onmouseout  = () => btn.style.opacity = '1';

                card.appendChild(icon);
                card.appendChild(info);
                card.appendChild(btn);
                contentDiv.appendChild(card);

                messageDiv.appendChild(contentDiv);
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            function addEmailCard(summaryText, recipient, subject, mode, filename) {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message bot';

                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';

                // Summary text
                const textNode = document.createElement('div');
                textNode.textContent = summaryText;
                textNode.style.marginBottom = '12px';
                contentDiv.appendChild(textNode);

                // Email confirmation card
                const card = document.createElement('div');
                card.style.cssText = `
                    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
                    border: 1px solid #86efac;
                    border-radius: 12px;
                    padding: 16px;
                    display: flex;
                    align-items: center;
                    gap: 14px;
                    margin-top: 4px;
                `;

                // Envelope icon
                const icon = document.createElement('div');
                icon.style.cssText = `
                    width: 44px; height: 44px;
                    background: linear-gradient(135deg, #22c55e 0%, #15803d 100%);
                    border-radius: 10px;
                    display: flex; align-items: center; justify-content: center;
                    font-size: 22px; flex-shrink: 0;
                `;
                icon.textContent = '✉️';

                // Info block
                const info = document.createElement('div');
                info.style.flex = '1';

                const title = document.createElement('div');
                title.style.cssText = 'font-weight: 600; font-size: 14px; color: #14532d;';
                title.textContent = 'Email Dispatched';

                const sub = document.createElement('div');
                sub.style.cssText = 'font-size: 11px; color: #166534; margin-top: 2px; line-height: 1.4;';
                sub.innerHTML = `To: <b>${recipient}</b><br/>Subject: <i>${subject}</i><br/>Attachment: <code>${filename || 'None'}</code>`;

                info.appendChild(title);
                info.appendChild(sub);

                // Status Badge
                const badge = document.createElement('div');
                badge.style.cssText = `
                    padding: 6px 12px;
                    background: ${mode === 'Mock Mode' ? '#fef3c7' : '#dcfce7'};
                    color: ${mode === 'Mock Mode' ? '#d97706' : '#15803d'};
                    border: 1px solid ${mode === 'Mock Mode' ? '#fcd34d' : '#86efac'};
                    border-radius: 20px;
                    font-size: 11px;
                    font-weight: 600;
                    white-space: nowrap;
                `;
                badge.textContent = mode;

                card.appendChild(icon);
                card.appendChild(info);
                card.appendChild(badge);
                contentDiv.appendChild(card);

                messageDiv.appendChild(contentDiv);
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            function setLoading(isLoading) {
                let typingIndicator = document.getElementById('typingIndicator');
                if (isLoading) {
                    sendButton.disabled = true;
                    messageInput.disabled = true;
                    
                    if (!typingIndicator) {
                        typingIndicator = document.createElement('div');
                        typingIndicator.className = 'message bot';
                        typingIndicator.id = 'typingIndicator';
                        
                        const contentDiv = document.createElement('div');
                        contentDiv.className = 'message-content typing-indicator-content';
                        
                        for (let i = 0; i < 3; i++) {
                            const dot = document.createElement('div');
                            dot.className = 'typing-dot';
                            contentDiv.appendChild(dot);
                        }
                        
                        typingIndicator.appendChild(contentDiv);
                        chatMessages.appendChild(typingIndicator);
                        chatMessages.scrollTop = chatMessages.scrollHeight;
                    }
                } else {
                    sendButton.disabled = false;
                    messageInput.disabled = false;
                    if (typingIndicator) {
                        typingIndicator.remove();
                    }
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