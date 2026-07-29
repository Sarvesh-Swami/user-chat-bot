import json
import re
from datetime import datetime
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
from database_manager import DatabaseManager
from schema_loader import SchemaLoader
from geocoding_service import GeocodingService
from llm_client import LLMClient
from query_processor import QueryProcessor
from session_manager import SessionManager
from visualization_engine import VisualizationEngine
from pdf_report_service import PdfReportService
from email_engine import EmailEngine
from intent_parser import IntentParser
from sql_prompt_templates import SQLPromptTemplates


# Load environment variables from .env file
load_dotenv()

# Define the request body structure using Pydantic
class QueryRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    session_id: Optional[str] = None

class ChatbotService:
    """Main orchestration service for the Fleet Management Chatbot with conversational memory"""
    
    def __init__(self):
        # 1. Initialize Database Manager
        self.db_manager = DatabaseManager()
        self.schema_loader = SchemaLoader(self.db_manager)
        self.geocoding_service = GeocodingService()
        
        # 2. Load and cache database schema at startup
        self.schema_cache, self.mock_today = self.schema_loader.load_schema()
        
        # 3. Initialize LLM Client with dependencies
        self.llm_client = LLMClient(self.db_manager, self.schema_cache)
        
        # 4. Initialize Query Processor with all dependencies
        self.query_processor = QueryProcessor(
            self.db_manager, self.llm_client, self.geocoding_service,
            schema_cache=self.schema_cache
        )
        
        # 5. Initialize Session Manager for conversational memory
        self.session_manager = SessionManager()
        
        # 6. Initialize Visualization Engine for chart generation
        self.visualization_engine = VisualizationEngine(self.llm_client)
        
        # 7. Initialize PDF Report Service for report generation
        self.pdf_report_service = PdfReportService()
        
        # 8. Initialize Email Engine for sending emails
        self.email_engine = EmailEngine()
        
        # 9. Initialize Intent Parser for query validation
        self.intent_parser = IntentParser(self.llm_client, self.schema_cache)
        
        # 10. Initialize SQL Prompt Template Router
        self.sql_prompt_templates = SQLPromptTemplates()

    
    def execute_pipeline(self, session_id: Optional[str], raw_user_prompt: str, temperature: float = 0.7) -> str:
        """Main pipeline with conversational memory support and robust intent/context routing"""
        
        # 1. Ensure session exists
        session_id = self.session_manager.get_or_create_session(session_id)
        
        # 2. Fetch history context for this session
        chat_history_str = self.session_manager.format_history_for_llm(session_id)
        history = self.session_manager.get_history(session_id)
        previous_data = self.session_manager.get_last_raw_data(session_id)
        
        print(f"\n[PIPELINE START] Prompt: '{raw_user_prompt}' | Session: '{session_id}' | History Turns: {len(history)} | Cached Data: {len(previous_data) if previous_data else 0} records")

        # 2.1. Check for PDF report / Affirmative confirmation requests BEFORE Intent Parser or Conversational intercept
        pdf_keywords = ["pdf", "report", "export", "download report", "generate report", "export report", "pdf report",
                        "yes, generate pdf", "yes generate pdf", "generate pdf", "yes please", "yes, generate",
                        "should i generate", "generate a pdf", "generate pdf document"]
        is_pdf_keyword = any(keyword in raw_user_prompt.lower() for keyword in pdf_keywords)
        
        # Robust affirmative matching (handling typos: yes, yres, yea, yeah, yup, sure, ok, go ahead, etc.)
        prompt_clean = raw_user_prompt.strip().lower()
        affirmative_patterns = [
            r"^\s*(y+e+s+|y+r+e+s+|y+e+a+h*|y+u+p+|s+u+r+e+|o+k+a*y*|g+o+\s*a+h+e+a+d+|p+l+e+a+s+e+|d+o+\s*i+t+)\s*$",
            r"\b(yes|yres|yea|yeah|yup|sure|ok|okay|do it|go ahead|please|generate|export|download)\b"
        ]
        is_affirmative = (history is not None and len(history) > 0 and previous_data is not None) and any(
            re.search(pat, prompt_clean) for pat in affirmative_patterns
        )

        if (is_pdf_keyword or is_affirmative) and history:
            print(f"[PDF DETECTION] PDF request or affirmative confirmation detected for session {session_id}. Prompt: '{raw_user_prompt}'")
            if previous_data:
                result = self.pdf_report_service.generate(
                    data=previous_data,
                    title="Fleet Management Report",
                    user_question=raw_user_prompt,
                )
                if "error" in result:
                    pdf_response = {
                        "type": "text",
                        "display_value": f"Sorry, I could not generate the PDF report: {result['error']}"
                    }
                else:
                    pdf_response = {
                        "type": "pdf_report",
                        "display_value": f"Your PDF report is ready with {result['record_count']:,} records.",
                        "filename":    result["filename"],
                        "url":         result["url_path"],
                        "record_count": result["record_count"],
                    }
                    print(f"[PDF DETECTION] PDF generated successfully: {result['url_path']} ({result['record_count']} records)")
                    self.session_manager.set_last_pdf_report(session_id, result)
                
                self.session_manager.add_turn(
                    session_id=session_id,
                    user_prompt=raw_user_prompt,
                    assistant_summary=f"Generated a PDF report with {result.get('record_count', 0)} records."
                )
                return json.dumps(pdf_response)
            else:
                no_data_response = {
                    "type": "text",
                    "display_value": "I don't have any recent data to generate a PDF report from. Please run a data query first, then ask me to export it."
                }
                self.session_manager.add_turn(
                    session_id=session_id,
                    user_prompt=raw_user_prompt,
                    assistant_summary="No previous data available for PDF report generation."
                )
                return json.dumps(no_data_response)

        # 2.2. Check for Email report requests BEFORE Intent Parser or Conversational intercept
        email_pattern = r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
        email_match = re.search(email_pattern, raw_user_prompt)
        
        if email_match and history:
            recipient_email = email_match.group(0)
            print(f"[EMAIL DETECTION] Email request detected for session {session_id} to {recipient_email}.")
            
            pdf_info = self.session_manager.get_last_pdf_report(session_id)
            if not pdf_info and previous_data:
                print(f"[EMAIL DETECTION] Generating PDF report from cached raw data...")
                pdf_info = self.pdf_report_service.generate(
                    data=previous_data,
                    title="Fleet Management Report",
                    user_question=raw_user_prompt,
                )
                if pdf_info and "error" not in pdf_info:
                    self.session_manager.set_last_pdf_report(session_id, pdf_info)
            
            if pdf_info and "error" not in pdf_info:
                attachment_path = pdf_info["filepath"]
                subject = "Fleet Management Report"
                body = f"""Hello,

Please find attached the Fleet Management Report you requested.

Report Details:
- Filename: {pdf_info['filename']}
- Total Records: {pdf_info['record_count']:,}
- Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Best regards,
Fleet Management Assistant"""
                
                email_result = self.email_engine.send_email_with_attachment(
                    recipient_email=recipient_email,
                    subject=subject,
                    body_content=body,
                    attachment_path=attachment_path
                )
                
                if email_result.get("success", False):
                    email_response = {
                        "type": "email_sent",
                        "display_value": f"Report successfully emailed to {recipient_email}.",
                        "recipient": recipient_email,
                        "subject": subject,
                        "mode": email_result.get("mode", "SMTP"),
                        "filename": pdf_info["filename"]
                    }
                else:
                    email_response = {
                        "type": "text",
                        "display_value": f"Failed to send email to {recipient_email}: {email_result.get('message', 'Unknown error')}"
                    }
                
                self.session_manager.add_turn(
                    session_id=session_id,
                    user_prompt=raw_user_prompt,
                    assistant_summary=f"Emailed report to {recipient_email}."
                )
                return json.dumps(email_response)
            else:
                no_data_response = {
                    "type": "text",
                    "display_value": "I don't have any recent data to email. Please query for some data first, then ask me to email the report."
                }
                self.session_manager.add_turn(
                    session_id=session_id,
                    user_prompt=raw_user_prompt,
                    assistant_summary="No data available to email."
                )
                return json.dumps(no_data_response)

        # 2.3. Check for Chart/Graph requests BEFORE Intent Parser or Conversational intercept
        chart_keywords = ["graph", "chart", "plot", "visualize", "visual", "bar chart", "line chart", "show chart"]
        is_chart_request = any(keyword in raw_user_prompt.lower() for keyword in chart_keywords)
        
        if is_chart_request and history:
            print(f"[CHART DETECTION] Chart request detected for session {session_id}.")
            if previous_data:
                chart_response = self.visualization_engine.generate_chart_html(previous_data, raw_user_prompt)
                self.session_manager.add_turn(
                    session_id=session_id,
                    user_prompt=raw_user_prompt,
                    assistant_summary="Generated a visual chart/graph from the previous query data."
                )
                return json.dumps(chart_response)
            else:
                no_data_response = {
                    "type": "text",
                    "display_value": "I don't have any recent data to create a chart from. Please run a data query first, then ask me to visualize it."
                }
                self.session_manager.add_turn(
                    session_id=session_id,
                    user_prompt=raw_user_prompt,
                    assistant_summary="No previous data available for chart generation."
                )
                return json.dumps(no_data_response)

        # 2.4. Check query supportability using Intent Parser
        intent_result = self.intent_parser.parse(
            raw_user_prompt=raw_user_prompt,
            chat_history_str=chat_history_str
        )
        print(f"[INTENT PARSER] Output: Supported={intent_result.get('is_supported', True)} | Intent='{intent_result.get('intent')}' | Category='{intent_result.get('query_category')}' (Confidence: {intent_result.get('confidence', 1.0):.2f})")

        if not intent_result.get("is_supported", True):
            denial_msg = intent_result.get("denial_reason") or "I am sorry, but that query is not supported by our fleet management system."
            print(f"[INTENT PARSER] Query denied: '{raw_user_prompt}'. Reason: {denial_msg}")
            
            denial_response = {
                "type": "text",
                "display_value": denial_msg
            }
            self.session_manager.add_turn(
                session_id=session_id,
                user_prompt=raw_user_prompt,
                assistant_summary=f"Query denied: {denial_msg}"
            )
            return json.dumps(denial_response)
            
        # 2.5. If the intent is conversational, respond directly and bypass SQL/query building entirely
        if intent_result.get("intent") == "conversational":
            response_text = intent_result.get("response_text") or "Hello! I am your Fleet Management Assistant. How can I help you today?"
            print(f"[INTENT PARSER] Conversational reply: '{response_text}'")
            
            conv_response = {
                "type": "text",
                "display_value": response_text
            }
            self.session_manager.add_turn(
                session_id=session_id,
                user_prompt=raw_user_prompt,
                assistant_summary=response_text
            )
            return json.dumps(conv_response)
        
        # 3. Condense query if active context history exists
        if history:
            condensation_prompt = self.session_manager.get_condensation_prompt(
                chat_history_str=chat_history_str,
                current_query=raw_user_prompt
            )
            
            # Call the new lightweight Bedrock client route
            resolved_query = self.llm_client.condense_history_query(condensation_prompt)
            print(f"[QUERY CONDENSATION] Condensed '{raw_user_prompt}' into standalone query: '{resolved_query}'")
        else:
            # First turn of the conversation; query passes through natively
            resolved_query = raw_user_prompt
            print(f"[QUERY CONDENSATION] First turn - using original query: '{resolved_query}'")
        
        # 4. Extract query_category and confidence from the intent result
        query_category = intent_result.get("query_category", "default")
        confidence     = float(intent_result.get("confidence", 1.0))
        print(f"[INTENT PARSER] Category: '{query_category}', Confidence: {confidence:.2f}")
        
        # 5. Pass resolved_query + routing metadata into the Text-to-SQL pipeline
        api_response_payload = self._run_text_to_sql_pipeline(
            resolved_query, temperature, query_category, confidence, session_id=session_id
        )
        
        # 4.5. Store raw data for potential chart generation
        try:
            # Parse the API response to extract raw data
            if isinstance(api_response_payload, str):
                response_data = json.loads(api_response_payload)
            else:
                response_data = api_response_payload
                
            # Try multiple locations for raw data in the response structure
            raw_data = None
            
            # First, check metadata for raw_data (if LLM synthesis was used)
            if "metadata" in response_data:
                # Check for trip_history, history_by_date, or other data arrays
                metadata = response_data["metadata"]
                raw_data = (metadata.get("trip_history") or 
                           metadata.get("history_by_date") or 
                           metadata.get("raw_data"))
            
            # If not found in metadata, check for direct data field (if raw data was returned)
            if not raw_data and "data" in response_data:
                raw_data = response_data["data"]
                
            # Store the data if it's a valid list with records
            if raw_data and isinstance(raw_data, list) and len(raw_data) > 0:
                # Ensure the records have useful data for charting
                if isinstance(raw_data[0], dict) and len(raw_data[0]) > 1:
                    self.session_manager.set_last_raw_data(session_id, raw_data)
                    print(f"[DATA STORAGE] Stored {len(raw_data)} records for potential chart generation")
                else:
                    print(f"[DATA STORAGE] Skipped storing data - insufficient structure for charting")
            else:
                print(f"[DATA STORAGE] No chartable data found in response")
                
        except Exception as e:
            print(f"[DATA STORAGE WARNING] Could not store raw data for charting: {e}")
        
        # 5. Extract a lean textual summary from the query processor execution results
        execution_summary = self._extract_execution_summary(api_response_payload, resolved_query)
        
        # 6. Commit this turn to the rolling context log
        self.session_manager.add_turn(
            session_id=session_id,
            user_prompt=raw_user_prompt,
            assistant_summary=execution_summary
        )
        
        return api_response_payload

    def _extract_execution_summary(self, api_response_payload: str, resolved_query: str) -> str:
        """Extract a lean summary from the API response for conversation history"""
        try:
            # Parse the JSON response to extract display_value
            if isinstance(api_response_payload, str):
                response_data = json.loads(api_response_payload)
            else:
                response_data = api_response_payload
                
            # Extract the display_value which is the main user-facing response
            display_value = response_data.get("display_value", "")
            if display_value:
                # Keep it concise for conversation history - just the main answer
                if len(display_value) > 150:
                    return display_value[:150] + "..."
                return display_value
            else:
                return f"Executed query about: {resolved_query[:50]}..."
                
        except Exception as e:
            print(f"[SUMMARY EXTRACTION ERROR]: {e}")
            return f"Query processed: {resolved_query[:50]}..."

    def answer_user_query(self, user_question: str) -> str:
        """Backward compatibility method - calls execute_pipeline without session"""
        return self.execute_pipeline(session_id=None, raw_user_prompt=user_question)

    def _run_text_to_sql_pipeline(
        self,
        user_question: str,
        temperature: float = 0.7,
        query_category: str = "default",
        confidence: float = 1.0,
        confidence_threshold: float = 0.65,
        session_id: str = None
    ) -> str:
        """Main method to process user queries and return structured responses.
        
        Args:
            user_question:        The standalone (condensed) user query.
            temperature:          LLM sampling temperature.
            query_category:       Category from IntentParser (e.g. 'vehicle_location').
            confidence:           IntentParser confidence in the category (0.0–1.0).
            confidence_threshold: Minimum confidence to use focused hints. Below this,
                                  the full hint set is used as a safe fallback.
        """
        
        # Check LLM configuration
        if not self.llm_client.api_url or not self.llm_client.api_key or not self.llm_client.model_name:
            return "Configuration error: Bedrock client is not set up. Please check your BEDROCK_API_URL, BEDROCK_API_KEY, and BEDROCK_MODEL environment variables."

        schema_context = self.llm_client._get_db_schema_string()
        
        # -----------------------------------------------------------------------
        # PART 1 — Invariant core block (always sent, regardless of category).
        # Contains: role, full schema, date, output rules, all table names,
        # all relationships, and universal query guidelines.
        # This ensures the LLM can always produce correct JOINs across any tables.
        # -----------------------------------------------------------------------
        core_block = f"""\
You are an expert PostgreSQL database engineer specializing in fleet management systems.
You have access to a focused PostgreSQL database with 6 core tables that handle all fleet operations.

### Database Schema Context:
{schema_context}

### Global Environment Variables:
- Current system date in the database is: '{self.mock_today}'
- Use this date for "today", "current", "now" queries

### CRITICAL OUTPUT REQUIREMENT:
You MUST respond with ONLY the SQL query. Do NOT provide explanations, descriptions, or any other text.
Do NOT use markdown code blocks or formatting. Return ONLY the raw SQL statement.

### Core Tables Overview:
1. **vehicles**      - Master vehicle registry with identifiers and device assignments
2. **organizations** - Fleet companies/owners (root entity)
3. **devices**       - GPS tracking hardware installed in vehicles
4. **trips**         - Individual journeys with start/end points and statistics
5. **livetrack**     - Real-time GPS positions and movement data
6. **allevents**     - Safety incidents and operational events

### Key Relationships:
- vehicles.organization_id  → organizations.id  (vehicle ownership)
- vehicles.device_id        → devices.id        (vehicle has GPS device)
- devices.imei              → livetrack.imei    (GPS data from device)
- devices.imei              → allevents.imei    (events from device)
- trips.vehicle_id          → vehicles.id       (trip made by vehicle)
- trips.device_id           → devices.id        (trip device relationship)

- Never assume which identifier the user provided — always check both.

- CRITICAL FOR VEHICLE IDENTIFIER LOOKUPS: When filtering by a vehicle identifier string (e.g. 'HQNS82400027', 'VHX8N4', 'VHC1031', 'ABC123'), ALWAYS check ALL FOUR vehicle identifier fields in the WHERE clause:
  (v.vehicle_id = '[ID]' OR v.license_plate_number = '[ID]' OR v.custom_vehicle_code = '[ID]' OR v.vin = '[ID]')
  Never omit license_plate_number or vehicle_id!
- CRITICAL FOR NAME SEARCHES: ALWAYS use ILIKE '%search_term%' for organization names (o.name), driver names, vehicle makes/models, etc. NEVER use exact equality (=) or include trailing periods/punctuation. Example: o.name ILIKE '%INLAND EXPRESS USA INC%'.
- CRITICAL FOR CURRENTLY MOVING VEHICLES: To find currently moving vehicles, ALWAYS use devices.last_speed > 0 (JOIN devices d ON v.device_id = d.id). NEVER join the livetrack table or use NOW() - INTERVAL filters for current vehicle movement status.
- Use ORDER BY on timestamp columns for chronological data.
- LIMIT large result sets appropriately.
- Prefer devices.last_ping_* for current location (faster than joining livetrack).
- You may JOIN across any of the 6 core tables as needed.
"""
        
        # -----------------------------------------------------------------------
        # PART 2 — Dynamic SQL hint block (query-category-specific examples).
        # If confidence is below threshold, fall back to the full hint set so the
        # LLM has all patterns available — same behaviour as the original prompt.
        # -----------------------------------------------------------------------
        if confidence >= confidence_threshold:
            hint_block = self.sql_prompt_templates.get_hints(query_category)
            print(f"[SQL ROUTER] Using focused hints for category='{query_category}' (confidence={confidence:.2f})")
        else:
            hint_block = self.sql_prompt_templates.get_full_hints()
            print(f"[SQL ROUTER] Low confidence ({confidence:.2f}), using full hint set as fallback.")
        
        system_prompt_sql = (
            core_block
            + "\n" + hint_block
            + "\n\nREMEMBER: Return ONLY the PostgreSQL query, nothing else. "
            "Focus on the 6 core tables with vehicles as the primary entry point for user queries."
        )
        
        max_retries = 2
        current_attempt = 1
        last_error_message = None
        processed_result = None
        generated_sql = ""

        while current_attempt <= max_retries:
            try:
                if current_attempt == 1:
                    user_msg_content = f"User Request: {user_question}\nSQL Query:"
                else:
                    print(f"[SELF-HEALING LOOP] Attempt {current_attempt}/{max_retries} for query: '{user_question}'")
                    user_msg_content = f"""\
Your previous SQL attempt failed when executing on PostgreSQL.

USER REQUEST: {user_question}

FAILED SQL:
{generated_sql}

POSTGRESQL ERROR:
{last_error_message}

CRITICAL REPAIR INSTRUCTIONS:
1. Carefully inspect the PostgreSQL error message above and identify which table, column, or syntax was wrong.
2. Refer back to the Database Schema Context and SQL hints.
3. SPECIAL FIX FOR "CURRENTLY MOVING" QUERIES: If this is about finding vehicles that are currently moving, 
   ALWAYS use devices.last_speed > 0 instead of joining livetrack with NOW() - INTERVAL time filters.
   WRONG: JOIN livetrack l ... WHERE l.ts_in_str >= NOW() - INTERVAL '15 minutes'
   RIGHT: JOIN devices d ON v.device_id = d.id WHERE d.last_speed > 0
4. Fix the mistake and return ONLY the corrected SQL statement. Do NOT include any explanations or markdown formatting.
"""

                # Generate the structured SQL instruction
                sql_generation_resp = self.llm_client._call_bedrock_api(
                    messages=[
                        {"role": "system", "content": system_prompt_sql},
                        {"role": "user", "content": user_msg_content}
                    ],
                    temperature=0.0
                )
                
                generated_sql = sql_generation_resp.choices[0].message.content.strip()
                
                print(f"[LLM RESPONSE] Attempt {current_attempt} raw response length: {len(generated_sql)} characters")
                if len(generated_sql) > 200:
                    print(f"[LLM RESPONSE] First 200 chars: {generated_sql[:200]}...")
                else:
                    print(f"[LLM RESPONSE] Full response: {generated_sql}")
                
                # Clean, validate-and-correct, then syntax-check the generated SQL
                cleaned_sql  = self.llm_client._clean_sql_formatting(generated_sql)
                fixed_sql    = self.query_processor._fix_postgresql_query_issues(cleaned_sql)
                # Option 2: column validator — auto-correct wrong column names before execution
                fixed_sql    = self.query_processor._validate_and_correct_columns(fixed_sql)
                
                # Validate syntax
                is_valid, validation_message = self.query_processor._validate_sql_syntax(fixed_sql)
                if not is_valid:
                    last_error_message = f"Syntax validation error: {validation_message}"
                    print(f"[SQL VALIDATION ERROR] Attempt {current_attempt}: {validation_message}")
                    current_attempt += 1
                    continue
                
                print(f"\n[TEXT-TO-SQL LOG] (Attempt {current_attempt}) Query:\n{fixed_sql}\n")
                
                # Try executing the query against database
                processed_result = self.query_processor._execute_with_smart_limits(fixed_sql, user_question)
                # Success! Break out of retry loop
                break

            except Exception as exec_error:
                last_error_message = str(exec_error)
                print(f"[SQL EXECUTION ERROR] Attempt {current_attempt} failed: {exec_error}")
                current_attempt += 1

        if not processed_result:
            print(f"[SELF-HEALING FAILED] All {max_retries} attempts failed. Last error: {last_error_message}")
            return json.dumps({
                "status": "error",
                "display_value": "I encountered an error executing your data query. Please try rephrasing your question.",
                "message": f"Execution failed after retries: {last_error_message}"
            })

        if processed_result.get("is_detail_preview", False):
            rows = processed_result.get("data", [])
            row  = rows[0] if rows else {}
            total_rows = processed_result.get("total_rows", len(rows))

            # Priority columns to display in the preview card (in order, with friendly labels)
            PRIORITY_COLS = [
                ("vehicle_id",         "Vehicle ID"),
                ("license_plate_number", "License Plate"),
                ("make",               "Make"),
                ("model",              "Model"),
                ("year",               "Year"),
                ("status",             "Status"),
                ("vin",                "VIN"),
                ("custom_vehicle_code", "Custom Code"),
                ("organization_name",  "Organization"),
                ("location",           "Location"),
                ("last_ping_lat",      "Latitude"),
                ("last_ping_lng",      "Longitude"),
                ("last_speed",         "Speed (mph)"),
                ("device_status",      "Device Status"),
                ("imei",               "IMEI"),
                ("last_ping_ms",       "Last Ping (ms)"),
            ]

            # Build preview table: try priority cols first, then fall back to remaining non-null cols
            SKIP_COLS = {"id", "organization_id", "device_id", "created_at", "updated_at",
                         "last_ping_lat", "last_ping_lng"}  # lat/lng shown via location field
            preview_table = []
            shown_keys = set()

            for col_key, col_label in PRIORITY_COLS:
                val = row.get(col_key)
                if val not in (None, "", "None", "nan") and str(val).strip():
                    value = str(val)
                    if len(value) > 60:
                        value = value[:57] + "…"
                    preview_table.append({"label": col_label, "value": value})
                    shown_keys.add(col_key)

            # Add any remaining non-null columns not already shown (cap at 14 total)
            for col_key, val in row.items():
                if len(preview_table) >= 14:
                    break
                if col_key in shown_keys or col_key.lower() in SKIP_COLS:
                    continue
                if val in (None, "", "None", "nan") or not str(val).strip():
                    continue
                value = str(val)
                if len(value) > 60:
                    value = value[:57] + "…"
                preview_table.append({"label": col_key.replace("_", " ").title(), "value": value})

            # Extract entity name for the PDF prompt
            entity_name = (
                row.get("license_plate_number") or
                row.get("vehicle_id") or
                row.get("driver_name") or
                "this vehicle"
            )

            print(f"[DETAIL PREVIEW] Returning detail card for entity '{entity_name}' with {len(preview_table)} preview fields.")

            # Store raw data in session NOW so the PDF affirmative check can find it when user says 'yes'
            if session_id and rows:
                self.session_manager.set_last_raw_data(session_id, rows)
                print(f"[DETAIL PREVIEW] Stored {len(rows)} record(s) in session '{session_id}' for PDF generation.")

            response_payload = {
                "type": "detail_preview",
                "display_value": f"Here's a summary for **{entity_name}**. Would you like me to generate a full PDF report with all details?",
                "preview_table": preview_table,
                "pdf_prompt": f"Would you like a full PDF report with all details for {entity_name}?",
                "entity_name": str(entity_name),
                "total_cols": processed_result.get("num_cols", len(row)),
                "total_rows": total_rows,
                "data": rows
            }
            return json.dumps(response_payload)

        if processed_result.get("is_pdf_report", False):
            total_rows = processed_result.get("total_rows", 0)
            print(f"[PDF REPORT GENERATION] {total_rows} rows > 7 threshold. Generating PDF report to save tokens...")
            pdf_info = self.pdf_report_service.generate(
                data=processed_result["data"],
                title="Fleet Data Report",
                user_question=user_question
            )
            response_payload = {
                "type": "pdf_report",
                "display_value": f"Your query returned {total_rows} records. Since the dataset is larger than 7 records, I've generated a detailed PDF report for you to download.",
                "url": pdf_info.get("url_path", ""),
                "filename": pdf_info.get("filename", "fleet_report.pdf"),
                "total_rows": total_rows
            }
            return json.dumps(response_payload)

        if processed_result.get("is_summary", False):
            # Return summary / no-data response directly (bypassing synthesis LLM)
            return processed_result["response"]
        else:
            # Continue with normal LLM synthesis processing
            data_context = processed_result["data"]
            
            # Check if this is a trip summary that needs special handling
            if processed_result.get("is_trip_summary", False):
                print("[TRIP SUMMARY] Using LLM synthesis for trip summary with aggregated data")

        # 2. Polymorphic Response Aggregation Prompt
        system_prompt_synthesis = """
            You are a data reporting translation layer. Your single task is to convert raw database row matrices into a clean, structured JSON response based on the nature of the data retrieved.

            ### Dynamic Structuring Instructions:
            1. Analyze the user's question and the structural layout of the provided database rows.
            2. Determine the query context: Is it a single tracking snapshot, trip information, sequential trip timeline, daily aggregated summary, geofence evaluation, or an empty result set?
            3. **LOCATION ENHANCEMENT**: When data contains "formatted_address" field, use it for human-readable locations instead of raw coordinates.
            4. **TRIP CONTEXT**: When data contains trip information (trip_code, start_address, end_address), prioritize trip context in the response.
            
            [CASE A: Vehicle / Asset Details & Position Context]
            - If the output contains vehicle data (ID, plate, make, model, year, status, organization, location, speed, device info):
              * Set "query_topic" to "vehicle_location".
              * Construct a comprehensive, well-structured summary for "display_value" covering ALL available vehicle parameters, not just location.
              * Include Vehicle ID, License Plate, Make/Model/Year (if available), Organization (if available), Status, Speed, and Location (use formatted_address if available, or coordinates).
              * Format cleanly using bold labels or clear bullet points so all key attributes are easily readable.
              * Example:
                "**Vehicle Details — [ID] ([Plate])**:
                • **Status**: [status] | **Make/Model/Year**: [make] [model] ([year])
                • **Organization**: [organization]
                • **Location**: [formatted_address or lat/lng]
                • **Speed**: [speed] mph"

            [CASE B: Trip Information Context]
            - If the output contains trip data (trip_code, start_address, end_address, trip_status):
              * Set "query_topic" to "vehicle_trip" for current trip, "trip_history" for multiple trips, "trip_origin" for start locations
              * For current trip origin: "Vehicle [ID] started its current trip from [start_address]" 
              * For current trip destination: "Vehicle [ID] is heading to [end_address]"
              * For trip details: "Vehicle [ID] is on trip [trip_code] from [start_address] to [end_address]"
              * Include trip_distance_miles, trip_duration_seconds, trip_status in metadata

            [CASE C: Historical Journey / Trip Timeline Context]
            - If the output contains a timeline grid of multiple trip records:
              * Set "query_topic" to "trip_history".
              * Construct a summary: "Vehicle [ID] has completed [count] trips, most recent from [start] to [end]"
              * Map trip histories inside an array key named **"trip_history"** inside the "metadata" object.
              * Include both coordinates and formatted_address for each trip.
              
            [CASE C2: Trip Summary Statistics Context]
            - If the output contains aggregated trip statistics (total_trips, total_distance_miles, average_trip_score, etc.):
              * Set "query_topic" to "vehicle_daily_summary".
              * Create comprehensive summary: "Vehicle [ID] has completed a total of [total_trips] trips, covering [total_distance_miles] miles with an average trip score of [average_trip_score]. The first trip started on [first_trip_start], and the last trip ended on [last_trip_end]"
              * Include all statistics in metadata with proper formatting

            [CASE D: Daily Activity Summary Breakdown]
            - If the rows present structural breakdowns grouped by date and mode (using COUNT or SUM parameters):
              * Set "query_topic" to "vehicle_daily_summary".
              * Set a descriptive message for "display_value".
              * Output rows inside a list array named **"history_by_date"** inside the "metadata" object.

            [CASE E: Empty / No Data Results]
            - If the Database Output Matrix is empty (0 rows or []):
              * Set "query_topic" to "no_data_found".
              * Set "display_value" to a polite, friendly, context-aware answer specifically addressing the user's request.
              * Examples:
                - For moving vehicles query: "No vehicles belonging to [Organization] are currently moving at this time."
                - For organization vehicles list: "No vehicles were found registered under [Organization]."
                - For general queries: "No records found matching your query: '[User Question]'."
              * Store {"total_available": 0, "raw_data": []} inside metadata.

            ### Location Data Usage Priority:
            1. **First Priority**: Use "formatted_address" field for human-readable location names
            2. **Second Priority**: Use raw "latitude/longitude" or "last_ping_lat/last_ping_lng" coordinates
            3. **Format**: Always include both formatted location AND coordinates in metadata for completeness
            """
        try:
            final_resp = self.llm_client._call_bedrock_api(
                messages=[
                    {"role": "system", "content": system_prompt_synthesis},
                    {"role": "user", "content": f"User Question: {user_question}\nDatabase Output Matrix:\n{data_context}"}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            response_content = final_resp.choices[0].message.content.strip()
            
            # Inject raw records for potential charting / PDF export if present
            raw_rows = processed_result.get("data") or processed_result.get("raw_records")
            if raw_rows and isinstance(raw_rows, list):
                try:
                    response_json = json.loads(response_content)
                    response_json["data"] = raw_rows
                    if "metadata" not in response_json:
                        response_json["metadata"] = {}
                    response_json["metadata"]["raw_data"] = raw_rows
                    response_content = json.dumps(response_json, default=self.db_manager._json_serializer)
                except Exception as inject_err:
                    print(f"[RAW RECORDS INJECTION ERROR]: {inject_err}")
            
            # Add information about limited results if applicable
            if processed_result.get("is_limited", False):
                try:
                    response_json = json.loads(response_content)
                    total_rows = processed_result.get("total_rows", 0)
                    showing = processed_result.get("showing", 0)
                    query_type = processed_result.get("query_type", "unknown")
                    
                    # Add limitation info to the response with query type context
                    current_display = response_json.get("display_value", "")
                    
                    if query_type == "general":
                        limited_note = f"\n\n📋 **General Query Results**: Showing top {showing} of {total_rows:,} vehicles to avoid timeouts. For specific vehicle details, ask about particular vehicle IDs, driver names, or locations."
                    else:
                        limited_note = f"\n\n📊 **Note**: Showing {showing} of {total_rows:,} total records. Add more specific filters to see targeted results."
                    
                    response_json["display_value"] = current_display + limited_note
                    
                    # Add metadata about limitation and query type
                    if "metadata" not in response_json:
                        response_json["metadata"] = {}
                    response_json["metadata"]["result_limited"] = True
                    response_json["metadata"]["total_available"] = total_rows
                    response_json["metadata"]["records_shown"] = showing
                    response_json["metadata"]["query_type"] = query_type
                    
                    return json.dumps(response_json)
                    
                except (json.JSONDecodeError, KeyError):
                    # If JSON parsing fails, return original response
                    pass
            
            return response_content
            
        except Exception as e:
            return f'{{"status": "error", "message": "Encountered processing failure: {str(e)}"}}'