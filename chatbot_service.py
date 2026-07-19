import json
import re
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
        self.query_processor = QueryProcessor(self.db_manager, self.llm_client, self.geocoding_service)
        
        # 5. Initialize Session Manager for conversational memory
        self.session_manager = SessionManager()
        
        # 6. Initialize Visualization Engine for chart generation
        self.visualization_engine = VisualizationEngine(self.llm_client)
    
    def execute_pipeline(self, session_id: Optional[str], raw_user_prompt: str, temperature: float = 0.7) -> str:
        """Main pipeline with conversational memory support"""
        
        # 1. Ensure session exists
        session_id = self.session_manager.get_or_create_session(session_id)
        
        # 2. Fetch history context for this session
        chat_history_str = self.session_manager.format_history_for_llm(session_id)
        history = self.session_manager.get_history(session_id)
        
        # 2.5. Check for chart/graph requests BEFORE query condensation
        chart_keywords = ["graph", "chart", "plot", "visualize", "visual", "bar chart", "line chart", "show chart"]
        is_chart_request = any(keyword in raw_user_prompt.lower() for keyword in chart_keywords)
        
        if is_chart_request and history:
            # User wants a graph of what we just discussed! Retrieve the PREVIOUS database result.
            print(f"[CHART DETECTION] Chart request detected for session {session_id}. Generating visualization.")
            
            # Get the last stored raw data from the session
            previous_data = self.session_manager.get_last_raw_data(session_id)
            
            if previous_data:
                # Generate the chart HTML using the visualization engine
                chart_response = self.visualization_engine.generate_chart_html(previous_data, raw_user_prompt)
                
                # Commit this chart generation turn to history
                self.session_manager.add_turn(
                    session_id=session_id,
                    user_prompt=raw_user_prompt,
                    assistant_summary="Generated a visual chart/graph from the previous query data."
                )
                
                # Return the chart response as JSON
                return json.dumps(chart_response)
            else:
                # No previous data available for charting
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
        
        # 4. Pass ONLY the resolved_query into the original Text-to-SQL execution flow
        api_response_payload = self._run_text_to_sql_pipeline(resolved_query, temperature)
        
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

    def _run_text_to_sql_pipeline(self, user_question: str, temperature: float = 0.7) -> str:
        """Main method to process user queries and return structured responses"""
        
        # Check LLM configuration
        if not self.llm_client.api_url or not self.llm_client.api_key or not self.llm_client.model_name:
            return "Configuration error: Bedrock client is not set up. Please check your BEDROCK_API_URL, BEDROCK_API_KEY, and BEDROCK_MODEL environment variables."

        schema_context = self.llm_client._get_db_schema_string()
        
        # 1. SQL Generation Prompt for PostgreSQL Database
        system_prompt_sql = f"""
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
        
        ### CRITICAL: Trip Query Pattern Recognition

        **User Intent Recognition for Trip Queries:**
        
        **Origin/Source Queries** (→ start_address, start_latitude/longitude):
        - "Where is vehicle [ID] coming from?"
        - "Where did vehicle [ID] start?"
        - "Origin of vehicle [ID]"
        - "Starting point of vehicle [ID]"
        - "Where did [ID] begin its journey?"
        
        **Destination Queries** (→ end_address, end_latitude/longitude):
        - "Where is vehicle [ID] going?"
        - "Destination of vehicle [ID]"
        - "Where is [ID] headed?"
        - "End point for vehicle [ID]"
        
        **Current Trip Queries** (→ active trip with trip_status = 'Started'):
        - "Current trip for vehicle [ID]"
        - "Active trip details for [ID]"
        - "What trip is vehicle [ID] on?"
        - "Trip in progress for [ID]"
        
        **Trip History Queries** (→ multiple trips, ORDER BY start_date/end_date):
        - "Trip history for vehicle [ID]"
        - "All trips for vehicle [ID]"
        - "Recent trips for [ID]"
        - "Show trips for vehicle [ID]"
        
        **Last Trip Queries** (→ most recent completed trip):
        - "Last trip for vehicle [ID]"
        - "Previous trip for [ID]"
        - "Most recent completed trip"

        **KEY RELATIONSHIPS FOR TRIP QUERIES:**
        - vehicles.id = trips.vehicle_id (PRIMARY relationship)
        - trips.trip_status = 'Started' (for active trips)
        - trips.trip_status = 'Completed' (for finished trips)
        - Use ORDER BY start_date DESC or end_date DESC for recency
        ### CRITICAL: User Query Interpretation Rules

        **When users ask "Where is vehicle [IDENTIFIER]" or "Location of [IDENTIFIER]":**
        - The [IDENTIFIER] could be EITHER vehicle_id OR license_plate_number
        - Users don't distinguish between these - they just say "vehicle ABC123"
        - **ALWAYS use**: WHERE v.vehicle_id = 'ABC123' OR v.license_plate_number = 'ABC123'
        - **NEVER assume** which column the identifier belongs to
        
        **Examples of user queries and required WHERE clauses:**
        - "Where is vehicle HQNS82400038" → WHERE v.vehicle_id = 'HQNS82400038' OR v.license_plate_number = 'HQNS82400038'
        - "Location of ABC123" → WHERE v.vehicle_id = 'ABC123' OR v.license_plate_number = 'ABC123'
        - "Show vehicle XYZ789" → WHERE v.vehicle_id = 'XYZ789' OR v.license_plate_number = 'XYZ789'

        ### Core Tables Overview:
        1. **vehicles** - Master vehicle registry with identifiers and device assignments
        2. **organizations** - Fleet companies/owners (root entity)
        3. **devices** - GPS tracking hardware installed in vehicles  
        4. **trips** - Individual journeys with start/end points and statistics
        5. **livetrack** - Real-time GPS positions and movement data
        6. **allevents** - Safety incidents and operational events

        ### Key Relationships:
        - vehicles.organization_id → organizations.id (vehicle ownership)
        - vehicles.device_id → devices.id (vehicle has GPS device)
        - devices.imei → livetrack.imei (GPS data from device)
        - devices.imei → allevents.imei (events from device)
        - trips.vehicle_id → vehicles.id (trip made by vehicle)
        - trips.device_id → devices.id (trip device relationship)
        ### Vehicle Location Queries (MOST IMPORTANT):

        **1. "Where is vehicle [ID]?" or "Location of license plate [PLATE]":**
        ```sql
        SELECT v.vehicle_id, v.license_plate_number, d.last_ping_lat, d.last_ping_lng, d.last_speed, d.last_ping_ms
        FROM vehicles v 
        JOIN devices d ON v.device_id = d.id 
        WHERE v.vehicle_id = '[ID]' OR v.license_plate_number = '[ID]';
        ```
        
        **CRITICAL: Always use OR condition to check BOTH vehicle_id AND license_plate_number because users may provide either identifier!**

        **2. For real-time tracking history:**
        ```sql
        SELECT v.vehicle_id, v.license_plate_number, l.latitude, l.longitude, l.speed, l.ts_in_str
        FROM vehicles v 
        JOIN devices d ON v.device_id = d.id 
        JOIN livetrack l ON d.imei = l.imei 
        WHERE v.vehicle_id = '[ID]' OR v.license_plate_number = '[ID]'
        ORDER BY l.ts_in_str DESC LIMIT 10;
        ```

        **3. For current status with organization:**
        ```sql
        SELECT v.vehicle_id, v.license_plate_number, o.name as organization, 
               d.last_ping_lat, d.last_ping_lng, d.last_speed
        FROM vehicles v 
        JOIN organizations o ON v.organization_id = o.id
        JOIN devices d ON v.device_id = d.id 
        WHERE v.vehicle_id = '[ID]' OR v.license_plate_number = '[ID]';
        ```
        ### Trip-Related Queries (HIGHLY IMPORTANT):

        **4. "Where is vehicle [ID] coming from?" or "Where did vehicle start?" (Current Trip Origin):**
        ```sql
        SELECT v.vehicle_id, v.license_plate_number, t.trip_code, t.start_address, 
               t.start_latitude, t.start_longitude, t.start_date, t.trip_status
        FROM vehicles v 
        JOIN trips t ON v.id = t.vehicle_id 
        WHERE (v.vehicle_id = '[ID]' OR v.license_plate_number = '[ID]')
        AND t.trip_status = 'Started'
        ORDER BY t.start_date DESC LIMIT 1;
        ```

        **5. "Where is vehicle [ID] going?" or "Current destination" (Current Trip Destination):**
        ```sql
        SELECT v.vehicle_id, v.license_plate_number, t.trip_code, t.end_address,
               t.end_latitude, t.end_longitude, t.start_date, t.trip_status
        FROM vehicles v 
        JOIN trips t ON v.id = t.vehicle_id 
        WHERE (v.vehicle_id = '[ID]' OR v.license_plate_number = '[ID]')
        AND t.trip_status = 'Started'
        ORDER BY t.start_date DESC LIMIT 1;
        ```

        **6. "Current trip for vehicle [ID]" or "Active trip details":**
        ```sql
        SELECT v.vehicle_id, v.license_plate_number, t.trip_code, t.start_address, t.end_address,
               t.start_date, t.end_date, t.trip_distance_miles, t.trip_duration_seconds, 
               t.trip_status, t.trip_score
        FROM vehicles v 
        JOIN trips t ON v.id = t.vehicle_id 
        WHERE (v.vehicle_id = '[ID]' OR v.license_plate_number = '[ID]')
        AND t.trip_status = 'Started'
        ORDER BY t.start_date DESC LIMIT 1;
        ```
        **7. "Last trip for vehicle [ID]" or "Most recent completed trip":**
        ```sql
        SELECT v.vehicle_id, v.license_plate_number, t.trip_code, t.start_address, t.end_address,
               t.start_date, t.end_date, t.trip_distance_miles, t.trip_duration_seconds, t.trip_score
        FROM vehicles v 
        JOIN trips t ON v.id = t.vehicle_id 
        WHERE (v.vehicle_id = '[ID]' OR v.license_plate_number = '[ID]')
        AND t.trip_status = 'Completed'
        ORDER BY t.end_date DESC LIMIT 1;
        ```

        **8. "Trip history for vehicle [ID]" or "Show all trips":**
        ```sql
        SELECT v.vehicle_id, v.license_plate_number, t.trip_code, t.start_address, t.end_address,
               t.start_date, t.end_date, t.trip_distance_miles, t.trip_status
        FROM vehicles v 
        JOIN trips t ON v.id = t.vehicle_id 
        WHERE v.vehicle_id = '[ID]' OR v.license_plate_number = '[ID]'
        ORDER BY t.start_date DESC LIMIT 10;
        ```

        ### PostgreSQL Query Guidelines:
        - **ALWAYS check BOTH vehicle_id AND license_plate_number** when users ask for a vehicle
        - **MANDATORY**: Use OR condition: WHERE v.vehicle_id = 'USER_INPUT' OR v.license_plate_number = 'USER_INPUT'
        - Users may provide either the internal vehicle_id OR the license plate number
        - **NEVER assume which identifier type the user provided - always check both!**
        - Use ORDER BY timestamp columns for chronological data
        - LIMIT results for large datasets
        - Prioritize devices.last_ping_* for current location (faster than livetrack)

        REMEMBER: Return ONLY the PostgreSQL query, nothing else. Focus on the 6 core tables with vehicles as the primary entry point for user queries.
        """
        
        try:
            # Generate the structured SQL instruction
            sql_generation_resp = self.llm_client._call_bedrock_api(
                messages=[
                    {"role": "system", "content": system_prompt_sql},
                    {"role": "user", "content": f"User Request: {user_question}\nSQL Query:"}
                ],
                temperature=0.0
            )
            
            generated_sql = sql_generation_resp.choices[0].message.content.strip()
            
            print(f"[LLM RESPONSE] Raw response length: {len(generated_sql)} characters")
            if len(generated_sql) > 200:
                print(f"[LLM RESPONSE] First 200 chars: {generated_sql[:200]}...")
            else:
                print(f"[LLM RESPONSE] Full response: {generated_sql}")
            
            # Clean and validate the generated SQL
            cleaned_sql = self.llm_client._clean_sql_formatting(generated_sql)
            
            # Fix PostgreSQL-specific query issues
            fixed_sql = self.query_processor._fix_postgresql_query_issues(cleaned_sql)
            
            # Validate the fixed SQL
            is_valid, validation_message = self.query_processor._validate_sql_syntax(fixed_sql)
            
            if not is_valid:
                print(f"[SQL VALIDATION ERROR]: {validation_message}")
                print(f"[ORIGINAL SQL]: {generated_sql}")
                print(f"[CLEANED SQL]: {cleaned_sql}")
                print(f"[FIXED SQL]: {fixed_sql}")
                return '{"status": "error", "message": "Generated SQL query has syntax issues. Please try rephrasing your question."}'
            print(f"\n[TEXT-TO-SQL LOG] Original Query:\n{generated_sql}")
            print(f"\n[TEXT-TO-SQL LOG] Cleaned Query:\n{cleaned_sql}")
            if fixed_sql != cleaned_sql:
                print(f"\n[TEXT-TO-SQL LOG] SQLite-Fixed Query:\n{fixed_sql}")
            print()
            
            # Row count check and smart data handling with query classification
            processed_result = self.query_processor._execute_with_smart_limits(fixed_sql, user_question)
            
            if processed_result.get("is_summary", False):
                # Return summary response directly
                return processed_result["response"]
            else:
                # Continue with normal LLM processing
                data_context = processed_result["data"]
                
                # Check if this is a trip summary that needs special handling
                if processed_result.get("is_trip_summary", False):
                    print("[TRIP SUMMARY] Using LLM synthesis for trip summary with aggregated data")
            # 2. Polymorphic Response Aggregation Prompt
            system_prompt_synthesis = """
            You are a data reporting translation layer. Your single task is to convert raw database row matrices into a clean, structured JSON response based on the nature of the data retrieved.

            ### Dynamic Structuring Instructions:
            1. Analyze the user's question and the structural layout of the provided database rows.
            2. Determine the query context: Is it a single tracking snapshot, trip information, sequential trip timeline, daily aggregated summary, or geofence evaluation?
            3. **LOCATION ENHANCEMENT**: When data contains "formatted_address" field, use it for human-readable locations instead of raw coordinates.
            4. **TRIP CONTEXT**: When data contains trip information (trip_code, start_address, end_address), prioritize trip context in the response.
            
            [CASE A: Single Snapshot / Latest State Context]
            - If the output contains only 1 row tracking an asset's position:
              * Set "query_topic" to "vehicle_location".
              * For "display_value", use formatted_address if available: "Vehicle [ID] is located at [formatted_address]"
              * If no formatted_address, fall back to coordinates: "Vehicle [ID] is at coordinates [lat], [lng]"
              * Store all parameters (vid, drivername, coordinates, formatted_address, speed) inside the "metadata" root.

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

            ### Location Data Usage Priority:
            1. **First Priority**: Use "formatted_address" field for human-readable location names
            2. **Second Priority**: Use raw "latitude/longitude" or "last_ping_lat/last_ping_lng" coordinates
            3. **Format**: Always include both formatted location AND coordinates in metadata for completeness
            """
            final_resp = self.llm_client._call_bedrock_api(
                messages=[
                    {"role": "system", "content": system_prompt_synthesis},
                    {"role": "user", "content": f"User Question: {user_question}\nDatabase Output Matrix:\n{data_context}"}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            response_content = final_resp.choices[0].message.content.strip()
            
            # Inject raw records for potential charting if present
            if "raw_records" in processed_result and processed_result["raw_records"]:
                try:
                    response_json = json.loads(response_content)
                    if "metadata" not in response_json:
                        response_json["metadata"] = {}
                    response_json["metadata"]["raw_data"] = processed_result["raw_records"]
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