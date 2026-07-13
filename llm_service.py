import os
import sqlite3
import pandas as pd
import requests
import json
import re
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the request body structure using Pydantic
class QueryRequest(BaseModel):
    prompt: str
    temperature: float = 0.7

class ChatbotService:
    def __init__(self):
        # 1. Initialize Bedrock Client configuration
        self.api_url = os.getenv("BEDROCK_API_URL")
        self.api_key = os.getenv("BEDROCK_API_KEY")
        self.model_name = os.getenv("BEDROCK_MODEL")
        
        # Setup headers for Bedrock API
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 2. Setup In-Memory SQLite Database
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        
        # --- NEW: REGISTER CUSTOM NATIVE MATH FUNCTION ---
        import math
        def calculate_distance(lat1, lng1, lat2, lng2):
            if None in (lat1, lng1, lat2, lng2):
                return 99999999.0
            try:
                rad_lat1, rad_lng1, rad_lat2, rad_lng2 = map(math.radians, [float(lat1), float(lng1), float(lat2), float(lng2)])
                return 6371000 * math.acos(
                    math.cos(rad_lat1) * math.cos(rad_lat2) * math.cos(rad_lng2 - rad_lng1) +
                    math.sin(rad_lat1) * math.sin(rad_lat2)
                )
            except Exception:
                return 99999999.0

        self.conn.create_function("CALCULATE_DISTANCE", 4, calculate_distance)
        self._load_csv_data()
    
    def _call_bedrock_api(self, messages, temperature=0.0, response_format=None):
        """Helper method to make API calls to Bedrock"""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature
        }
        
        # Add response_format if specified (for JSON responses)
        if response_format:
            payload["response_format"] = response_format
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=60  # Increased from 30 to 60 seconds
            )
            response.raise_for_status()
            
            # Create a mock response object similar to OpenAI/Groq format
            response_data = response.json()
            
            # Create a simple object to mimic the expected response structure
            class MockChoice:
                def __init__(self, content):
                    self.message = type('obj', (object,), {'content': content})
            
            class MockResponse:
                def __init__(self, content):
                    self.choices = [MockChoice(content)]
            
            # Extract content from Bedrock response
            content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            return MockResponse(content)
            
        except requests.exceptions.Timeout:
            raise Exception(f"Bedrock API timeout - request took longer than 60 seconds")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Bedrock API call failed: {str(e)}")
        except (KeyError, IndexError) as e:
            raise Exception(f"Unexpected Bedrock API response format: {str(e)}")
    
    def _clean_sql_formatting(self, sql_query):
        """Clean and fix common SQL formatting issues from LLM generation"""
        
        # Remove leading/trailing whitespace
        cleaned = sql_query.strip()
        
        # Step 1: Extract SQL from markdown code blocks if present
        # Look for ```sql ... ``` or ``` ... ``` patterns
        sql_block_match = re.search(r'```(?:sql)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL | re.IGNORECASE)
        if sql_block_match:
            print("[SQL EXTRACTION] Found SQL in markdown code block")
            cleaned = sql_block_match.group(1).strip()
        
        # Step 2: If no markdown blocks, try to extract SQL from verbose response
        # Look for patterns like "Here's the SQL:" or "The query is:" followed by SQL
        elif not self._looks_like_sql(cleaned):
            print("[SQL EXTRACTION] Response doesn't look like SQL, attempting extraction...")
            
            # Try to find SQL embedded in explanation text
            sql_patterns = [
                # Pattern 1: SQL after common phrases
                r'(?:here\'s? (?:the )?|the )?(?:sql (?:query|statement)?|query) (?:is|would be)?:?\s*\n?\s*(SELECT.*?)(?:\n|$)',
                
                # Pattern 2: SQL in parentheses or quotes  
                r'["\'\(]*(SELECT.*?)["\'\)]*(?:\s*$|\n|\.)',
                
                # Pattern 3: SQL at end of explanation
                r'.*?(?:query|sql|statement).*?\n\s*(SELECT.*?)(?:\s*$|\n)',
                
                # Pattern 4: Direct SQL detection (starts with SELECT, WITH, INSERT, etc.)
                r'^\s*((?:SELECT|WITH|INSERT|UPDATE|DELETE).*?)(?:\s*$|\n)',
                
                # Pattern 5: Find standalone SQL lines in multi-line text
                r'\n\s*(SELECT.*?);?\s*(?:\n|$)'
            ]
            
            extracted_sql = None
            for i, pattern in enumerate(sql_patterns):
                matches = re.findall(pattern, cleaned, re.DOTALL | re.IGNORECASE)
                if matches:
                    # Get the longest match (likely the most complete SQL)
                    extracted_sql = max(matches, key=len).strip()
                    print(f"[SQL EXTRACTION] Pattern {i+1} extracted SQL: {extracted_sql[:100]}...")
                    break
            
            if extracted_sql and self._looks_like_sql(extracted_sql):
                cleaned = extracted_sql
                print("[SQL EXTRACTION] Successfully extracted SQL from verbose response")
            else:
                print("[SQL EXTRACTION] Could not extract valid SQL from response")
                # Keep original response - might be a simple SQL statement
        
        # Step 3: Validate extracted content looks like SQL
        if not self._looks_like_sql(cleaned):
            print(f"[SQL EXTRACTION] Warning: Extracted content doesn't look like SQL: {cleaned[:100]}...")
        
        # Step 4: Clean and format the SQL
        # Remove any remaining markdown artifacts
        cleaned = re.sub(r"```(?:sql)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        cleaned = cleaned.strip()
        
        # Fix missing spaces around SQL keywords
        # SELECT keyword
        cleaned = re.sub(r'\bSELECT(?=[a-zA-Z_])', 'SELECT ', cleaned, flags=re.IGNORECASE)
        
        # FROM keyword  
        cleaned = re.sub(r'\bFROM(?=[a-zA-Z_])', 'FROM ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'(?<=[a-zA-Z_\d])FROM\b', ' FROM', cleaned, flags=re.IGNORECASE)
        
        # WHERE keyword
        cleaned = re.sub(r'\bWHERE(?=[a-zA-Z_])', 'WHERE ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'(?<=[a-zA-Z_\d])WHERE\b', ' WHERE', cleaned, flags=re.IGNORECASE)
        
        # ORDER BY keywords (fix the regex to handle ORDER BY as a unit)
        cleaned = re.sub(r'(?<=[a-zA-Z_\d])(ORDER\s*BY)\b', r' \1', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\b(ORDER\s*BY)(?=[a-zA-Z_])', r'\1 ', cleaned, flags=re.IGNORECASE)
        
        # GROUP BY keywords
        cleaned = re.sub(r'\bGROUP\s*BY(?=[a-zA-Z_])', 'GROUP BY ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'(?<=[a-zA-Z_\d])GROUP\s*BY\b', ' GROUP BY', cleaned, flags=re.IGNORECASE)
        
        # INNER/LEFT/RIGHT JOIN keywords
        cleaned = re.sub(r'(?<=[a-zA-Z_\d])(INNER\s*JOIN|LEFT\s*JOIN|RIGHT\s*JOIN|JOIN)\b', r' \1', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\b(INNER\s*JOIN|LEFT\s*JOIN|RIGHT\s*JOIN|JOIN)(?=[a-zA-Z_])', r'\1 ', cleaned, flags=re.IGNORECASE)
        
        # Fix ORDER BY specifically (handle the malformed case first)
        cleaned = re.sub(r'(?<=[a-zA-Z_\d\'"])ORDER\s*BY(?=[a-zA-Z_])', ' ORDER BY ', cleaned, flags=re.IGNORECASE)
        
        # Fix GROUP BY specifically  
        cleaned = re.sub(r'(?<=[a-zA-Z_\d\'"])GROUP\s*BY(?=[a-zA-Z_])', ' GROUP BY ', cleaned, flags=re.IGNORECASE)
        
        # Now fix AND/OR, but use more specific patterns
        cleaned = re.sub(r'(?<=[a-zA-Z_\d\'"])\bAND\b(?=[a-zA-Z_])', ' AND ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'(?<=[a-zA-Z_\d\'"])\bOR\b(?=[a-zA-Z_])', ' OR ', cleaned, flags=re.IGNORECASE)
        
        # Fix spacing around commas in SELECT lists
        cleaned = re.sub(r',(?=[a-zA-Z_])', ', ', cleaned)
        cleaned = re.sub(r'(?<=[a-zA-Z_\d]),', ', ', cleaned)
        
        # Fix spacing around operators
        cleaned = re.sub(r'(?<=[a-zA-Z_\d])=(?=[a-zA-Z_\d\'"])', ' = ', cleaned)
        cleaned = re.sub(r'(?<=[a-zA-Z_\d])!=(?=[a-zA-Z_\d\'"])', ' != ', cleaned)
        cleaned = re.sub(r'(?<=[a-zA-Z_\d])>(?=[a-zA-Z_\d\'"])', ' > ', cleaned)
        cleaned = re.sub(r'(?<=[a-zA-Z_\d])<(?=[a-zA-Z_\d\'"])', ' < ', cleaned)
        
        # Clean up extra spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        # Ensure semicolon at end if not present
        if not cleaned.endswith(';'):
            cleaned += ';'
            
        return cleaned
    
    def _looks_like_sql(self, text):
        """Check if text looks like a SQL query"""
        if not text or not isinstance(text, str):
            return False
        
        text_upper = text.strip().upper()
        
        # Check if it starts with common SQL keywords
        sql_starters = ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']
        starts_with_sql = any(text_upper.startswith(keyword) for keyword in sql_starters)
        
        # Check for basic SQL keywords presence
        has_sql_keywords = any(keyword in text_upper for keyword in ['SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE'])
        
        # Check if it's mostly SQL-like (no long sentences)
        lines = text.strip().split('\n')
        sql_like_lines = 0
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
            # Check if line contains SQL keywords or looks like SQL structure
            if (any(keyword in line_clean.upper() for keyword in ['SELECT', 'FROM', 'WHERE', 'ORDER BY', 'GROUP BY', 'INSERT', 'UPDATE']) or
                re.search(r'[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[\'"]?[a-zA-Z0-9-]+[\'"]?', line_clean)):
                sql_like_lines += 1
        
        # Consider it SQL-like if it starts with SQL keyword OR has SQL keywords and most lines look SQL-like
        is_sql_like = starts_with_sql or (has_sql_keywords and sql_like_lines >= len([l for l in lines if l.strip()]) * 0.5)
        
        return is_sql_like
    
    def _validate_sql_syntax(self, sql_query):
        """Basic SQL syntax validation"""
        
        # Check for required keywords
        sql_upper = sql_query.upper()
        
        # Must have SELECT
        if 'SELECT' not in sql_upper:
            return False, "Missing SELECT keyword"
        
        # Must have FROM for most queries (except some special cases)
        if 'FROM' not in sql_upper and 'UNION' not in sql_upper:
            return False, "Missing FROM keyword"
        
        # Check for balanced parentheses
        open_parens = sql_query.count('(')
        close_parens = sql_query.count(')')
        if open_parens != close_parens:
            return False, "Unbalanced parentheses"
        
        # Check for basic structure
        if sql_query.strip().endswith(','):
            return False, "Query ends with comma"
        
        return True, "Valid"
    
    def _fix_sqlite_window_function_issues(self, sql_query):
        """Fix SQLite window function limitations by converting complex LAG queries to CTE-based approach"""
        
        sql_upper = sql_query.upper()
        
        # Check if this is a geofence entry/exit query with LAG function issues
        if ('LAG(' in sql_upper and 
            'GEOFENCE' in sql_upper and 
            'CALCULATE_DISTANCE' in sql_upper and
            ('ENTRY' in sql_upper or 'EXIT' in sql_upper)):
            
            print("[SQLITE FIX] Detected problematic geofence LAG query, converting to CTE approach")
            
            # Check if geofence_lookup table exists
            try:
                cursor = self.conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='geofence_lookup'")
                if not cursor.fetchone():
                    print("[SQLITE FIX] Warning: geofence_lookup table not found, cannot apply fix")
                    return sql_query
            except Exception as e:
                print(f"[SQLITE FIX] Could not check for geofence_lookup table: {e}")
                return sql_query
            
            # Extract vehicle ID if present
            vid_match = re.search(r'vid\s*=\s*(\d+)', sql_query, re.IGNORECASE)
            vehicle_id = vid_match.group(1) if vid_match else None
            
            if vehicle_id:
                # Generate SQLite-compatible geofence entry/exit query using CTEs
                fixed_query = f"""
                WITH vehicle_positions AS (
                    SELECT 
                        fh.vid,
                        fh.lat,
                        fh.lng,
                        fh.date,
                        fh.gpstime,
                        fh.date || ' ' || fh.gpstime AS full_timestamp,
                        gl.geofence_id,
                        gl.geofence_name,
                        gl.lat as geofence_lat,
                        gl.lng as geofence_lng,
                        gl.radius_meters,
                        CALCULATE_DISTANCE(fh.lat, fh.lng, gl.lat, gl.lng) AS distance_to_geofence,
                        CASE WHEN CALCULATE_DISTANCE(fh.lat, fh.lng, gl.lat, gl.lng) <= gl.radius_meters 
                             THEN 1 ELSE 0 END AS is_inside,
                        ROW_NUMBER() OVER (
                            PARTITION BY fh.vid, gl.geofence_id 
                            ORDER BY fh.date ASC, fh.gpstime ASC
                        ) AS row_num
                    FROM fleet_history fh
                    CROSS JOIN geofence_lookup gl
                    WHERE fh.vid = {vehicle_id}
                ),
                position_with_previous AS (
                    SELECT 
                        vid,
                        geofence_id,
                        geofence_name,
                        full_timestamp,
                        is_inside,
                        distance_to_geofence,
                        LAG(is_inside, 1, 0) OVER (
                            PARTITION BY vid, geofence_id 
                            ORDER BY date ASC, gpstime ASC
                        ) AS prev_inside,
                        date,
                        gpstime
                    FROM vehicle_positions
                ),
                entry_exit_events AS (
                    SELECT 
                        vid,
                        geofence_name,
                        full_timestamp AS event_timestamp,
                        CASE 
                            WHEN prev_inside = 0 AND is_inside = 1 THEN 'ENTRY'
                            WHEN prev_inside = 1 AND is_inside = 0 THEN 'EXIT'
                            ELSE NULL
                        END AS event_type,
                        ROUND(distance_to_geofence, 2) AS distance_meters,
                        date,
                        gpstime
                    FROM position_with_previous
                    WHERE (prev_inside = 0 AND is_inside = 1) OR (prev_inside = 1 AND is_inside = 0)
                )
                SELECT 
                    vid,
                    geofence_name,
                    event_type,
                    event_timestamp,
                    distance_meters
                FROM entry_exit_events
                WHERE event_type IS NOT NULL
                ORDER BY date ASC, gpstime ASC;
                """
                
                print("[SQLITE FIX] Successfully converted to CTE-based query")
                
                # Test the query syntax by preparing it (without executing)
                try:
                    cursor = self.conn.cursor()
                    cursor.execute("EXPLAIN QUERY PLAN " + fixed_query.strip())
                    print("[SQLITE FIX] CTE query syntax validated successfully")
                    return fixed_query.strip()
                except Exception as e:
                    print(f"[SQLITE FIX] CTE query validation failed: {e}")
                    print("[SQLITE FIX] Falling back to simplified geofence query")
                    
                    # Fallback to a simpler query that just gets positions and distances
                    simple_fallback = f"""
                    SELECT 
                        fh.vid,
                        fh.date,
                        fh.gpstime,
                        fh.date || ' ' || fh.gpstime AS timestamp,
                        gl.geofence_name,
                        ROUND(CALCULATE_DISTANCE(fh.lat, fh.lng, gl.lat, gl.lng), 2) AS distance_meters,
                        gl.radius_meters,
                        CASE 
                            WHEN CALCULATE_DISTANCE(fh.lat, fh.lng, gl.lat, gl.lng) <= gl.radius_meters 
                            THEN 'INSIDE' 
                            ELSE 'OUTSIDE' 
                        END AS status
                    FROM fleet_history fh 
                    CROSS JOIN geofence_lookup gl 
                    WHERE fh.vid = {vehicle_id}
                    ORDER BY fh.date ASC, fh.gpstime ASC;
                    """
                    
                    try:
                        cursor.execute("EXPLAIN QUERY PLAN " + simple_fallback.strip())
                        print("[SQLITE FIX] Simple fallback query validated successfully")
                        return simple_fallback.strip()
                    except Exception as fallback_error:
                        print(f"[SQLITE FIX] Even simple fallback failed: {fallback_error}")
                        return sql_query
            else:
                print("[SQLITE FIX] No vehicle ID found in query, cannot apply geofence fix")
                return sql_query
        
        # Check for other complex LAG usage patterns that might fail
        elif 'LAG(' in sql_upper and ('WHERE' in sql_upper and 'LAG(' in sql_query[sql_query.upper().find('WHERE'):]):
            print("[SQLITE FIX] Detected LAG() in WHERE clause, attempting to restructure")
            
            # For other LAG issues, try to move LAG to a CTE
            try:
                # Simple pattern: if LAG is used in WHERE, try to restructure
                # This is a basic fix - more complex cases might need specific handling
                if 'ORDER BY' in sql_upper:
                    # Extract the main SELECT part and convert to CTE structure
                    select_part = re.search(r'SELECT\s+.*?FROM', sql_query, re.IGNORECASE | re.DOTALL)
                    if select_part:
                        print("[SQLITE FIX] Restructuring LAG query with CTE")
                        # For now, return original query with a warning
                        # This can be expanded for specific cases as they arise
                        pass
            except Exception as e:
                print(f"[SQLITE FIX] Could not restructure LAG query: {e}")
        
        # Return original query if no fixes were applied
        return sql_query
    
    def _classify_query_type(self, user_question, sql_query):
        """Classify if query is general/broad or specific using LLM intent analysis"""
        
        try:
            # Use LLM to classify the query intent
            classification_prompt = """Analyze this user query about fleet/vehicle management and classify it as either "GENERAL" or "SPECIFIC".

GENERAL queries ask for:
- Broad overviews or lists (all running vehicles, stopped vehicles, vehicle statuses)
- Multiple vehicles without specifying which ones
- Current status of fleet or vehicle categories
- General reports or summaries
Examples: "show running vehicles", "list idle vehicles", "vehicles that are stopped", "I want to see all vehicles"

SPECIFIC queries ask for:
- Particular vehicles by ID, name, or identifier
- Specific drivers by name
- Vehicles at specific locations or geofences
- Historical tracking of specific assets
- Detailed information about identified entities
Examples: "show vehicle ABC123", "find driver John Smith", "vehicles at Delhi depot", "track vehicle 12345"

User Query: "{query}"

Respond with exactly one word: GENERAL or SPECIFIC"""

            # Make a quick LLM call for classification
            classification_response = self._call_bedrock_api(
                messages=[
                    {"role": "user", "content": classification_prompt.format(query=user_question)}
                ],
                temperature=0.0  # Use 0 temperature for consistent classification
            )
            
            classification = classification_response.choices[0].message.content.strip().upper()
            
            # Validate the response and extract classification
            if "GENERAL" in classification:
                print(f"[QUERY CLASSIFICATION] LLM classified as: GENERAL")
                return "general"
            elif "SPECIFIC" in classification:
                print(f"[QUERY CLASSIFICATION] LLM classified as: SPECIFIC") 
                return "specific"
            else:
                print(f"[QUERY CLASSIFICATION] Unexpected LLM response: '{classification}', defaulting to SPECIFIC")
                return "specific"
                
        except Exception as e:
            print(f"[QUERY CLASSIFICATION] LLM classification failed: {e}")
            print("[QUERY CLASSIFICATION] Falling back to regex-based classification")
            
            # Fallback to simplified regex-based classification
            return self._fallback_regex_classification(user_question, sql_query)
    
    def _fallback_regex_classification(self, user_question, sql_query):
        """Fallback regex-based classification if LLM fails"""
        
        user_lower = user_question.lower()
        
        # Simple fallback patterns for general queries
        general_patterns = [
            r'\b(?:show|list|display|see|get|find)\s+(?:all|running|stopped|idle|moving|any)\b',
            r'\b(?:all|any)\s+(?:vehicles?|assets?|fleet)\b',
            r'\bvehicles?\s+(?:that\s+are|with|having)\s+(?:running|stopped|idle)\b',
            r'\b(?:running|stopped|idle|moving)(?:\s*,\s*(?:running|stopped|idle|moving))*.*vehicles?\b',
            r'\bwhat\s+(?:vehicles?|assets?)\s+are\b',
            r'\bcurrent\s+(?:status|state)\b',
            r'\bi\s+want\s+to\s+see.*(?:running|stopped|idle)\b',
        ]
        
        # Check for general patterns
        for pattern in general_patterns:
            if re.search(pattern, user_lower):
                print(f"[QUERY CLASSIFICATION] Regex fallback: GENERAL (matched: {pattern})")
                return "general"
        
        # Simple patterns for specific queries
        specific_patterns = [
            r'\bvehicle\s+[a-zA-Z0-9]{3,}\b',
            r'\bvid\s*[:\s]*[a-zA-Z0-9]{3,}\b',
            r'\bdriver\s+[a-zA-Z]+(?:\s+[a-zA-Z]+)?\b',
            r'\b(?:at|near|in|from)\s+[A-Z][a-z]+\b',
        ]
        
        for pattern in specific_patterns:
            if re.search(pattern, user_lower):
                print(f"[QUERY CLASSIFICATION] Regex fallback: SPECIFIC (matched: {pattern})")
                return "specific"
        
        print(f"[QUERY CLASSIFICATION] Regex fallback: defaulting to GENERAL")
        return "general"  # Default to general for better user experience
    
    def _json_serializer(self, obj):
        """JSON serializer for objects not serializable by default json code"""
        if hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif isinstance(obj, (pd.Int64Dtype, pd.Float64Dtype)):
            return int(obj) if pd.notna(obj) else None
        return str(obj)  # Fallback to string representation
    
    def _convert_dataframe_to_serializable_dict(self, df):
        """Convert DataFrame to JSON-serializable dict with all numpy types converted"""
        try:
            # Convert DataFrame to list of dictionaries
            records = df.to_dict(orient="records")
            
            # Process each record to ensure JSON serializability
            serializable_records = []
            for record in records:
                serializable_record = {}
                for key, value in record.items():
                    # Convert numpy types to Python native types
                    if hasattr(value, 'item'):  # numpy scalar
                        serializable_record[key] = value.item()
                    elif pd.isna(value):  # Handle NaN/None values
                        serializable_record[key] = None
                    elif isinstance(value, (pd.Int64Dtype, pd.Float64Dtype)):
                        serializable_record[key] = int(value) if pd.notna(value) else None
                    else:
                        serializable_record[key] = value
                
                serializable_records.append(serializable_record)
            
            return serializable_records
            
        except Exception as e:
            print(f"[SERIALIZATION ERROR]: Failed to convert DataFrame: {e}")
            # Fallback: use json.dumps with custom serializer
            try:
                return json.loads(json.dumps(df.to_dict(orient="records"), default=self._json_serializer))
            except Exception as fallback_error:
                print(f"[SERIALIZATION FALLBACK ERROR]: {fallback_error}")
                return []
    
    def _is_specific_vehicle_query(self, sql_query):
        """Check if query is looking for a specific vehicle ID"""
        return bool(re.search(r'\bvid\s*=\s*\d+', sql_query, re.IGNORECASE))
    
    def _extract_vehicle_id(self, sql_query):
        """Extract vehicle ID from SQL query"""
        match = re.search(r'\bvid\s*=\s*(\d+)', sql_query, re.IGNORECASE)
        return match.group(1) if match else None
    
    def _get_last_known_data(self, vehicle_id):
        """Get the most recent data for a specific vehicle"""
        try:
            fallback_sql = f"""
            SELECT vid, speed, mode, date, gpstime, addr, drivername
            FROM fleet_history 
            WHERE vid = {vehicle_id} 
            ORDER BY date DESC, gpstime DESC 
            LIMIT 1
            """
            
            df_result = pd.read_sql_query(fallback_sql, self.conn)
            
            if len(df_result) > 0:
                # Convert to dict and ensure JSON serializable types
                record = df_result.iloc[0].to_dict()
                for key, value in record.items():
                    if hasattr(value, 'item'):  # numpy scalar
                        record[key] = value.item()
                return record
            else:
                return None
                
        except Exception as e:
            print(f"[FALLBACK QUERY ERROR]: {e}")
            return None
    
    def _generate_vehicle_not_found_response(self, vehicle_id, last_known_data):
        """Generate helpful response when vehicle has no current data"""
        
        if last_known_data:
            # Format the location (truncate if too long)
            location = str(last_known_data.get('addr', 'Unknown location'))[:50]
            if len(str(last_known_data.get('addr', ''))) > 50:
                location += "..."
            
            return json.dumps({
                "status": "no_current_data",
                "query_type": "vehicle_status",
                "vehicle_id": int(vehicle_id),
                "message": f"No data available for vehicle {vehicle_id} on 2026-06-20 (today)",
                "last_known": {
                    "date": str(last_known_data.get('date', 'Unknown')),
                    "time": str(last_known_data.get('gpstime', 'Unknown')),
                    "speed": float(last_known_data.get('speed', 0)),
                    "status": str(last_known_data.get('mode', 'Unknown')),
                    "location": location,
                    "driver": str(last_known_data.get('drivername', 'Not assigned'))
                },
                "suggestions": [
                    f"Check recent history: 'vehicle {vehicle_id} yesterday'",
                    f"View last week data: 'vehicle {vehicle_id} last week'",
                    f"Get route history: 'vehicle {vehicle_id} route history'",
                    "Contact fleet manager if vehicle should be active"
                ],
                "note": f"Last seen on {last_known_data.get('date', 'unknown date')} at {location}"
            }, default=self._json_serializer)
        else:
            return json.dumps({
                "status": "vehicle_not_found",
                "query_type": "vehicle_status", 
                "vehicle_id": int(vehicle_id),
                "message": f"Vehicle {vehicle_id} not found in the fleet database",
                "suggestions": [
                    "Verify the vehicle ID is correct",
                    "Check if vehicle is registered in system",
                    "Contact fleet administrator",
                    "Try: 'show all vehicles' to see available fleet"
                ],
                "note": "This vehicle ID does not exist in our records"
            }, default=self._json_serializer)
    
    def _execute_with_smart_limits(self, sql_query, user_question):
        """Execute SQL with intelligent row count management and auto-summarization"""
        
        # Classify query type to determine appropriate limits
        query_type = self._classify_query_type(user_question, sql_query)
        
        # Configuration thresholds based on query type
        if query_type == "general":
            SMALL_THRESHOLD = 5      # Very small for general queries to avoid timeouts
            MEDIUM_THRESHOLD = 15    # Still small for general queries
            print(f"[SMART LIMITS] Detected GENERAL query - using conservative limits")
        else:
            SMALL_THRESHOLD = 50     # Normal limits for specific queries
            MEDIUM_THRESHOLD = 200   
            print(f"[SMART LIMITS] Detected SPECIFIC query - using standard limits")
        
        try:
            # Step 1: Try to get row count first
            count_sql = f"SELECT COUNT(*) as row_count FROM ({sql_query.rstrip(';')})"
            
            try:
                count_result = pd.read_sql_query(count_sql, self.conn)
                total_rows = int(count_result['row_count'].iloc[0])  # Ensure it's a Python int
                print(f"[SMART LIMITS] Query will return {total_rows} rows (Type: {query_type.upper()})")
            
            except Exception as count_error:
                print(f"[SMART LIMITS] Count query failed: {count_error}")
                print("[SMART LIMITS] Falling back to direct execution")
                
                # Fallback: execute original query directly
                try:
                    df_result = pd.read_sql_query(sql_query, self.conn)
                    total_rows = len(df_result)
                    print(f"[SMART LIMITS] Direct execution returned {total_rows} rows")
                    
                    # Check for empty results on specific vehicle queries
                    if total_rows == 0 and self._is_specific_vehicle_query(sql_query):
                        vehicle_id = self._extract_vehicle_id(sql_query)
                        if vehicle_id:
                            print(f"[VEHICLE FALLBACK] No current data for vehicle {vehicle_id}, checking last known data")
                            last_known = self._get_last_known_data(vehicle_id)
                            return {
                                "is_summary": True,
                                "response": self._generate_vehicle_not_found_response(vehicle_id, last_known),
                                "total_rows": 0,
                                "query_type": query_type
                            }
                    
                    # Apply limits after execution if needed
                    if total_rows <= SMALL_THRESHOLD:
                        return {
                            "is_summary": False,
                            "data": self._convert_dataframe_to_serializable_dict(df_result),
                            "total_rows": int(total_rows),
                            "query_type": query_type
                        }
                    elif total_rows <= MEDIUM_THRESHOLD:
                        limited_df = df_result.head(SMALL_THRESHOLD)
                        return {
                            "is_summary": False,
                            "data": self._convert_dataframe_to_serializable_dict(limited_df),
                            "total_rows": int(total_rows),
                            "is_limited": True,
                            "showing": len(limited_df),
                            "query_type": query_type
                        }
                    else:
                        # Generate summary from the actual data
                        summary_response = self._generate_summary_from_dataframe(df_result, user_question, total_rows)
                        return {
                            "is_summary": True,
                            "response": summary_response,
                            "total_rows": total_rows,
                            "query_type": query_type
                        }
                        
                except Exception as exec_error:
                    print(f"[SMART LIMITS] Direct execution also failed: {exec_error}")
                    raise exec_error
            
            # Step 2: Decide execution strategy based on row count and query type
            if total_rows <= SMALL_THRESHOLD:
                # Small result: return full data
                df_result = pd.read_sql_query(sql_query, self.conn)
                
                # Check for empty results on specific vehicle queries
                if len(df_result) == 0 and self._is_specific_vehicle_query(sql_query):
                    vehicle_id = self._extract_vehicle_id(sql_query)
                    if vehicle_id:
                        print(f"[VEHICLE FALLBACK] No current data for vehicle {vehicle_id}, checking last known data")
                        last_known = self._get_last_known_data(vehicle_id)
                        return {
                            "is_summary": True,
                            "response": self._generate_vehicle_not_found_response(vehicle_id, last_known),
                            "total_rows": 0,
                            "query_type": query_type
                        }
                
                return {
                    "is_summary": False,
                    "data": self._convert_dataframe_to_serializable_dict(df_result),
                    "total_rows": int(total_rows),
                    "query_type": query_type
                }
                
            elif total_rows <= MEDIUM_THRESHOLD:
                # Medium result: return limited data with context
                limited_sql = f"{sql_query.rstrip(';')} LIMIT {SMALL_THRESHOLD};"
                df_result = pd.read_sql_query(limited_sql, self.conn)
                
                return {
                    "is_summary": False,
                    "data": self._convert_dataframe_to_serializable_dict(df_result),
                    "total_rows": int(total_rows),
                    "is_limited": True,
                    "showing": len(df_result),
                    "query_type": query_type
                }
                
            else:
                # Large result: return summary only
                summary_response = self._generate_summary_response(sql_query, user_question, total_rows)
                return {
                    "is_summary": True,
                    "response": summary_response,
                    "total_rows": total_rows,
                    "query_type": query_type
                }
                
        except Exception as e:
            print(f"[SMART LIMITS ERROR]: {e}")
            # Final fallback to original execution
            try:
                df_result = pd.read_sql_query(sql_query, self.conn)
                total_rows = len(df_result)
                
                # Check for empty results on specific vehicle queries
                if total_rows == 0 and self._is_specific_vehicle_query(sql_query):
                    vehicle_id = self._extract_vehicle_id(sql_query)
                    if vehicle_id:
                        print(f"[VEHICLE FALLBACK] No current data for vehicle {vehicle_id}, checking last known data")
                        last_known = self._get_last_known_data(vehicle_id)
                        return {
                            "is_summary": True,
                            "response": self._generate_vehicle_not_found_response(vehicle_id, last_known),
                            "total_rows": 0,
                            "query_type": query_type
                        }
                
                return {
                    "is_summary": False,
                    "data": self._convert_dataframe_to_serializable_dict(df_result),
                    "query_type": query_type
                }
            except Exception as final_error:
                print(f"[FINAL FALLBACK ERROR]: {final_error}")
                raise final_error
    
    def _generate_summary_response(self, original_sql, user_question, total_rows):
        """Generate intelligent summary for large result sets"""
        
        try:
            # Analyze the original query to understand intent
            sql_lower = original_sql.lower()
            
            # Generate appropriate summary queries based on table and intent
            summary_queries = []
            
            # Remove semicolon from original SQL for use in subqueries
            clean_original_sql = original_sql.rstrip(';')
            
            # Status/Mode breakdown
            if "fleet_history" in sql_lower or "vehicles" in sql_lower:
                summary_queries.extend([
                    "SELECT mode, COUNT(*) as count FROM ({}) GROUP BY mode ORDER BY count DESC".format(clean_original_sql),
                    "SELECT date, COUNT(*) as records FROM ({}) GROUP BY date ORDER BY date DESC LIMIT 7".format(clean_original_sql)
                ])
                
                # Add location analysis if location fields present
                if "lat" in sql_lower or "lng" in sql_lower:
                    summary_queries.append(
                        "SELECT SUBSTR(addr, 1, 50) as location, COUNT(*) as count FROM ({}) GROUP BY SUBSTR(addr, 1, 50) ORDER BY count DESC LIMIT 10".format(clean_original_sql)
                    )
                
                # Add speed analysis if available
                if "speed" in sql_lower:
                    summary_queries.append(
                        "SELECT ROUND(AVG(speed), 2) as avg_speed, MAX(speed) as max_speed, MIN(speed) as min_speed, COUNT(*) as records FROM ({}) WHERE speed > 0".format(clean_original_sql)
                    )
            
            # Execute summary queries
            summary_data = {}
            for i, query in enumerate(summary_queries):
                try:
                    df_summary = pd.read_sql_query(query, self.conn)
                    # Convert numpy types to Python types for JSON serialization using our helper method
                    summary_dict = self._convert_dataframe_to_serializable_dict(df_summary)
                    summary_data[f"summary_{i+1}"] = summary_dict
                except Exception as e:
                    print(f"[SUMMARY QUERY ERROR]: {e}")
                    continue
            
            # Format the summary response in structured data format
            response_data = {
                "status": "success",
                "query_type": "fleet_summary", 
                "total_records": int(total_rows),
                "summary": {},
                "suggestions": []
            }
            
            # Process each summary section into structured format
            for key, data in summary_data.items():
                if "summary_1" in key and data:  # Mode/Status breakdown
                    total_vehicles = sum(row.get('count', 0) for row in data)
                    status_breakdown = []
                    for row in data:
                        mode = row.get('mode', 'Unknown')
                        count = int(row.get('count', 0))
                        percentage = round((count / total_vehicles * 100), 1) if total_vehicles > 0 else 0
                        status_breakdown.append({
                            "status": mode,
                            "count": count,
                            "percentage": percentage
                        })
                    response_data["summary"]["vehicle_status"] = status_breakdown
                
                elif "summary_2" in key and data:  # Date breakdown
                    date_breakdown = []
                    for row in data[:5]:  # Show top 5 dates
                        date = row.get('date', 'Unknown')
                        records = int(row.get('records', 0))
                        date_breakdown.append({
                            "date": date,
                            "records": records
                        })
                    response_data["summary"]["recent_activity"] = date_breakdown
                
                elif "summary_3" in key and data:  # Location breakdown
                    location_breakdown = []
                    for row in data[:5]:  # Show top 5 locations
                        location = row.get('location', 'Unknown')[:40]  # Truncate long names
                        count = int(row.get('count', 0))
                        location_breakdown.append({
                            "location": location,
                            "count": count
                        })
                    response_data["summary"]["top_locations"] = location_breakdown
                
                elif "summary_4" in key and data:  # Speed analysis
                    if data and len(data) > 0:
                        speed_data = data[0]
                        response_data["summary"]["speed_stats"] = {
                            "average": float(speed_data.get('avg_speed', 0)),
                            "maximum": float(speed_data.get('max_speed', 0)),
                            "minimum": float(speed_data.get('min_speed', 0)),
                            "unit": "km/h"
                        }
            
            # Add helpful suggestions based on the data
            suggestions = []
            
            # Status-based suggestions
            if "vehicle_status" in response_data["summary"]:
                status_list = response_data["summary"]["vehicle_status"]
                if status_list:
                    top_status = status_list[0]["status"].lower()
                    suggestions.append(f"Filter by specific status: 'show {top_status} vehicles'")
            
            # Location-based suggestions  
            if "top_locations" in response_data["summary"]:
                locations = response_data["summary"]["top_locations"]
                if locations:
                    top_location = locations[0]["location"].split('_')[0]  # Get first part
                    suggestions.append(f"Search by location: 'vehicles at {top_location}'")
            
            # Speed-based suggestions
            if "speed_stats" in response_data["summary"]:
                avg_speed = response_data["summary"]["speed_stats"]["average"]
                suggestions.append(f"Speed analysis: 'vehicles with speed > {int(avg_speed)}'")
            
            # General suggestions
            suggestions.extend([
                "Get specific vehicle details: 'show vehicle [ID]'",
                "View by time period: 'vehicles yesterday'",
                "Check geofence status: 'vehicles outside geofence'"
            ])
            
            response_data["suggestions"] = suggestions[:5]  # Limit to 5 suggestions
            
            # Add tip about dataset size
            if total_rows > 50:
                response_data["note"] = f"Dataset contains {total_rows:,} records. Use filters above for detailed views."
            
            return json.dumps(response_data, default=self._json_serializer)
            
        except Exception as e:
            print(f"[SUMMARY GENERATION ERROR]: {e}")
            # Fallback summary with structured format
            return json.dumps({
                "status": "success",
                "query_type": "fleet_summary",
                "total_records": int(total_rows),
                "summary": {
                    "message": f"Query returned {int(total_rows):,} records. This dataset is too large to analyze in detail."
                },
                "suggestions": [
                    "Add date filter: 'vehicles today'",
                    "Filter by status: 'show running vehicles'", 
                    "Search by location: 'vehicles at [location]'",
                    "Get specific vehicle: 'show vehicle [ID]'"
                ],
                "note": "Use the filters above to get detailed vehicle information."
            }, default=self._json_serializer)
    
    def _generate_summary_from_dataframe(self, df, user_question, total_rows):
        """Generate summary from an actual dataframe when count query fails"""
        
        try:
            response_data = {
                "status": "success",
                "query_type": "fleet_summary",
                "total_records": int(total_rows),
                "summary": {},
                "suggestions": []
            }
            
            # Basic dataset overview
            response_data["summary"]["dataset_info"] = {
                "total_records": int(len(df)),
                "columns": df.columns.tolist()
            }
            
            # Mode/Status analysis if available
            if 'mode' in df.columns:
                mode_counts = df['mode'].value_counts()
                status_breakdown = []
                total_vehicles = len(df)
                for mode, count in mode_counts.head(5).items():
                    percentage = round((count / total_vehicles * 100), 1) if total_vehicles > 0 else 0
                    status_breakdown.append({
                        "status": str(mode),
                        "count": int(count),
                        "percentage": percentage
                    })
                response_data["summary"]["vehicle_status"] = status_breakdown
            
            # Date analysis if available  
            if 'date' in df.columns:
                date_counts = df['date'].value_counts()
                date_breakdown = []
                for date, count in date_counts.head(5).items():
                    date_breakdown.append({
                        "date": str(date),
                        "records": int(count)
                    })
                response_data["summary"]["date_distribution"] = date_breakdown
            
            # Speed analysis if available
            if 'speed' in df.columns:
                speed_data = df[df['speed'] > 0]['speed']
                if not speed_data.empty:
                    response_data["summary"]["speed_stats"] = {
                        "average": round(float(speed_data.mean()), 2),
                        "maximum": round(float(speed_data.max()), 2),
                        "minimum": round(float(speed_data.min()), 2),
                        "unit": "km/h"
                    }
            
            # Add helpful suggestions
            suggestions = [
                "Filter by status: 'show running vehicles'",
                "Search by location: 'vehicles at [location]'",
                "Get specific details: 'show vehicle [ID]'",
                "Analyze by time: 'vehicles yesterday'"
            ]
            response_data["suggestions"] = suggestions
            response_data["note"] = f"Dataset contains {total_rows:,} records. Use filters for detailed views."
            
            return json.dumps(response_data, default=self._json_serializer)
            
        except Exception as e:
            print(f"[DATAFRAME SUMMARY ERROR]: {e}")
            # Fallback summary with structured format
            return json.dumps({
                "status": "success",
                "query_type": "fleet_summary", 
                "total_records": int(total_rows),
                "summary": {
                    "message": f"Query returned {total_rows:,} records. Dataset analysis failed but data is available."
                },
                "suggestions": [
                    "Add specific filters to narrow results",
                    "Try: 'show running vehicles'",
                    "Try: 'vehicles at [location]'"
                ],
                "note": "Use more specific queries for detailed vehicle information."
            }, default=self._json_serializer)
        
    def _load_csv_data(self):
        """Loads the historical fleet dataset and geofence lookups into the unified memory db"""
        file_path = "db/merged_fleet_final_fuel_data(2).csv"
        geofence_path = "db/fleet_geofence_lookup.csv"
        table_name = "fleet_history"
        geo_table_name = "geofence_lookup"
        
        # 1. Load Telemetry Streams
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.replace(' ', '_').str.lower()
            df.to_sql(table_name, self.conn, if_exists="replace", index=False)
            print(f"Loaded {file_path} into single unified table '{table_name}'")
            
            cursor = self.conn.cursor()
            try:
                cursor.execute(f"SELECT MAX(date) FROM {table_name}")
                row = cursor.fetchone()
                if row and row[0]:
                    self.mock_today = str(row[0])
                else:
                    self.mock_today = "2026-06-19"
            except Exception as e:
                print(f"Error fetching max date: {e}")
                self.mock_today = "2026-06-19"
                
            print(f"[SYSTEM ARCHITECTURE] Virtual 'Today' initialized to: {self.mock_today}")
        else:
            print(f"CRITICAL ERROR: Data file not found at path {file_path}")

        # 2. Load Static Geofence Reference Table
        if os.path.exists(geofence_path):
            geo_df = pd.read_csv(geofence_path)
            geo_df.columns = geo_df.columns.str.replace(' ', '_').str.lower()
            
            # Static coordinate anchors mapping directly to geofence names
            coord_mapping = {
                'Golden Quadilateral (Warehouse Hub)': (26.4499, 80.3319),
                'Kanyakumari Road (Manufacturing Plant)': (14.2500, 77.8500),
                'Logistics Hub Base Alpha-12 (Client Dropoff Zone)': (28.5355, 77.3910),
                'Kundli - Manesar - Palwal Expressway (Corporate Office)': (28.3500, 77.0200),
                'Asian Highway 43 (Restricted Logistics Area)': (21.3000, 79.4000),
                '181/3 (Warehouse Hub)': (28.7761, 77.4725)
            }
            
            geo_df['lat'] = geo_df['geofence_name'].map(lambda x: coord_mapping.get(x, (0.0, 0.0))[0])
            geo_df['lng'] = geo_df['geofence_name'].map(lambda x: coord_mapping.get(x, (0.0, 0.0))[1])
            
            geo_df.to_sql(geo_table_name, self.conn, if_exists="replace", index=False)
            print(f"Loaded {geofence_path} into lookup table '{geo_table_name}' with coordinates.")
        else:
            print(f"WARNING: Geofence metadata file not found at path {geofence_path}")
            
    def _get_db_schema_string(self) -> str:
        """Generates a multi-table schema overview for the Text-to-SQL engine"""
        schema_info = """
### Available Database Tables & Structural Definitions:

1. Table Name: fleet_history
Description: Contains chronological point-in-time telemetry tracking messages streamed from active vehicles.
Columns:
  - vid (INTEGER): Unique vehicle asset identifier.
  - drivername (TEXT): Assigned driver name.
  - phonenumber (TEXT): Contact number.
  - lat (REAL): Current latitude coordinate value.
  - lng (REAL): Current longitude coordinate value.
  - latlong (TEXT): Lat and Lng packed together as string ("lat,lng").
  - addr (TEXT): Resolved street address position string.
  - gpstime (TEXT): Tracking interval log timestamp format ("DD-MM-YYYY HH:MM").
  - speed (REAL): Vehicle speed value.
  - mode (TEXT): Operational vehicle status state (Samples: 'RUNNING', 'STOPPED', 'IDLE', 'NOT WORKING').
  - date (TEXT): Standard date stamp index string format ('YYYY-MM-DD').
  - distance_traveled (REAL): Interval distance delta covered during this log point frame in kilometers.
  - odometer (REAL): Total cumulative machine lifetime counter mileage in kilometers.

2. Table Name: geofence_lookup
Description: Static reference map definitions detailing warehouse facilities, plants, and zone centers.
Columns:
  - geofence_id (INTEGER): Unique geofence layout key.
  - geofence_name (TEXT): Human descriptive title.
  - geo_fence (TEXT): Descriptive address layout string.
  - radius_meters (INTEGER): Allowed proximity validation fence threshold radius parameter.
  - zone_type (TEXT): Label categorization.
  - lat (REAL): Center coordinate latitude point of the fence area.
  - lng (REAL): Center coordinate longitude point of the fence area.
"""
        return schema_info


    def answer_user_query(self, user_question: str) -> str:
        if not self.api_url or not self.api_key or not self.model_name:
            return "Configuration error: Bedrock client is not set up. Please check your BEDROCK_API_URL, BEDROCK_API_KEY, and BEDROCK_MODEL environment variables."

        schema_context = self._get_db_schema_string()
        
        # 1. SQL Generation Prompt with Spatial Calculation Constraints
        system_prompt_sql = f"""
        You are an elite database engineer specializing in translating natural language into perfectly optimized SQLite queries.
        You have access to two relational tables: `fleet_history` and `geofence_lookup`.

        ### Database Schema Context:
        {schema_context}
        
        ### Global Environment Variables:
        - The current virtual 'Today's Date' in the database is strictly: '{self.mock_today}'

        ### CRITICAL OUTPUT REQUIREMENT:
        You MUST respond with ONLY the SQL query. Do NOT provide explanations, descriptions, or any other text.
        Do NOT use markdown code blocks or formatting. Return ONLY the raw SQL statement.
        
        Example of CORRECT response:
        SELECT vid, mode, speed FROM fleet_history WHERE date = '2026-06-20' ORDER BY gpstime DESC LIMIT 10;
        
        Example of INCORRECT response:
        Here's the SQL query to get the information:
        ```sql
        SELECT vid, mode, speed FROM fleet_history WHERE date = '2026-06-20' ORDER BY gpstime DESC LIMIT 10;
        ```

        ### Strict Core Instructions:
        1. Query Composition: Generate a valid SQLite query pulling from the table structures specified above. Use table joins where applicable.
        2. String Comparisons: Use the `LIKE` operator with case-insensitivity (`%target%`) when matching driver names, geofence titles, zone labels, or addresses.
        3. Handling Time/Current State: 
           - If a user asks for the "current", "now", "latest", or "today's" position of a vehicle, limit search strings to our virtual date `'{self.mock_today}'` OR use `ORDER BY date DESC, gpstime DESC LIMIT 1`.
           - If a user asks for a "trip", "history", "route", or "track logs", return the timeline ordered chronologically: `ORDER BY date ASC, gpstime ASC`.
        4. Date Format Filtering & Relative Time Translation: 
           - The `date` column uses 'YYYY-MM-DD' strings. Convert words like "today" directly into `'{self.mock_today}'` inside your queries. Avoid SQLite's native `DATE('now')`.
        5. Mileage & Distance Calculations Logic:
           - To find cumulative trip distance, execute `SUM(distance_traveled)`. 
           - To analyze total odometer change, execute `(MAX(odometer) - MIN(odometer))`. NEVER execute `SUM(odometer)`.
        6. Fleet-Wide Aggregation Guardrail:
           - If a user asks a broad summary question ("see running or stopped vehicles", "show overall distances"), NEVER run a generic `SELECT *`. Construct aggregate queries using `COUNT()`, `SUM()`, or `GROUP BY` to compress the rows returned.
        7. Handling Running/Stopped Time (Historical Summaries by Date):
           - If asked for "running time" or "stopped time" durations, translate the phrase into a daily status row count evaluation block: select `date`, `mode`, `COUNT(*) AS intervals_logged`, and `SUM(distance_traveled)`. Group them using `GROUP BY date, mode ORDER BY date ASC`.
        8. Geofence Proximity & Spatial Math Analytics (CRITICAL):
           - To find the exact distance in meters between a vehicle log position and a geofence center, use our custom SQL function: **`CALCULATE_DISTANCE(fh.lat, fh.lng, gl.lat, gl.lng)`**.
           - Proximity Rules:
             * Inside a geofence: `CALCULATE_DISTANCE(fh.lat, fh.lng, gl.lat, gl.lng) <= gl.radius_meters`
             * Outside a geofence: `CALCULATE_DISTANCE(fh.lat, fh.lng, gl.lat, gl.lng) > gl.radius_meters`
           - Fleet-Wide Outside Counting Logic:
             * If the user wants to count vehicles currently "outside all geofences" right now, filter for the virtual date `'{self.mock_today}'` and assert that no record exists in `geofence_lookup` where the distance is less than or equal to the radius limit.
             * Example query layout:
               SELECT COUNT(DISTINCT fh.vid) FROM fleet_history fh WHERE fh.date = '{self.mock_today}' AND NOT EXISTS (SELECT 1 FROM geofence_lookup gl WHERE CALCULATE_DISTANCE(fh.lat, fh.lng, gl.lat, gl.lng) <= gl.radius_meters);
        9. Handling Entry and Exit Timings (State Transitions):
            If a user explicitly asks for 'entry' or 'exit' timings into geofences, use a simplified approach compatible with SQLite:
            * AVOID complex LAG() functions in WHERE clauses or CASE statements
            * Use a straightforward chronological query to get vehicle positions near geofences
            * Let the system post-process the results to detect entry/exit events
            * Example simple approach:
              SELECT fh.vid, fh.date, fh.gpstime, gl.geofence_name, 
                     CALCULATE_DISTANCE(fh.lat, fh.lng, gl.lat, gl.lng) as distance,
                     CASE WHEN CALCULATE_DISTANCE(fh.lat, fh.lng, gl.lat, gl.lng) <= gl.radius_meters THEN 'INSIDE' ELSE 'OUTSIDE' END as status
              FROM fleet_history fh CROSS JOIN geofence_lookup gl 
              WHERE fh.vid = [vehicle_id] 
              ORDER BY fh.date ASC, fh.gpstime ASC;

        REMEMBER: Return ONLY the SQL query, nothing else.
        """
        
        try:
            # Generate the structured SQL instruction
            sql_generation_resp = self._call_bedrock_api(
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
            cleaned_sql = self._clean_sql_formatting(generated_sql)
            
            # Fix SQLite-specific window function issues
            fixed_sql = self._fix_sqlite_window_function_issues(cleaned_sql)
            
            # Validate the fixed SQL
            is_valid, validation_message = self._validate_sql_syntax(fixed_sql)
            
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
            processed_result = self._execute_with_smart_limits(fixed_sql, user_question)
            
            if processed_result.get("is_summary", False):
                # Return summary response directly
                return processed_result["response"]
            else:
                # Continue with normal LLM processing
                data_context = processed_result["data"]
            
            # 2. Polymorphic Response Aggregation Prompt
            system_prompt_synthesis = """
            You are a data reporting translation layer. Your single task is to convert raw database row matrices into a clean, structured JSON response based on the nature of the data retrieved.

            ### Dynamic Structuring Instructions:
            1. Analyze the user's question and the structural layout of the provided database rows.
            2. Determine the query context: Is it a single tracking snapshot, a sequential trip timeline, a daily aggregated history summary, or a geofence proximity evaluation?
            
            [CASE A: Single Snapshot / Latest State Context]
            - If the output contains only 1 row tracking an asset's position:
              * Set "query_topic" to "vehicle_location".
              * Place the coordinate point string inside "display_value".
              * Store parameters (vid, drivername, address, date, mode) inside the "metadata" root.

            [CASE B: Historical Journey / Trip Timeline Context]
            - If the output contains a timeline grid of multiple tracking trail records:
              * Set "query_topic" to "vehicle_journey".
              * Construct a path recap statement for "display_value".
              * Map point histories inside an array key named **"waypoints"** inside the "metadata" object.

            [CASE C: Daily Activity Summary Breakdown]
            - If the rows present structural breakdowns grouped by date and mode (using COUNT or SUM parameters):
              * Set "query_topic" to "vehicle_daily_summary".
              * Set a descriptive message for "display_value".
              * Output rows inside a list array named **"history_by_date"** inside the "metadata" object.

            [CASE D: Geofence Validation and Spatial Intelligence]
            "geofence_events": [
                {
                    "vid": 32037, 
                    "timestamp": "16-06-2026 14:22", 
                    "geofence_name": "Logistics Hub Base Alpha-12 (Client Dropoff Zone)",
                    "event_type": "ENTRY"
                },
                {
                    "vid": 32037, 
                    "timestamp": "16-06-2026 15:45", 
                    "geofence_name": "Logistics Hub Base Alpha-12 (Client Dropoff Zone)",
                    "event_type": "EXIT"
                }
            ]

            ### Expected Output Format Templates:
            
            For Case D (Geofence Reports):
            {
                "query_topic": "geofence_analytics",
                "display_value": "Vehicle 28451 crossed into Logistics Hub Base Alpha-12 perimeter on 2026-06-13.",
                "metadata": {
                    "geofence_events": [
                        {"vid": 28451, "timestamp": "13-06-2026 12:59", "calculated_distance_meters": 45.8, "status": "INSIDE", "geofence_name": "Logistics Hub Base Alpha-12 (Client Dropoff Zone)"},
                        {"vid": 28451, "timestamp": "13-06-2026 13:25", "calculated_distance_meters": 185.2, "status": "OUTSIDE", "geofence_name": "Logistics Hub Base Alpha-12 (Client Dropoff Zone)"}
                    ]
                }
            }
            """
            
            final_resp = self._call_bedrock_api(
                messages=[
                    {"role": "system", "content": system_prompt_synthesis},
                    {"role": "user", "content": f"User Question: {user_question}\nDatabase Output Matrix:\n{data_context}"}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            response_content = final_resp.choices[0].message.content.strip()
            
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