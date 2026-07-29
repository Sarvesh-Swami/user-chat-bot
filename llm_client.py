import os
import requests
import json
import re
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMClient:
    def __init__(self, db_manager, schema_cache):
        # Initialize Bedrock Client configuration
        self.api_url = os.getenv("BEDROCK_API_URL")
        self.api_key = os.getenv("BEDROCK_API_KEY")
        self.model_name = os.getenv("BEDROCK_MODEL")
        
        # Setup headers for Bedrock API
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Store dependencies
        self.db_manager = db_manager
        self.schema_cache = schema_cache


    def condense_history_query(self, condensation_prompt: str) -> str:
        """Lightweight LLM call to condense query with conversation history"""
        try:
            response = self._call_bedrock_api(
                messages=[
                    {"role": "user", "content": condensation_prompt}
                ],
                temperature=0.0  # Use deterministic temperature for consistency
            )
            
            condensed_query = response.choices[0].message.content.strip()
            return condensed_query
            
        except Exception as e:
            print(f"[QUERY CONDENSATION ERROR]: {e}")
            # Fallback: return original query from the prompt
            lines = condensation_prompt.split('\n')
            for line in lines:
                if line.startswith('Current user question:'):
                    return line.replace('Current user question:', '').strip()
            return "Could not process query"


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
- Superlatives, rankings, or limits (highest distance, lowest battery, top 5, fastest, most trips)
- Specific drivers by name
- Vehicles at specific locations or geofences
- Historical tracking of specific assets
- Detailed information about identified entities
Examples: "show vehicle ABC123", "which vehicle has highest travel distance", "find driver John Smith", "top 5 vehicles"
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
        sql_lower = sql_query.lower()
        
        # Specific indicators - these suggest detailed queries about particular entities
        specific_indicators = [
            # Vehicle-specific patterns
            r'\bvehicle\s+[a-zA-Z0-9]+\b',          # "vehicle ABC123"  
            r'\bvid\s*=\s*\d+',                      # "vid = 123"
            r'\b[a-zA-Z]{2,4}\d{2,6}\b',             # License plate patterns like "ABC123", "HQNS82400038"
            
            # Driver-specific patterns
            r'\bdriver\s+[a-zA-Z]+',                 # "driver John"
            r'\bname\s*=\s*[\'"][a-zA-Z\s]+[\'"]',   # name = "John Smith"
            
            # Location-specific patterns  
            r'\bat\s+[a-zA-Z\s]+\b',                 # "at Delhi", "at warehouse"
            r'\bin\s+[a-zA-Z\s]+\b',                 # "in Mumbai", "in depot"
            r'\bnear\s+[a-zA-Z\s]+\b',               # "near airport"
            
            # ID-based patterns
            r'\bid\s*=\s*[\'"]*[a-zA-Z0-9]+[\'"]*', # "id = 'ABC123'"
            r'\bwhere\s+\w+\s*=\s*[\'"]*[a-zA-Z0-9]+[\'"]*', # "where vehicle_id = 'ABC'"
            
            # Time-specific (usually for historical data of specific entities)
            r'\bon\s+\d{4}-\d{2}-\d{2}\b',          # "on 2024-01-01"  
            r'\bbetween\s+\d{4}-\d{2}-\d{2}',       # "between 2024-01-01"
        ]
        
        # Check if any specific indicators are present
        for pattern in specific_indicators:
            if re.search(pattern, user_lower) or re.search(pattern, sql_lower):
                return "specific"
        
        # General indicators - these suggest broad overview queries
        general_indicators = [
            # Broad fleet queries
            r'\ball\s+vehicles?\b',                   # "all vehicles"
            r'\bshow\s+vehicles?\b',                  # "show vehicles"  
            r'\blist\s+vehicles?\b',                  # "list vehicles"
            r'\btotal\s+vehicles?\b',                 # "total vehicles"
            r'\bhow\s+many\s+vehicles?\b',            # "how many vehicles"
            
            # Status queries without specific entities
            r'\brunning\s+vehicles?\b',               # "running vehicles"
            r'\bidle\s+vehicles?\b',                  # "idle vehicles"
            r'\bstopped\s+vehicles?\b',               # "stopped vehicles"
            r'\bactive\s+vehicles?\b',                # "active vehicles"
            
            # General time-based queries
            r'\btoday\b|\byesterday\b|\bthis\s+week\b|\bthis\s+month\b',
            
            # Overview/summary terms
            r'\boverview\b|\bsummary\b|\bstatus\b|\breport\b',
        ]
        
        # Check if any general indicators are present
        for pattern in general_indicators:
            if re.search(pattern, user_lower):
                return "general"
        
        # Default classification logic
        # If SQL has specific WHERE clauses with exact matches, likely specific
        if re.search(r"where\s+\w+\s*=\s*['\"][^'\"]+['\"]", sql_lower):
            return "specific"
        
        # If SQL has broad SELECT without specific WHERE, likely general
        if "select" in sql_lower and "where" not in sql_lower:
            return "general"
        
        return "general"  # Default to general for better user experience

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

    def condense_history_query(self, condensation_prompt: str) -> str:
        """Lightweight LLM call to condense query with conversation history"""
        try:
            response = self._call_bedrock_api(
                messages=[
                    {"role": "user", "content": condensation_prompt}
                ],
                temperature=0.0  # Use deterministic temperature for consistency
            )
            
            condensed_query = response.choices[0].message.content.strip()
            return condensed_query
            
        except Exception as e:
            print(f"[QUERY CONDENSATION ERROR]: {e}")
            # Fallback: return original query from the prompt
            lines = condensation_prompt.split('\n')
            for line in lines:
                if line.startswith('Current user question:'):
                    return line.replace('Current user question:', '').strip()
            return "Could not process query"

    def _generate_summary_response(self, original_sql, user_question, total_rows):
        """Generate intelligent summary for large result sets"""
        
        try:
            # Analyze the original query to understand intent
            sql_lower = original_sql.lower()
            
            # Generate appropriate summary queries based on table and intent
            summary_queries = []
            
            # Remove semicolon from original SQL for use in subqueries
            clean_original_sql = original_sql.rstrip(';')
            
            # Detect which table is being queried to use correct column names
            has_trips_table = "trips" in sql_lower or "trip_" in sql_lower
            has_vehicles_table = "vehicles" in sql_lower or "vehicle_" in sql_lower
            has_livetrack_table = "livetrack" in sql_lower
            
            # Status/Mode breakdown - use appropriate column names
            if has_trips_table:
                # For trips table, use trip_status
                summary_queries.extend([
                    f"SELECT COALESCE(trip_status, 'Unknown') as status, COUNT(*) as count FROM ({clean_original_sql}) AS subquery GROUP BY COALESCE(trip_status, 'Unknown') ORDER BY count DESC",
                    f"SELECT DATE(start_date) as date, COUNT(*) as records FROM ({clean_original_sql}) AS subquery WHERE start_date IS NOT NULL GROUP BY DATE(start_date) ORDER BY date DESC LIMIT 7"
                ])
                
                # Add location analysis for trips
                summary_queries.append(
                    f"SELECT SUBSTRING(COALESCE(start_address, end_address, 'Unknown') FROM 1 FOR 50) as location, COUNT(*) as count FROM ({clean_original_sql}) AS subquery GROUP BY SUBSTRING(COALESCE(start_address, end_address, 'Unknown') FROM 1 FOR 50) ORDER BY count DESC LIMIT 10"
                )
                
                # Add trip score analysis if available
                if "trip_score" in sql_lower:
                    summary_queries.append(
                        f"SELECT ROUND(AVG(trip_score)::numeric, 2) as avg_score, MAX(trip_score) as max_score, MIN(trip_score) as min_score, COUNT(*) as records FROM ({clean_original_sql}) AS subquery WHERE trip_score IS NOT NULL"
                    )
            
            elif has_livetrack_table:
                # For livetrack table, use speed and coordinates
                summary_queries.extend([
                    f"SELECT CASE WHEN speed > 50 THEN 'High Speed' WHEN speed > 20 THEN 'Medium Speed' WHEN speed > 0 THEN 'Low Speed' ELSE 'Stationary' END as status, COUNT(*) as count FROM ({clean_original_sql}) AS subquery GROUP BY CASE WHEN speed > 50 THEN 'High Speed' WHEN speed > 20 THEN 'Medium Speed' WHEN speed > 0 THEN 'Low Speed' ELSE 'Stationary' END ORDER BY count DESC"
                ])
                
                # Add speed analysis
                if "speed" in sql_lower:
                    summary_queries.append(
                        f"SELECT ROUND(AVG(speed)::numeric, 2) as avg_speed, MAX(speed) as max_speed, MIN(speed) as min_speed, COUNT(*) as records FROM ({clean_original_sql}) AS subquery WHERE speed >= 0"
                    )
            
            elif has_vehicles_table:
                # For vehicles table, use vehicle-specific fields
                summary_queries.extend([
                    f"SELECT COALESCE(vehicle_id, 'Unknown') as vehicle, COUNT(*) as count FROM ({clean_original_sql}) AS subquery GROUP BY COALESCE(vehicle_id, 'Unknown') ORDER BY count DESC LIMIT 10"
                ])
            
            # Execute summary queries
            summary_data = {}
            for i, query in enumerate(summary_queries):
                try:
                    results = self.db_manager.execute_query(query)
                    
                    # Convert results to JSON serializable format
                    summary_dict = []
                    for row in results:
                        record = dict(row)
                        for key, value in record.items():
                            if hasattr(value, 'item'):  # Handle numpy/decimal types
                                record[key] = value.item()
                            elif value is None:
                                record[key] = None
                            else:
                                # Convert decimals and other types to appropriate Python types
                                if isinstance(value, (int, float)):
                                    record[key] = value
                                else:
                                    record[key] = str(value)
                        summary_dict.append(record)
                    
                    summary_data[f"summary_{i+1}"] = summary_dict
                except Exception as e:
                    print(f"[SUMMARY QUERY ERROR]: Query failed - {e}")
                    print(f"[SUMMARY QUERY ERROR]: Failed query - {query}")
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
            
            return json.dumps(response_data, default=self.db_manager._json_serializer)
            
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
            }, default=self.db_manager._json_serializer)


    def _get_db_schema_string(self) -> str:
        """Generate dynamic schema documentation from live PostgreSQL database"""
        
        if not self.schema_cache.get('tables'):
            return "### Database Schema: Not Available\nPlease check database connection."
        
        schema_parts = ["### Available PostgreSQL Database Tables & Schema:\n"]
        
        for table_name, table_info in self.schema_cache['tables'].items():
            schema_parts.append(f"\n{len(schema_parts)}. Table Name: {table_name}")
            schema_parts.append(f"Description: {table_info['description']}")
            schema_parts.append("Columns:")
            
            for column in table_info['columns']:
                nullable_text = "nullable" if column['nullable'] else "not null"
                default_text = f", default: {column['default']}" if column['default'] else ""
                
                schema_parts.append(f"  - {column['name']} ({column['type'].upper()}, {nullable_text}{default_text}): {column['description']}")
        
        # Add relationships section
        if self.schema_cache.get('relationships'):
            schema_parts.append("\n### Table Relationships (Foreign Keys):")
            for rel in self.schema_cache['relationships']:
                schema_parts.append(f"- {rel['table']}.{rel['column']} → {rel['references_table']}.{rel['references_column']}")
        
        # Add spatial function information  
        schema_parts.append("\n### Available Spatial Functions:")
        schema_parts.append("- ST_Distance(point1, point2): Calculate distance between two geographic points")
        schema_parts.append("- ST_DWithin(geometry, geometry, distance): Check if geometries are within specified distance")
        schema_parts.append("- For lat/lng calculations, use: ST_Distance(ST_Point(lng1, lat1), ST_Point(lng2, lat2))")
        
        # Add known enum/categorical column values
        enum_values = self.schema_cache.get('enum_values', {})
        if enum_values:
            schema_parts.append("\n### KNOWN COLUMN VALUES (use these EXACT values in WHERE clauses — do NOT invent your own):")
            for col_key, values in enum_values.items():
                values_str = ", ".join(f"'{v}'" for v in values)
                schema_parts.append(f"  - {col_key}: [{values_str}]")
            schema_parts.append("IMPORTANT: When filtering on any of the columns above, you MUST use one of the listed values exactly as shown. Do NOT guess or translate user words into snake_case or other formats.")
        
        return "\n".join(schema_parts)
