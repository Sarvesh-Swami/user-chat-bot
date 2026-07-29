import json
import re
from typing import Dict, Any, Optional

class IntentParser:
    """Parses user intent and validates query safety/supportability before SQL generation"""

    def __init__(self, llm_client, schema_cache: Dict[str, Any]):
        self.llm_client = llm_client
        self.schema_cache = schema_cache

    def parse(self, raw_user_prompt: str, chat_history_str: str) -> Dict[str, Any]:
        """
        Analyze the raw user prompt and conversation history to determine if the query is supported.
        Returns a dict with keys: 'is_supported' (bool), 'denial_reason' (str or None), and 'intent' (str).
        """
        # Hard check for dangerous database modifications before calling the LLM
        dangerous_keywords = [
            r"\binsert\b", r"\bupdate\b", r"\bdelete\b", r"\bdrop\b", r"\balter\b", 
            r"\btruncate\b", r"\bcreate\b", r"\bgrant\b", r"\brevoke\b"
        ]
        prompt_lower = raw_user_prompt.lower()
        if any(re.search(pattern, prompt_lower) for pattern in dangerous_keywords):
            # Check if it looks like a database modification intent rather than data request
            # (e.g. "update vehicle status", "delete trip", "create table")
            modification_intents = ["delete", "remove", "update", "modify", "change", "insert", "add", "create", "drop"]
            if any(word in prompt_lower for word in modification_intents):
                return {
                    "is_supported": False,
                    "denial_reason": "I am a read-only assistant. I cannot modify, create, update, or delete any vehicle or fleet database records.",
                    "intent": "unsupported"
                }

        # Pre-check: Allow common affirmative words/typos and short responses without denying
        prompt_clean = raw_user_prompt.strip().lower()
        common_affirmatives = {
            "yes", "yres", "yea", "yeah", "yup", "sure", "ok", "okay", 
            "go ahead", "do it", "please", "yes please", "hi", "hello", 
            "hey", "thanks", "thank you", "no", "nope"
        }
        if prompt_clean in common_affirmatives or re.match(r"^\s*(y+e+s+|y+r+e+s+|y+e+a+h*|y+u+p+|s+u+r+e+|o+k+a*y*)\s*$", prompt_clean):
            print(f"[INTENT PARSER] Pre-check matched affirmative/conversational prompt: '{raw_user_prompt}'")
            return {
                "is_supported": True,
                "denial_reason": None,
                "intent": "conversational",
                "response_text": "How can I assist you with your fleet today?",
                "query_category": "default",
                "confidence": 1.0
            }

        # Build schema summary for the intent parser context
        schema_summary = self._get_capabilities_context()

        system_prompt = f"""You are an Intent Parser and Gatekeeper for a Fleet Management Chatbot.
Your single job is to analyze the user's current query (with context from conversation history) and decide if it is SUPPORTED or UNSUPPORTED by our system.

### SUPPORTED CAPABILITIES:
1. **Vehicles**: Querying vehicle details (ID, license plate, make, model, year, status, driver, device ID, safety score, total distance, speeding counts).
2. **Real-time Tracking**: Querying current vehicle coordinates, speed, heading, timestamp, or speed limit.
3. **Trips**: Querying trip status (Started, Completed), start/end time, start/end coordinates/addresses, distance, duration, trip score.
4. **Drivers / Users**: Querying driver name, email, phone, status, license ID, license type, license expiry.
5. **Safety Events**: Querying speeding events, harsh braking events, event locations, event speeds, timestamps.
6. **Custom Actions**: 
   - Exporting queried data as a PDF report.
   - Emailing reports to specific addresses.
   - Generating charts, graphs, or plots of queried data.
7. **Conversational / Greetings / Help**: Conversational inputs such as greetings (e.g. 'hi', 'hello', 'hey'), politeness (e.g. 'how are you?', 'thank you'), or general inquiries about system capabilities (e.g. 'what can you do?', 'help', 'what tables do you have?').

### UNSUPPORTED CAPABILITIES (Must be Denied):
1. **Database Modification**: Creating, inserting, updating, or deleting records (e.g., adding a vehicle, changing a driver's name, deleting an event log).
2. **Data We Do Not Store**: Any queries regarding vehicle fuel economy/consumption, vehicle maintenance logs/repairs, vehicle insurance details, financial costs/budgets, or driver salaries.
3. **General / Out-of-Scope Q&A**: Unrelated questions (e.g., weather forecasts, coding, math, general web search, writing essays).

### SCHEMA CONTEXT:
{schema_summary}

### CONVERSATION HISTORY:
{chat_history_str}

### OUTPUT FORMAT:
You MUST respond with a single JSON object. Do not include markdown formatting or extra text.
JSON Schema:
{{
  "is_supported": boolean,
  "denial_reason": string or null,
  "intent": "text_to_sql" | "pdf_report" | "email_report" | "chart_generation" | "conversational" | "unsupported",
  "response_text": string or null,
  "query_category": "vehicle_location" | "vehicle_livetrack" | "trip_current" | "trip_origin" | "trip_destination" | "trip_last" | "trip_history" | "safety_events" | "driver_info" | "fleet_overview" | "vehicle_stats" | "default",
  "confidence": float between 0.0 and 1.0
}}

Rules for query_category and confidence:
- Only populate query_category when intent is 'text_to_sql'. Use 'default' for all other intents.
- vehicle_location: user asks for current/last known position of a specific vehicle.
- vehicle_livetrack: user asks for GPS history, tracking trail, speed history, or currently moving vehicles fleet-wide.
- trip_current: user asks for details about a vehicle's active (in-progress) trip.
- trip_origin: user asks where a vehicle came from / started.
- trip_destination: user asks where a vehicle is going / its destination.
- trip_last: user asks for the most recent completed trip.
- trip_history: user asks for multiple trips, trip history, or trip statistics for a SPECIFIC vehicle.
- safety_events: user asks about speeding, harsh braking, incidents, or events.
- driver_info: user asks about drivers, users, licenses, or driver assignments.
- fleet_overview: user asks broad fleet-wide questions (total vehicles, utilization, fleet summary, vehicles by manufacturer, fleet safety score).
- vehicle_stats: user asks for rankings, comparisons, or aggregates ACROSS ALL vehicles (e.g., 'which vehicle has the highest/most/best/worst X', 'top 5 vehicles by distance', 'vehicle with most trips', 'highest mileage vehicle', 'lowest safety score vehicle'). ALWAYS use vehicle_stats — NOT vehicle_location — for superlative/ranking queries.
- default: query doesn't clearly fit the above categories.
- confidence: your certainty about the query_category (1.0 = very certain, 0.0 = unsure)."""

        user_prompt = f"Current User Query: {raw_user_prompt}\nJSON Output:"

        try:
            response = self.llm_client._call_bedrock_api(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            raw_content = response.choices[0].message.content.strip()
            try:
                print(f"[INTENT PARSER] Raw response: {raw_content}")
            except Exception:
                try:
                    print(f"[INTENT PARSER] Raw response (sanitized): {raw_content.encode('ascii', 'ignore').decode('ascii')}")
                except Exception:
                    pass
            
            # Clean JSON response from markdown markers if present
            cleaned_json = raw_content
            if cleaned_json.startswith("```"):
                cleaned_json = re.sub(r"^```(?:json)?\n?", "", cleaned_json)
                cleaned_json = re.sub(r"\n?```$", "", cleaned_json)
                cleaned_json = cleaned_json.strip()
            
            parsed_result = json.loads(cleaned_json)
            
            # Validate output keys and apply defaults for new fields
            if "is_supported" not in parsed_result:
                parsed_result["is_supported"] = True
            if "intent" not in parsed_result:
                parsed_result["intent"] = "text_to_sql"
            if "denial_reason" not in parsed_result:
                parsed_result["denial_reason"] = None
            if "response_text" not in parsed_result:
                parsed_result["response_text"] = None
            if "query_category" not in parsed_result or not parsed_result["query_category"]:
                parsed_result["query_category"] = "default"
            # Clamp confidence to [0.0, 1.0]
            try:
                parsed_result["confidence"] = max(0.0, min(1.0, float(parsed_result.get("confidence", 1.0))))
            except (TypeError, ValueError):
                parsed_result["confidence"] = 1.0
                
            return parsed_result
            
        except Exception as e:
            print(f"[INTENT PARSER ERROR] Failed to parse intent: {e}. Falling back to default execution.")
            return {
                "is_supported": True,
                "denial_reason": None,
                "intent": "text_to_sql",
                "response_text": None,
                "query_category": "default",
                "confidence": 0.0  # Force full-hint fallback on parser error
            }

    def _get_capabilities_context(self) -> str:
        """Helper to build a clean string of the available tables and columns in schema cache"""
        if not self.schema_cache or 'tables' not in self.schema_cache:
            return "No schema metadata cached."
            
        lines = []
        for table_name, table_info in self.schema_cache['tables'].items():
            col_names = [col['name'] for col in table_info.get('columns', [])]
            lines.append(f"- Table '{table_name}': columns [{', '.join(col_names)}]")
        return "\n".join(lines)
