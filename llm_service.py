import os
import sqlite3
import pandas as pd
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the request body structure using Pydantic
class QueryRequest(BaseModel):
    prompt: str
    temperature: float = 0.7

class ChatbotService:
    def __init__(self):
        # 1. Initialize Groq Client
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.model_name = os.getenv("GROQ_MODEL")
        
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
        if not self.client or not self.model_name:
            return "Configuration error: Groq client is not set up. Please check your GROQ_API_KEY and GROQ_MODEL environment variables."

        schema_context = self._get_db_schema_string()
        
        # 1. SQL Generation Prompt with Spatial Calculation Constraints
        system_prompt_sql = f"""
        You are an elite database engineer specializing in translating natural language into perfectly optimized SQLite queries.
        You have access to two relational tables: `fleet_history` and `geofence_lookup`.

        ### Database Schema Context:
        {schema_context}
        
        ### Global Environment Variables:
        - The current virtual 'Today's Date' in the database is strictly: '{self.mock_today}'

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
            If a user explicitly asks for 'entry' or 'exit' timings into geofences, do not return all rows where the vehicle is inside. Instead, use a window function (LAG) over the chronological tracking history (ORDER BY date ASC, gpstime ASC) to detect when a vehicle crosses the boundary.

            An Entry occurs when the previous distance was > radius_meters (or NULL) and the current distance is <= radius_meters.

            An Exit occurs when the previous distance was <= radius_meters and the current distance is > radius_meters
        """
        
        try:
            # Generate the structured SQL instruction
            sql_generation_resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt_sql},
                    {"role": "user", "content": f"User Request: {user_question}\nSQL Query:"}
                ],
                temperature=0.0
            )
            
            generated_sql = sql_generation_resp.choices[0].message.content.strip()
            
            if generated_sql.startswith("```"):
                generated_sql = generated_sql.replace("```sql", "").replace("```", "").strip()
                
            print(f"\n[TEXT-TO-SQL LOG] Generated Query:\n{generated_sql}\n")
            
            try:
                df_result = pd.read_sql_query(generated_sql, self.conn)
                data_context = df_result.to_dict(orient="records")
            except Exception as sql_err:
                print(f"[SQL EXECUTION ERROR]: {sql_err}")
                return '{"status": "error", "message": "No records found"}'
            
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
            
            final_resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt_synthesis},
                    {"role": "user", "content": f"User Question: {user_question}\nDatabase Output Matrix:\n{data_context}"}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return final_resp.choices[0].message.content.strip()
            
        except Exception as e:
            return f'{{"status": "error", "message": "Encountered processing failure: {str(e)}"}}'