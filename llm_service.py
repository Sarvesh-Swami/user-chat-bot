import os
import sqlite3
import pandas as pd
from pydantic import BaseModel
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the request body structure using Pydantic
class QueryRequest(BaseModel):
    prompt: str
    temperature: float = 0.7

class ChatbotService:
    def __init__(self):
        # 1. Initialize Azure OpenAI Client internally via instance state
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        # 2. Setup In-Memory SQLite Database
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._load_csv_data()
        
    def _load_csv_data(self):
        """Loads the single merged historical fleet dataset into a unified table"""
        file_path = "db/merged_fleet_final_fuel_data.csv"
        table_name = "fleet_history"
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Clean column names (remove spaces/special chars) to make SQL execution safer
            df.columns = df.columns.str.replace(' ', '_').str.lower()
            df.to_sql(table_name, self.conn, if_exists="replace", index=False)
            print(f"Loaded {file_path} into single unified table '{table_name}'")
            
            # Print schema to console for verification
            cursor = self.conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            print(f"Schema for {table_name}: {[col[1] for col in cursor.fetchall()]}")
        else:
            print(f"CRITICAL ERROR: Data file not found at path {file_path}")

    def _get_db_schema_string(self) -> str:
        """
        Generates a clear schema string for the history table,
        injecting value samples for categorical and chronological fields.
        """
        schema_info = ""
        cursor = self.conn.cursor()
        table = "fleet_history"
        
        try:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            
            column_specs = []
            for col in columns:
                col_name = col[1]
                col_type = col[2]
                
                sample_str = ""
                # Profile tracking statuses, metrics, and new contextual categories
                if col_name in ["ignstate", "acstate", "mode", "gpsstatus", "panic", "vendor_source", "date"]:
                    try:
                        cursor.execute(f"SELECT DISTINCT {col_name} FROM {table} WHERE {col_name} IS NOT NULL ORDER BY {col_name} LIMIT 4")
                        samples = [f"'{str(row[0])}'" if isinstance(row[0], str) else str(row[0]) for row in cursor.fetchall()]
                        if samples:
                            sample_str = f" (Allowed values/samples: {', '.join(samples)})"
                    except Exception:
                        pass
                elif col_name in ["drivername", "phonenumber", "gpstime", "addr"]:
                    try:
                        cursor.execute(f"SELECT {col_name} FROM {table} WHERE {col_name} IS NOT NULL AND {col_name} != 'NA' AND {col_name} != '' LIMIT 1")
                        row = cursor.fetchone()
                        if row:
                            sample_str = f" (e.g., '{row[0]}')"
                    except Exception:
                        pass

                column_specs.append(f"  - {col_name} ({col_type}){sample_str}")
            
            schema_info += f"Table Name: {table}\nColumns:\n" + "\n".join(column_specs) + "\n\n"
            
        except Exception as schema_err:
            print(f"Error fetching dynamic schema details: {schema_err}")
            
        return schema_info

    def answer_user_query(self, user_question: str) -> str:
        if not self.client or not self.deployment_name:
            return "Configuration error: Azure OpenAI client is not set up."

        schema_context = self._get_db_schema_string()
        
        # 1. System context optimized for single table chronological analytical evaluation
        system_prompt_sql = f"""
        You are an elite database engineer specializing in translating natural language into perfectly optimized SQLite queries.
        Your target table holds multi-date, historical fleet telemetry data.

        ### Database Schema Context:
        {schema_context}

        ### Strict Core Instructions:
        1. Query Composition: Generate a valid SQLite query based ONLY on the single table provided above. Do not attempt UNION operations across non-existent tables.
        2. String Comparisons: Use the `LIKE` operator with case-insensitivity if checking for specific names, addresses, or metadata.
        3. Handling Time/Current State: 
           - Because the table keeps records over multiple dates, if a user asks for the "current", "now", "latest", or "today's" status of a vehicle or driver, you MUST sort the matches using `ORDER BY date DESC, gpstime DESC LIMIT 1`.
           - To get historical trends or analytical summary aggregations across the entire timeframe, omit the `LIMIT 1` construct and process group actions where applicable.
        4. Date Format Filtering: The `date` column is formatted as standard ISO strings ('YYYY-MM-DD'). When executing comparisons based on dates, construct conditions using simple string syntax matching that exact format (e.g., `WHERE date BETWEEN '2026-06-12' AND '2026-06-16'`).
        5. Formatting: Output the RAW SQL query string only. 
           - DO NOT wrap the code block in ```sql ... ``` markdown syntax.
           - DO NOT include explanations, introduction, or trailing comments.
           - Start your response directly with the keyword SELECT.
        """
        
        try:
            # Generate the structured SQL instruction
            sql_generation_resp = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt_sql},
                    {"role": "user", "content": f"User Request: {user_question}\nSQL Query:"}
                ],
                temperature=0.0
            )
            
            generated_sql = sql_generation_resp.choices[0].message.content.strip()
            
            # Clean accidental markdown encapsulation safely
            if generated_sql.startswith("```"):
                generated_sql = generated_sql.replace("```sql", "").replace("```", "").strip()
                
            print(f"\n[TEXT-TO-SQL LOG] Generated Query:\n{generated_sql}\n")
            
            # 2. Execute directly against our memory-mapped engine with complete safety fallbacks
            try:
                df_result = pd.read_sql_query(generated_sql, self.conn)
                data_context = df_result.to_dict(orient="records")
            except Exception as sql_err:
                print(f"[SQL EXECUTION ERROR]: {sql_err}")
                return '{"status": "error", "message": "No records found"}'
            
            # 3. Structural JSON synthesis framework
            system_prompt_synthesis = """
            You are a data reporting translation layer. Your single task is to convert raw database row matrices into a clean, structured JSON response.

            ### Instructions:
            1. Analyze the user's question and the provided database rows.
            2. Extract the core metric or answer and place it in the "display_value" field.
            3. Put all relevant supporting telemetry columns (like lat, lng, addr, drivername, vendor_source, or date) inside the "metadata" object.
            4. Respond ONLY with valid, raw JSON. Do not include markdown backticks like ```json.

            ### Expected Output Format:
            {
                "query_topic": "vehicle_location",
                "display_value": "28.589508, 77.245247",
                "metadata": {
                    "vid": 165533,
                    "drivername": "SANJEEV KUMAR",
                    "address": "Gurudwara Rd, Jiwan Nagar",
                    "vendor_source": "neelkanthtravels",
                    "date": "2026-06-12"
                }
            }
            """
            
            # 4. Formulate the clean structured JSON response
            final_resp = self.client.chat.completions.create(
                model=self.deployment_name,
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