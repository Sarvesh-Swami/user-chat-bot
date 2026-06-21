import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AzureOpenAI
from dotenv import load_dotenv
import sqlite3
import pandas as pd
# Load environment variables from .env file
load_dotenv()


# Initialize the Azure OpenAI Client
# It automatically picks up the API key and endpoint if configured correctly,
# but passing them explicitly guarantees clarity.
try:
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
except Exception as e:
    print(f"Error initializing Azure OpenAI Client: {e}")
    client = None

# Define the request body structure using Pydantic
class QueryRequest(BaseModel):
    prompt: str
    temperature: float = 0.7

# --- API Endpoints ---

class ChatbotService:
    def __init__(self):
        # 1. Initialize Azure OpenAI Client
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
        """Loads all CSV files into SQLite tables matching their filenames"""
        csv_files = {
            "amanbus": "db/amanbus_17062026124338.csv",
            "neelkanthtravels": "db/neelkanthtravels_17062026124343.csv",
            "shantitpt": "db/shantitpt_17062026124343.csv"
        }
        
        for table_name, file_path in csv_files.items():
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # Clean column names (remove spaces/special chars) to make SQL execution safer
                df.columns = df.columns.str.replace(' ', '_').str.lower()
                df.to_sql(table_name, self.conn, if_exists="replace", index=False)
                print(f"Loaded {file_path} into table '{table_name}'")
                
                # Print schema to console for your verification
                cursor = self.conn.cursor()
                cursor.execute(f"PRAGMA table_info({table_name})")
                print(f"Schema for {table_name}: {[col[1] for col in cursor.fetchall()]}")

    def _get_db_schema_string(self) -> str:
        """
        Generates a clean schema string, filtering out unnecessary columns
        and injecting distinctive distinct samples for categorical matching.
        """
        schema_info = ""
        cursor = self.conn.cursor()
        tables = ["amanbus", "neelkanthtravels", "shantitpt"]
        
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            
            column_specs = []
            for col in columns:
                col_name = col[1]
                col_type = col[2]
                
                # 1. Skip the noisy ELOCKInfo metadata columns to save tokens
                if col_name.startswith("elockinfo_"):
                    continue
                
                # 2. Fetch distinct sample values for categorical columns to prevent hallucinations
                sample_str = ""
                if col_name in ["ignstate", "acstate", "mode", "gpsstatus", "panic", "elock"]:
                    try:
                        cursor.execute(f"SELECT DISTINCT {col_name} FROM {table} WHERE {col_name} IS NOT NULL LIMIT 4")
                        samples = [f"'{str(row[0])}'" if isinstance(row[0], str) else str(row[0]) for row in cursor.fetchall()]
                        if samples:
                            sample_str = f" (Allowed values: {', '.join(samples)})"
                    except Exception:
                        pass
                elif col_name in ["drivername", "phonenumber", "gpstime"]:
                    # Provide a quick formatting/string template example for text matching
                    try:
                        cursor.execute(f"SELECT {col_name} FROM {table} WHERE {col_name} IS NOT NULL LIMIT 1")
                        row = cursor.fetchone()
                        if row:
                            sample_str = f" (e.g., '{row[0]}')"
                    except Exception:
                        pass

                column_specs.append(f"  - {col_name} ({col_type}){sample_str}")
            
            schema_info += f"Table Name: {table}\nColumns:\n" + "\n".join(column_specs) + "\n\n"
            
        return schema_info

    def answer_user_query(self, user_question: str) -> str:
        if not self.client or not self.deployment_name:
            return "Configuration error: Azure OpenAI client is not set up."

        schema_context = self._get_db_schema_string()
        
        # 1. System context to generate the clean SQLite query
        system_prompt_sql = f"""
        You are an elite database engineer specializing in translating natural language into perfectly optimized SQL queries.
        Your dialect target is: SQLite.

        ### Database Schema Context:
        {schema_context}

        ### Strict Instructions:
        1. Query Composition: Generate a valid SQLite query based ONLY on the tables and columns provided above.
        2. String Comparisons: Use the `LIKE` operator with case-insensitivity or string matching if the user query contains names or locations that might have mixed casing.
        3. Multi-table handling: If the user asks for a total or comparison across all files, utilize `UNION ALL` or `JOIN` where appropriate.
        4. Formatting: Output the RAW SQL query string only. 
           - DO NOT wrap the code block in ```sql ... ``` markdown syntax.
           - DO NOT include explanations or trailing text.
           - Start your response directly with SELECT.
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
            
            # Clean accidental markdown if the model leaks backticks anyway
            if generated_sql.startswith("```"):
                generated_sql = generated_sql.replace("```sql", "").replace("```", "").strip()
                
            print(f"\n[TEXT-TO-SQL LOG] Generated Query:\n{generated_sql}\n")
            
            # 2. Execute directly against our structural CSV engine with error safety
            try:
                df_result = pd.read_sql_query(generated_sql, self.conn)
                data_context = df_result.to_string(index=False)
            except Exception as sql_err:
                print(f"[SQL EXECUTION ERROR]: {sql_err}")
                return "records: 0"
            
            # --- CRITICAL CHANGE: STRICT SHORT RESPONSE FORMATTING ---
            system_prompt_synthesis = """
            You are a minimalist data reporting interface. 
            Your sole job is to present the raw SQL result context in an ultra-short, single-line format.

            ### Rules:
            1. DO NOT write paragraphs, conversational greetings, intros, or polite closings.
            2. DO NOT use complete sentences.
            3. Return only the metric name and the final metric value matching the user query format (e.g., "drivers: 631" or "active buses: 42").
            4. If the data context contains a list of names, print them as a comma-separated single line.
            5. If no records are found, output exactly: "records: 0"
            """
            
            # 3. Formulate the short response
            final_resp = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt_synthesis},
                    {"role": "user", "content": f"User Question: {user_question}\nDatabase Output Matrix:\n{data_context}"}
                ],
                temperature=0.0
            )
            return final_resp.choices[0].message.content.strip()
            
        except Exception as e:
            return f"I encountered an error looking that up: {str(e)}"