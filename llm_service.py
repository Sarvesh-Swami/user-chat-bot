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

    def _get_db_schema_string(self):
        """Generates a schema string to inject into the LLM prompt"""
        schema_info = ""
        cursor = self.conn.cursor()
        for table in ["amanbus", "neelkanthtravels", "shantitpt"]:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [f"{col[1]} ({col[2]})" for col in cursor.fetchall()]
            schema_info += f"Table: {table}\nColumns: {', '.join(columns)}\n\n"
        return schema_info

    def answer_user_query(self, user_question: str) -> str:
        if not self.client or not self.deployment_name:
            return "Configuration error: Azure OpenAI client is not set up."

        schema_context = self._get_db_schema_string()
        
        # --- PROMPT ENGINEERING SYSTEM CONTEXT ---
        system_prompt = f"""
        You are an elite database engineer specializing in translating natural language into perfectly optimized SQL queries.
        Your dialect target is: SQLite.

        ### Database Schema Context:
        {schema_context}

        ### Strict Instructions:
        1. Query Composition: Generate a valid SQLite query based ONLY on the tables and columns provided above.
        2. String Comparisons: Use the `LIKE` operator with case-insensitivity or string matching if the user query contains names, titles, or locations that might have mixed casing.
        3. Multi-table handling: If the user asks for a comparison across operators or travels, utilize `UNION ALL` or `JOIN` where appropriate based on common columns.
        4. No Hallucinations: If a question asks for details not present in the columns above, do not invent column names.
        5. Formatting: Output the RAW SQL query string only. 
           - DO NOT wrap the code block in ```sql ... ``` markdown syntax.
           - DO NOT include explanations, comments, or trailing text.
           - Start your response directly with SELECT.
        """
        
        try:
            # 1. Generate the structured SQL instruction
            sql_generation_resp = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User Request: {user_question}\nSQL Query:"}
                ],
                temperature=0.0  # Keep temperature strictly at 0.0 for deterministic code generation
            )
            
            generated_sql = sql_generation_resp.choices[0].message.content.strip()
            
            # Clean accidental markdown if the model leaks backticks anyway
            if generated_sql.startswith("```"):
                generated_sql = generated_sql.replace("```sql", "").replace("```", "").strip()
                
            print(f"\n[TEXT-TO-SQL LOG] Generated Query:\n{generated_sql}\n")
            
            # 2. Execute directly against our structural CSV engine
            df_result = pd.read_sql_query(generated_sql, self.conn)
            data_context = df_result.to_string(index=False)
            
            # 3. Formulate the response with context grounding
            final_resp = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are the automated voice interface for our travel platform. Summarize the returned SQL data comprehensively for the user. If the data matrix is empty, politely inform them that no matching schedule records were found."
                    },
                    {"role": "user", "content": f"User Question: {user_question}\nDatabase Output Matrix:\n{data_context}"}
                ],
                temperature=0.3
            )
            return final_resp.choices[0].message.content
            
        except Exception as e:
            return f"I encountered an error looking that up: {str(e)}"

    def _get_db_schema_string(self) -> str:
        """
        Generates a highly detailed schema string, including data types 
        and low-cardinality sample values to prevent LLM hallucinations.
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
                
                # Fetch distinct sample values to give the LLM categorical context
                try:
                    cursor.execute(f"SELECT DISTINCT {col_name} FROM {table} LIMIT 3")
                    samples = [str(row[0]) for row in cursor.fetchall() if row[0] is not None]
                    sample_str = f" (e.g., {', '.join(samples)})" if samples else ""
                except Exception:
                    sample_str = ""
                    
                column_specs.append(f"  - {col_name} ({col_type}){sample_str}")
                
            schema_info += f"Table Name: {table}\nColumns:\n" + "\n".join(column_specs) + "\n\n"
            
        return schema_info