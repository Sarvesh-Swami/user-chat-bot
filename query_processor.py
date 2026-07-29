import pandas as pd
import json
import re
import time
from typing import Dict, Any, List

class QueryProcessor:
    def __init__(self, db_manager, llm_client, geocoding_service, schema_cache=None):
        self.db_manager = db_manager
        self.llm_client = llm_client
        self.geocoding_service = geocoding_service
        # Build real column map from schema_cache for the SQL column validator
        self._column_map = self._build_column_map(schema_cache or {})

    def _convert_dataframe_to_serializable_dict(self, df, enhance_locations=True, skip_geocoding=False):
        """Convert DataFrame to JSON-serializable dict with all numpy types converted and optional location enhancement"""
        try:
            # Deduplicate DataFrame column names if query returned duplicate names (e.g., SELECT v.*, d.*)
            if not df.columns.is_unique:
                cols = pd.Series(df.columns)
                for duplicate in cols[cols.duplicated()].unique():
                    dup_indices = cols[cols == duplicate].index.values
                    for i, idx in enumerate(dup_indices[1:], start=2):
                        cols[idx] = f"{duplicate}_{i}"
                df.columns = cols

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
                    elif hasattr(value, 'isoformat'):  # datetimes, dates, timestamps
                        serializable_record[key] = value.isoformat()
                    elif isinstance(value, bytes):
                        serializable_record[key] = value.decode('utf-8')
                    elif type(value).__name__ == 'Decimal':  # decimal.Decimal
                        serializable_record[key] = float(value)
                    else:
                        serializable_record[key] = value
                
                serializable_records.append(serializable_record)
            
            # Enhance with location data if requested and appropriate (skipped for PDF exports)
            if enhance_locations and not skip_geocoding and self._should_enhance_with_locations(serializable_records):
                print("[LOCATION ENHANCEMENT] Adding human-readable locations to response")
                serializable_records = self.geocoding_service.enhance_location_data(serializable_records)
            
            return serializable_records
            
        except Exception as e:
            print(f"[SERIALIZATION ERROR]: Failed to convert DataFrame: {e}")
            # Fallback: use json.dumps with custom serializer
            try:
                records = json.loads(json.dumps(df.to_dict(orient="records"), default=self.db_manager._json_serializer))
                if enhance_locations and not skip_geocoding and self._should_enhance_with_locations(records):
                    print("[LOCATION ENHANCEMENT] Adding locations to fallback response")
                    records = self.geocoding_service.enhance_location_data(records)
                return records
            except Exception as fallback_error:
                print(f"[SERIALIZATION FALLBACK ERROR]: {fallback_error}")
                return []

    def _should_enhance_with_locations(self, records):
        """Determine if records should be enhanced with location data"""
        if not records:
            return False
        
        # Check if any record has GPS coordinates
        for record in records[:3]:  # Check first few records only
            # Look for latitude/longitude fields
            has_coords = False
            for lat_field in ['last_ping_lat', 'latitude', 'lat', 'start_latitude', 'end_latitude']:
                for lng_field in ['last_ping_lng', 'longitude', 'lng', 'start_longitude', 'end_longitude']:
                    if (lat_field in record and lng_field in record and 
                        record[lat_field] is not None and record[lng_field] is not None):
                        has_coords = True
                        break
                if has_coords:
                    break
            
            if has_coords:
                return True
        
        return False

    def _execute_with_smart_limits(self, sql_query, user_question):
        """Execute SQL with intelligent row count management and auto-summarization on PostgreSQL"""
        
        query_type = self.llm_client._classify_query_type(user_question, sql_query)
        MAX_CHAT_ROWS = 7
        PDF_MAX_ROWS  = 200   # Hard cap: max rows for PDF report generation (instant render)
        
        try:
            # Fast probe query with LIMIT (PDF_MAX_ROWS + 1) without heavy ORDER BY/LIMIT to avoid slow table sorts
            has_explicit_limit = bool(re.search(r'\bLIMIT\s+\d+', sql_query, re.IGNORECASE))

            if has_explicit_limit:
                probe_sql = sql_query
                df_probe = pd.read_sql_query(probe_sql, self.db_manager.conn)
                fetched_count = len(df_probe)
            else:
                # Fast probe query with LIMIT (PDF_MAX_ROWS + 1) without heavy ORDER BY/LIMIT to avoid slow table sorts
                probe_clean = re.sub(r'\s+ORDER\s+BY\s+.*', '', sql_query, flags=re.IGNORECASE | re.DOTALL)
                probe_clean = re.sub(r'\s+LIMIT\s+\d+.*', '', probe_clean, flags=re.IGNORECASE | re.DOTALL)
                probe_sql = f"{probe_clean.rstrip(';')} LIMIT {PDF_MAX_ROWS + 1};"
                df_probe = pd.read_sql_query(probe_sql, self.db_manager.conn)
                fetched_count = len(df_probe)
            
            print(f"[SMART LIMITS] Fast probe fetched {fetched_count} rows (Query Type: {query_type.upper()})")
            
            # Case 1: 0 rows returned -> anti-hallucination / context-aware empty response
            if fetched_count == 0:
                return self._handle_empty_result(sql_query, user_question, query_type)
            
            # Case 2: 1 to 7 rows -> small result, pass full data to chat LLM synthesis
            if fetched_count <= MAX_CHAT_ROWS:
                df_result = pd.read_sql_query(sql_query, self.db_manager.conn)
                serialized = self._convert_dataframe_to_serializable_dict(df_result)
                num_cols = len(serialized[0]) if serialized else 0

                # is_detail_preview: triggered when user asks for full/all details on a SINGLE entity.
                # Condition: SPECIFIC query type, exactly 1 row returned, and ≥8 columns in the result
                # (indicating a rich JOIN query like SELECT v.*, d.*, o.* that covers all vehicle info).
                # In this case we show a clean preview card + offer a PDF instead of a markdown text blob.
                is_detail_preview = (
                    query_type.lower() == "specific"
                    and fetched_count == 1
                    and num_cols >= 8
                )

                if is_detail_preview:
                    print(f"[SMART LIMITS] Single-row wide result ({num_cols} cols). Flagging as detail_preview.")
                else:
                    print(f"[SMART LIMITS] Result count {fetched_count} <= {MAX_CHAT_ROWS}. Routing to Chat Synthesis LLM for text response.")

                return {
                    "is_summary": False,
                    "is_detail_preview": is_detail_preview,
                    "data": serialized,
                    "total_rows": int(len(df_result)),
                    "num_cols": num_cols,
                    "query_type": query_type
                }
            
            # Case 3: > 7 rows -> large result, generate PDF report using probe data (skip geocoding & zero DB overhead)
            print(f"[SMART LIMITS] Result count > {MAX_CHAT_ROWS} threshold. Triggering PDF report generation.")
            pdf_df = df_probe.head(PDF_MAX_ROWS)
            full_data = self._convert_dataframe_to_serializable_dict(pdf_df, skip_geocoding=True)
            
            total_rows_display = f"{PDF_MAX_ROWS}+" if fetched_count > PDF_MAX_ROWS else len(pdf_df)
            print(f"[PDF REPORT] Generating PDF with {len(pdf_df)} rows")
            
            return {
                "is_pdf_report": True,
                "data": full_data,
                "total_rows": total_rows_display,
                "query_type": query_type
            }
                
        except Exception as e:
            print(f"[SMART LIMITS ERROR]: {e}")
            try:
                if self.db_manager.conn and not self.db_manager.conn.closed:
                    self.db_manager.conn.rollback()
            except Exception:
                pass
            raise e

    def _handle_empty_result(self, sql_query, user_question, query_type):
        """Handle 0-row query results: deterministic fallback for specific vehicle IDs, or pass empty dataset to synthesis LLM for context-aware responses."""
        if self._is_specific_vehicle_query(sql_query):
            vehicle_id = self._extract_vehicle_id(sql_query)
            if vehicle_id:
                print(f"[VEHICLE FALLBACK] No current data for vehicle {vehicle_id}, checking last known data")
                last_known = self._get_last_known_data(vehicle_id)
                return {
                    "is_summary": True,
                    "response": self._generate_vehicle_not_found_response(vehicle_id, last_known),
                    "total_rows": 0,
                    "query_type": query_type,
                    "is_empty": True
                }

        print(f"[EMPTY RESULT] 0 rows returned. Passing empty dataset to synthesis LLM for context-aware response.")
        return {
            "is_summary": False,
            "data": [],
            "total_rows": 0,
            "query_type": query_type,
            "is_empty": True
        }

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
            
            return json.dumps(response_data, default=self.db_manager._json_serializer)
            
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
            }, default=self.db_manager._json_serializer)

    def _should_synthesize_with_llm(self, sql_query, user_question):
        """Determine if a query should use LLM synthesis even for large result sets"""
        sql_lower = sql_query.lower()
        question_lower = user_question.lower()
        
        # Trip summary queries should use LLM synthesis for better formatting
        trip_summary_indicators = [
            "summary" in question_lower and "trip" in question_lower,
            "summary" in question_lower and "vehicle" in question_lower,
            "trip" in sql_lower and ("count" in sql_lower or "sum" in sql_lower or "avg" in sql_lower),
            "vehicle" in question_lower and ("total" in question_lower or "summary" in question_lower)
        ]
        
        return any(trip_summary_indicators)

    def _get_trip_summary_stats(self, sql_query):
        """Get aggregated trip statistics for LLM synthesis"""
        try:
            
            # Execute a modified version of the query to get summary statistics
            stats_sql = f"""
            SELECT 
                vehicle_id,
                license_plate_number,
                COUNT(*) as total_trips,
                COALESCE(SUM(trip_distance_miles), 0) as total_distance_miles,
                COALESCE(AVG(trip_score), 0) as average_trip_score,
                COALESCE(SUM(trip_duration_seconds), 0) as total_duration_seconds,
                MIN(start_date) as first_trip_start,
                MAX(end_date) as last_trip_end
            FROM ({sql_query.rstrip(';')}) AS trip_data
            GROUP BY vehicle_id, license_plate_number
            """
            
            df_result = pd.read_sql_query(stats_sql, self.db_manager.conn)
            return self._convert_dataframe_to_serializable_dict(df_result)
            
        except Exception as e:
            print(f"[TRIP SUMMARY STATS ERROR]: {e}")
            # Fallback to basic count
            try:
                basic_sql = f"""
                SELECT 
                    vehicle_id,
                    license_plate_number,
                    COUNT(*) as total_trips
                FROM ({sql_query.rstrip(';')}) AS trip_data
                GROUP BY vehicle_id, license_plate_number
                """
                df_result = pd.read_sql_query(basic_sql, self.db_manager.conn)
                return self._convert_dataframe_to_serializable_dict(df_result)
            except Exception as fallback_error:
                print(f"[TRIP SUMMARY FALLBACK ERROR]: {fallback_error}")
                return [{"error": "Unable to generate trip summary statistics"}]

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

    def _fix_postgresql_query_issues(self, sql_query):
        """Optimize queries for PostgreSQL and handle database-specific syntax"""
        
        # Check for inefficient "currently moving" query patterns and warn
        sql_lower = sql_query.lower()
        if ('livetrack' in sql_lower and 
            ('now()' in sql_lower or 'current_timestamp' in sql_lower or 'interval' in sql_lower) and
            ('speed' in sql_lower or 'moving' in sql_lower)):
            print("[POSTGRESQL WARNING] Detected potentially inefficient 'currently moving' query using livetrack with time filters.")
            print("[POSTGRESQL WARNING] For better performance, use: devices.last_speed > 0 instead of livetrack + NOW() - INTERVAL filters.")
        
        # Optimize heavy livetrack/allevents joins by adding top-N limit so PostgreSQL uses instant Top-N Heapsort
        if ('livetrack' in sql_query.lower() or 'allevents' in sql_query.lower()) and 'limit' not in sql_query.lower():
            print("[POSTGRESQL] Appending top-N LIMIT to prevent full 300k row sorting")
            sql_query = f"{sql_query.rstrip(';')} LIMIT 200;"

        sql_upper = sql_query.upper()
        
        # Handle spatial distance calculations
        if 'CALCULATE_DISTANCE' in sql_upper:
            print("[POSTGRESQL] Converting custom distance function to PostgreSQL spatial calculation")
            # Replace custom SQLite function with PostgreSQL haversine formula
            sql_query = re.sub(
                r'CALCULATE_DISTANCE\s*\(\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^)]+)\s*\)',
                r'(6371000 * acos(greatest(-1, least(1, cos(radians(\1)) * cos(radians(\3)) * cos(radians(\4) - radians(\2)) + sin(radians(\1)) * sin(radians(\3))))))',
                sql_query,
                flags=re.IGNORECASE
            )
        
        # Convert SQLite string concatenation to PostgreSQL CONCAT where needed
        if '||' in sql_query and 'date' in sql_query.lower():
            # Convert "date || ' ' || time" patterns to CONCAT for better readability
            sql_query = re.sub(
                r"([a-zA-Z_]+\.?)date\s*\|\|\s*'\s*'\s*\|\|\s*([a-zA-Z_]+\.?)([a-zA-Z_]+)",
                r"(\1date || ' ' || \2\3)",  # Keep || syntax as it works in PostgreSQL
                sql_query,
                flags=re.IGNORECASE
            )
        
        # Handle geofence entry/exit queries with PostgreSQL optimizations
        if ('LAG(' in sql_upper and 
            'GEOFENCE' in sql_upper and 
            ('ENTRY' in sql_upper or 'EXIT' in sql_upper)):
            
            print("[POSTGRESQL] Detected geofence LAG query - using PostgreSQL native capabilities")
            
            # Extract vehicle ID
            vid_match = re.search(r'(?:vehicle_id|vid)\s*=\s*(\d+)', sql_query, re.IGNORECASE)
            vehicle_id = vid_match.group(1) if vid_match else None
            
            if vehicle_id:
                # Generate PostgreSQL-optimized geofence query
                optimized_query = f"""
                WITH vehicle_positions AS (
                    SELECT 
                        COALESCE(fh.vehicle_id, fh.vid) as vehicle_id,
                        fh.latitude as lat,
                        fh.longitude as lng,
                        fh.recorded_at as timestamp,
                        fh.date,
                        gl.geofence_name,
                        gl.geofence_id,
                        gl.radius_meters,
                        -- PostgreSQL haversine distance calculation
                        (6371000 * acos(greatest(-1, least(1, 
                            cos(radians(fh.latitude)) * cos(radians(gl.latitude)) * 
                            cos(radians(gl.longitude) - radians(fh.longitude)) + 
                            sin(radians(fh.latitude)) * sin(radians(gl.latitude))
                        )))) AS distance_meters,
                        CASE 
                            WHEN (6371000 * acos(greatest(-1, least(1, 
                                cos(radians(fh.latitude)) * cos(radians(gl.latitude)) * 
                                cos(radians(gl.longitude) - radians(fh.longitude)) + 
                                sin(radians(fh.latitude)) * sin(radians(gl.latitude))
                            )))) <= gl.radius_meters 
                            THEN 1 ELSE 0 
                        END AS is_inside
                    FROM fleet_history fh
                    CROSS JOIN geofence_lookup gl
                    WHERE COALESCE(fh.vehicle_id, fh.vid) = {vehicle_id}
                    ORDER BY fh.recorded_at ASC, fh.date ASC
                ),
                position_changes AS (
                    SELECT 
                        *,
                        LAG(is_inside, 1, 0) OVER (
                            PARTITION BY vehicle_id, geofence_id 
                            ORDER BY timestamp ASC
                        ) AS prev_inside
                    FROM vehicle_positions
                ),
                entry_exit_events AS (
                    SELECT 
                        vehicle_id,
                        geofence_name,
                        timestamp,
                        CASE 
                            WHEN prev_inside = 0 AND is_inside = 1 THEN 'ENTRY'
                            WHEN prev_inside = 1 AND is_inside = 0 THEN 'EXIT'
                            ELSE NULL
                        END AS event_type,
                        ROUND(distance_meters::numeric, 2) AS distance_meters
                    FROM position_changes
                    WHERE (prev_inside = 0 AND is_inside = 1) OR (prev_inside = 1 AND is_inside = 0)
                )
                SELECT 
                    vehicle_id,
                    geofence_name,
                    event_type,
                    timestamp::text AS event_timestamp,
                    distance_meters
                FROM entry_exit_events
                WHERE event_type IS NOT NULL
                ORDER BY timestamp ASC;
                """
                
                print("[POSTGRESQL] Generated PostgreSQL-optimized geofence query")
                return optimized_query.strip()
        
        # Convert SQLite-specific syntax to PostgreSQL equivalents
        # Handle LIMIT with OFFSET for pagination
        if 'LIMIT' in sql_upper and 'OFFSET' not in sql_upper:
            # PostgreSQL supports both SQLite-style LIMIT and its own LIMIT/OFFSET
            pass  # Keep as is, PostgreSQL handles SQLite LIMIT syntax
        
        # Convert any remaining SQLite-specific functions
        sql_query = re.sub(r'\bIFNULL\b', 'COALESCE', sql_query, flags=re.IGNORECASE)
        
        return sql_query

    # ---------------------------------------------------------------------------
    # Option 2 — SQL Column Validator & Auto-Corrector
    # ---------------------------------------------------------------------------

    def _build_column_map(self, schema_cache: dict) -> dict:
        """
        Build a lookup dict {table_name: set(column_names)} from schema_cache.
        Called once at startup; used by _validate_and_correct_columns on every query.
        """
        column_map = {}
        tables = schema_cache.get("tables", {})
        for table_name, table_info in tables.items():
            cols = {c["name"].lower() for c in table_info.get("columns", [])}
            column_map[table_name.lower()] = cols
        if column_map:
            print(f"[COLUMN VALIDATOR] Built column map for {len(column_map)} tables: {', '.join(sorted(column_map.keys()))}")
        return column_map

    # Known LLM column name mistakes → correct DB column names.
    # Format: {table_name: {wrong_name: correct_name}}
    # None as correct_name means the column doesn't exist at all (flag but don't replace).
    _KNOWN_CORRECTIONS = {
        "vehicles": {
            "total_distance":          "total_distance_in_mile",
            "total_distance_miles":    "total_distance_in_mile",
            "mileage":                 "total_distance_in_mile",
            "odometer":                "total_distance_in_mile",
            "safety_score":            "score",
            "driver_score":            "score",
            "trip_count":              None,   # computed, not stored
            "trips_count":             None,
            "num_trips":               None,
        },
        "trips": {
            "distance":                "trip_distance_miles",
            "miles":                   "trip_distance_miles",
            "duration":                "trip_duration_seconds",
            "status":                  "trip_status",
        },
        "livetrack": {
            "timestamp":               "ts_in_str",
            "ts":                      "ts_in_str",
            "lat":                     "latitude",
            "lng":                     "longitude",
            "long":                    "longitude",
        },
        "allevents": {
            "timestamp":               "ts_in_str",
            "ts":                      "ts_in_str",
            "lat":                     "latitude",
            "lng":                     "longitude",
            "long":                    "longitude",
            "event":                   "event_type",
        },
        "devices": {
            "lat":                     "last_ping_lat",
            "lng":                     "last_ping_lng",
            "long":                    "last_ping_lng",
            "speed":                   "last_speed",
            "ping_time":               "last_ping_ms",
        },
    }

    def _validate_and_correct_columns(self, sql: str) -> str:
        """
        Option 2 — Post-generation SQL column validator.

        Steps:
          1. Parse FROM / JOIN clauses to build alias → table_name mapping.
          2. Scan all alias.column references in the SQL.
          3. Apply KNOWN_CORRECTIONS for common LLM column name mistakes.
          4. Check remaining references against the real schema_cache column map.
          5. Log every correction and any still-unknown references (soft warning only).

        Returns the corrected SQL string. Never raises — on any error it returns
        the original SQL unchanged so the pipeline is not disrupted.
        """
        try:
            corrections_made = []

            # ── Step 1: build alias→table map from FROM / JOIN clauses ──────────
            # Matches patterns like: FROM vehicles v, JOIN trips t ON ...
            alias_map = {}   # {alias_lower: table_name_lower}
            # Match "table_name alias" or "table_name AS alias" patterns
            from_join_pattern = re.compile(
                r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)',
                re.IGNORECASE
            )
            for match in from_join_pattern.finditer(sql):
                table_name = match.group(1).lower()
                alias      = match.group(2).lower()
                # Skip SQL keywords that might be accidentally captured
                sql_keywords = {"on", "where", "and", "or", "inner", "left", "right",
                                "outer", "cross", "natural", "using", "set", "as"}
                if alias not in sql_keywords:
                    alias_map[alias] = table_name
                    # Also register table name itself as its own alias
                if table_name not in sql_keywords:
                    alias_map[table_name] = table_name

            # ── Step 2: find all alias.column references ─────────────────────
            col_ref_pattern = re.compile(
                r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b'
            )

            corrected_sql = sql

            for match in col_ref_pattern.finditer(sql):
                alias  = match.group(1).lower()
                column = match.group(2).lower()
                ref    = match.group(0)          # original text e.g. "v.total_distance"

                # Resolve alias to table name
                table = alias_map.get(alias)
                if not table:
                    continue   # alias unknown — skip (could be a subquery alias, CTE, etc.)

                # ── Step 3: apply KNOWN_CORRECTIONS first ─────────────────────
                table_corrections = self._KNOWN_CORRECTIONS.get(table, {})
                if column in table_corrections:
                    correct_col = table_corrections[column]
                    if correct_col is None:
                        # Column doesn't exist as a stored field — warn only
                        print(f"[COLUMN VALIDATOR] [WARN] '{ref}' — '{column}' is not a stored column on '{table}'. May need a subquery.")
                    else:
                        new_ref = f"{match.group(1)}.{correct_col}"
                        if new_ref != ref:
                            corrected_sql = re.sub(rf'\b{re.escape(ref)}\b', new_ref, corrected_sql)
                            corrections_made.append(f"{ref} -> {new_ref}")
                    continue

                # ── Step 4: check against real schema_cache column map ─────────
                if self._column_map and table in self._column_map:
                    if column not in self._column_map[table]:
                        print(f"[COLUMN VALIDATOR] [WARN] Unknown column '{ref}' (table='{table}' has no column '{column}')")
                # (Soft warning only — do not modify the SQL for unknown cols;
                #  the DB will return an error which triggers self-healing retry.)

            if corrections_made:
                print(f"[COLUMN VALIDATOR] [OK] Auto-corrected {len(corrections_made)} column reference(s): {', '.join(corrections_made)}")
            else:
                print(f"[COLUMN VALIDATOR] [OK] All column references look valid.")

            return corrected_sql

        except Exception as e:
            # Never break the pipeline — return original SQL on any error
            print(f"[COLUMN VALIDATOR] Warning: validation failed ({e}), using original SQL.")
            return sql

    def _is_specific_vehicle_query(self, sql_query):
        """Check if query is looking for a specific vehicle ID or license plate (exact = match only, not ILIKE org searches)"""
        # Only match v.vehicle_id = '...' or v.license_plate_number = '...' with exact equality
        # Exclude ILIKE patterns (used for org name searches)
        return bool(re.search(r'\bv\.(vehicle_id|license_plate_number)\s*=\s*\'([^\']+)\'', sql_query, re.IGNORECASE))

    def _extract_vehicle_id(self, sql_query):
        """Extract vehicle ID or license plate from SQL query"""
        match = re.search(r'\bv\.(vehicle_id|license_plate_number)\s*=\s*\'([^\']+)\'', sql_query, re.IGNORECASE)
        return match.group(2) if match else None

    def _get_last_known_data(self, vehicle_id):
        """Get the most recent data for a specific vehicle from PostgreSQL"""
        try:
            
            # Target actual PostgreSQL tables: trips, devices, vehicles
            fallback_queries = [
                f"""
                SELECT v.vehicle_id, d.last_speed as speed, 'active' as mode, 
                       t.start_date as date, d.last_ping_ms as gpstime, t.start_address as addr, null as drivername
                FROM vehicles v
                LEFT JOIN devices d ON v.device_id = d.id
                LEFT JOIN trips t ON v.id = t.vehicle_id
                WHERE v.vehicle_id = '{vehicle_id}' OR v.license_plate_number = '{vehicle_id}'
                ORDER BY t.start_date DESC
                LIMIT 1
                """
            ]
            
            for query in fallback_queries:
                try:
                    results = self.db_manager.execute_query(query)
                    if results:
                        result = results[0]  # Get first row
                        
                        # Convert to dict and ensure JSON serializable types
                        record = dict(result)
                        for key, value in record.items():
                            if hasattr(value, 'item'):  # numpy scalar
                                record[key] = value.item()
                            elif value is None:
                                record[key] = None
                            else:
                                record[key] = str(value) if not isinstance(value, (int, float)) else value
                        return record
                except Exception as e:
                    print(f"[FALLBACK QUERY ERROR] Query failed: {e}")
                    continue
            
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
            
            disp_msg = f"No current location data available for vehicle **{vehicle_id}** today. Last recorded activity was on {last_known_data.get('date', 'Unknown date')} near {location}."
            return json.dumps({
                "type": "text",
                "status": "no_current_data",
                "display_value": disp_msg,
                "query_type": "vehicle_status",
                "vehicle_id": str(vehicle_id),
                "message": disp_msg,
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
                    f"Get route history: 'vehicle {vehicle_id} route history'"
                ],
                "note": f"Last seen on {last_known_data.get('date', 'unknown date')} at {location}"
            }, default=self.db_manager._json_serializer)
        else:
            disp_msg = f"Vehicle **{vehicle_id}** was not found in the fleet database. Please verify the vehicle ID or license plate number."
            return json.dumps({
                "type": "text",
                "status": "vehicle_not_found",
                "display_value": disp_msg,
                "query_type": "vehicle_status", 
                "vehicle_id": str(vehicle_id),
                "message": disp_msg,
                "suggestions": [
                    "Verify the vehicle ID or license plate is correct",
                    "Try searching for driver name or organization"
                ],
                "note": "This vehicle ID does not exist in our records"
            }, default=self.db_manager._json_serializer)
    
    
    
