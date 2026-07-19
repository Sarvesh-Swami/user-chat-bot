import pandas as pd
import json
import re
import time
from typing import Dict, Any, List

class QueryProcessor:
    def __init__(self, db_manager, llm_client, geocoding_service):
        self.db_manager = db_manager
        self.llm_client = llm_client
        self.geocoding_service = geocoding_service

    def _convert_dataframe_to_serializable_dict(self, df, enhance_locations=True):
        """Convert DataFrame to JSON-serializable dict with all numpy types converted and optional location enhancement"""
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
            
            # Enhance with location data if requested and appropriate
            if enhance_locations and self._should_enhance_with_locations(serializable_records):
                print("[LOCATION ENHANCEMENT] Adding human-readable locations to response")
                serializable_records = self.geocoding_service.enhance_location_data(serializable_records)
            
            return serializable_records
            
        except Exception as e:
            print(f"[SERIALIZATION ERROR]: Failed to convert DataFrame: {e}")
            # Fallback: use json.dumps with custom serializer
            try:
                records = json.loads(json.dumps(df.to_dict(orient="records"), default=self.db_manager._json_serializer))
                if enhance_locations and self._should_enhance_with_locations(records):
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
        
        # Classify query type to determine appropriate limits
        query_type = self.llm_client._classify_query_type(user_question, sql_query)
        
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
            # Step 1: Try to get row count first using PostgreSQL syntax
            count_sql = f"SELECT COUNT(*) as row_count FROM ({sql_query.rstrip(';')}) AS count_subquery"
            
            try:
                count_results = self.db_manager.execute_query(count_sql)
                total_rows = int(count_results[0][0])
                print(f"[SMART LIMITS] Query will return {total_rows} rows (Type: {query_type.upper()})")
            
            except Exception as count_error:
                print(f"[SMART LIMITS] Count query failed: {count_error}")
                print("[SMART LIMITS] Falling back to direct execution")
                
                # Fallback: execute original query directly
                try:
                    df_result = pd.read_sql_query(sql_query, self.db_manager.conn)
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
                df_result = pd.read_sql_query(sql_query, self.db_manager.conn)
                
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
                df_result = pd.read_sql_query(limited_sql, self.db_manager.conn)
                
                return {
                    "is_summary": False,
                    "data": self._convert_dataframe_to_serializable_dict(df_result),
                    "total_rows": int(total_rows),
                    "is_limited": True,
                    "showing": len(df_result),
                    "query_type": query_type
                }
                
            else:
                # Large result: Check if this is a trip summary query that needs LLM synthesis
                if self._should_synthesize_with_llm(sql_query, user_question):
                    # For trip summaries, get aggregated stats and pass to LLM for synthesis
                    trip_stats = self._get_trip_summary_stats(sql_query)
                    return {
                        "is_summary": False,
                        "data": trip_stats,
                        "total_rows": int(total_rows),
                        "query_type": query_type,
                        "is_trip_summary": True
                    }
                else:
                    # Standard large result: return summary only
                    summary_response = self.llm_client._generate_summary_response(sql_query, user_question, total_rows)
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
                df_result = pd.read_sql_query(sql_query, self.db_manager.conn)
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
        
        sql_upper = sql_query.upper()
        
        # PostgreSQL has excellent window function support, so most LAG issues are resolved
        print("[POSTGRESQL] Processing query for PostgreSQL optimization")
        
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

    def _is_specific_vehicle_query(self, sql_query):
        """Check if query is looking for a specific vehicle ID"""
        return bool(re.search(r'\bvid\s*=\s*\d+', sql_query, re.IGNORECASE))

    def _extract_vehicle_id(self, sql_query):
        """Extract vehicle ID from SQL query"""
        match = re.search(r'\bvid\s*=\s*(\d+)', sql_query, re.IGNORECASE)
        return match.group(1) if match else None

    def _get_last_known_data(self, vehicle_id):
        """Get the most recent data for a specific vehicle from PostgreSQL"""
        try:
            
            # Try different possible column names and table structures
            fallback_queries = [
                f"""
                SELECT COALESCE(vehicle_id, vid) as vehicle_id, speed, status as mode, 
                       date, recorded_at as gpstime, address as addr, driver_name as drivername
                FROM fleet_history 
                WHERE COALESCE(vehicle_id, vid) = {vehicle_id} 
                ORDER BY recorded_at DESC, date DESC 
                LIMIT 1
                """,
                f"""
                SELECT vid as vehicle_id, speed, mode, date, gpstime, addr, drivername
                FROM fleet_history 
                WHERE vid = {vehicle_id} 
                ORDER BY date DESC, gpstime DESC 
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
            }, default=self.db_manager._json_serializer)
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
            }, default=self.db_manager._json_serializer)
    
    
    
