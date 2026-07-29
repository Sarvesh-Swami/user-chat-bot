import os
import psycopg2
import psycopg2.extras
from database_manager import DatabaseManager

class SchemaLoader:
    def __init__(self, db_manager):
        self.db_manager = db_manager  # Receive DatabaseManager instance
        self.schema_cache = {}
        self.mock_today = None

    def _load_database_schema(self):
        """Load and cache database schema information from PostgreSQL for core 6 tables"""
        try:
            
            print("[SCHEMA] Loading focused 6-table database schema from PostgreSQL...")
            print("[SCHEMA] Core tables: vehicles, organizations, devices, trips, livetrack, allevents")
            
            # Get table and column information for core tables only
            tables_info = self._get_tables_info()
            
            # Get relationships between core tables only
            relationships = self._get_relationships()
            
            # Get distinct values for enum-like columns
            enum_values = self._get_enum_column_values()
            
            # Get actual date context from the database
            print("[SCHEMA] Getting current date context from database...")
            self.mock_today = self._get_current_date_context()
            
            # Cache the schema information
            self.schema_cache = {
                'tables': tables_info,
                'relationships': relationships,
                'enum_values': enum_values,
                'last_updated': self.mock_today,
                'schema_type': 'focused_6_tables'
            }
            
            print(f"[SCHEMA] Successfully loaded {len(tables_info)} core tables")
            print(f"[SCHEMA] Table names: {', '.join(tables_info.keys())}")
            print(f"[SCHEMA] Found {len(relationships)} relationships between core tables")
            print(f"[SCHEMA] Using date context: {self.mock_today}")
            
        except Exception as e:
            print(f"[SCHEMA ERROR] Failed to load database schema: {e}")
            # Set minimal default schema to prevent crashes
            self.schema_cache = {
                'tables': {}, 
                'relationships': [],
                'enum_values': {},
                'last_updated': '2026-06-20',
                'schema_type': 'fallback'
            }
            self.mock_today = self._get_current_date_context()


    def _get_tables_info(self):
        """Get detailed table and column information from PostgreSQL information_schema for core 6 tables only"""
        
        # Define the 6 core tables we want to focus on (added vehicles table)
        core_tables = ('vehicles', 'organizations', 'devices', 'trips', 'livetrack', 'allevents')
        
        schema_query = """
        SELECT 
            t.table_name,
            t.table_type,
            c.column_name,
            c.data_type,
            c.is_nullable,
            c.column_default,
            c.ordinal_position,
            tc.constraint_type,
            COALESCE(obj_description(pgc.oid), t.table_name) as table_comment
        FROM information_schema.tables t
        LEFT JOIN information_schema.columns c ON t.table_name = c.table_name
        LEFT JOIN information_schema.table_constraints tc ON t.table_name = tc.table_name 
            AND tc.constraint_type = 'PRIMARY KEY'
        LEFT JOIN pg_class pgc ON pgc.relname = t.table_name
        WHERE t.table_schema = 'public' 
            AND t.table_type = 'BASE TABLE'
            AND c.table_schema = 'public'
            AND t.table_name IN ('vehicles', 'organizations', 'devices', 'trips', 'livetrack', 'allevents')
        ORDER BY t.table_name, c.ordinal_position;
        """
        
        try:
            print("[SCHEMA] Executing schema query...")
            results = self.db_manager.execute_query(schema_query)
            print(f"[SCHEMA] Query completed, found {len(results)} rows")
        except Exception as e:
            print(f"[SCHEMA ERROR] Failed to execute schema query: {e}")
            return {}
        
        # Group results by table
        tables = {}
        for row in results:
            table_name = row['table_name']
            if table_name not in tables:
                tables[table_name] = {
                    'columns': [],
                    'table_type': row['table_type'],
                    'description': self._get_table_description(table_name)
                }
            
            # Add column information
            tables[table_name]['columns'].append({
                'name': row['column_name'],
                'type': row['data_type'],
                'nullable': row['is_nullable'] == 'YES',
                'default': row['column_default'],
                'position': row['ordinal_position'],
                'description': self._get_column_description(table_name, row['column_name'])
            })
        
        print(f"[SCHEMA] Processed {len(tables)} tables")
        return tables

    def _get_relationships(self):
        """Get foreign key relationships between core tables only"""
        
        print("[SCHEMA] Getting table relationships...")
        
        # Define the 6 core tables we want to focus on (added vehicles table)
        core_tables = ('vehicles', 'organizations', 'devices', 'trips', 'livetrack', 'allevents')
        
        fk_query = """
        SELECT
            tc.table_name,
            kcu.column_name,
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name,
            tc.constraint_name
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage AS ccu
            ON ccu.constraint_name = tc.constraint_name
            AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_schema = 'public'
            AND tc.table_name IN ('vehicles', 'organizations', 'devices', 'trips', 'livetrack', 'allevents')
            AND ccu.table_name IN ('vehicles', 'organizations', 'devices', 'trips', 'livetrack', 'allevents');
        """
        
        try:
            results = self.db_manager.execute_query(fk_query)
            print(f"[SCHEMA] Found {len(results)} relationships")
        except Exception as e:
            print(f"[SCHEMA ERROR] Failed to get relationships: {e}")
            return []
        
        relationships = []
        for row in results:
            relationships.append({
                'table': row['table_name'],
                'column': row['column_name'],
                'references_table': row['foreign_table_name'],
                'references_column': row['foreign_column_name'],
                'constraint_name': row['constraint_name']
            })
        
        return relationships

    def _get_current_date_context(self):
        """Get the current date context using actual system date (not database date)"""
        try:
            from datetime import datetime
            
            print("[SCHEMA] Getting current date context from system...")
            
            # Use the actual system date (real current date)
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            print(f"[SCHEMA] Using actual system date: {current_date}")
            
            # Also query database to show what data range is available (informational only)
            try:
                date_queries = [
                    "SELECT MAX(end_date) as max_date FROM trips WHERE end_date IS NOT NULL LIMIT 1",
                    "SELECT MIN(start_date) as min_date FROM trips WHERE start_date IS NOT NULL LIMIT 1"
                ]
                
                max_db_date = None
                min_db_date = None
                
                results = self.db_manager.execute_query(date_queries[0])
                if results and len(results) > 0:
                    row = results[0]
                    date_value = row['max_date'] if isinstance(row, dict) else row[0]
                    if date_value:
                        if isinstance(date_value, str):
                            max_db_date = date_value[:10]
                        else:
                            max_db_date = date_value.strftime('%Y-%m-%d')
                
                results = self.db_manager.execute_query(date_queries[1])
                if results and len(results) > 0:
                    row = results[0]
                    date_value = row['min_date'] if isinstance(row, dict) else row[0]
                    if date_value:
                        if isinstance(date_value, str):
                            min_db_date = date_value[:10]
                        else:
                            min_db_date = date_value.strftime('%Y-%m-%d')
                
                if min_db_date and max_db_date:
                    print(f"[SCHEMA] Database data range: {min_db_date} to {max_db_date}")
                    
                    # Calculate age of data
                    from datetime import datetime
                    max_date_obj = datetime.strptime(max_db_date, '%Y-%m-%d')
                    current_date_obj = datetime.strptime(current_date, '%Y-%m-%d')
                    days_old = (current_date_obj - max_date_obj).days
                    
                    if days_old > 0:
                        print(f"[SCHEMA] Note: Database data is {days_old} days old (latest: {max_db_date}, current: {current_date})")
                    
            except Exception as info_error:
                print(f"[SCHEMA] Could not determine database date range (non-critical): {info_error}")
            
            return current_date
            
        except Exception as e:
            print(f"[SCHEMA] Could not get system date: {e}")
            # Fallback to a reasonable default
            return '2026-07-29'
    
    def _get_table_description(self, table_name):
        """Generate business-friendly description for core 6 tables only"""
        
        # Mapping of our 6 core tables to business descriptions
        descriptions = {
            'vehicles': 'Master vehicle registry with identifiers, license plates, device assignments, and vehicle specifications',
            'organizations': 'Fleet management companies and organizations that own vehicles and manage operations',
            'devices': 'GPS tracking devices installed in vehicles for real-time monitoring and data collection',
            'trips': 'Individual journeys taken by vehicles, including start/end locations, duration, and statistics',
            'livetrack': 'Real-time GPS tracking data showing current positions, speed, and movement status of vehicles',
            'allevents': 'Safety and operational events including speeding, harsh braking, geofence violations, and other incidents'
        }
        
        return descriptions.get(table_name, f"Fleet management table: {table_name}")

    def _get_column_description(self, table_name, column_name):
        """Generate business-friendly description for columns in core 6 tables"""
        
        # Table-specific column descriptions for our core tables
        table_specific_descriptions = {
            'vehicles': {
                'vehicle_id': 'Unique user-friendly vehicle identifier (e.g., VEH001, ABC123)',
                'license_plate_number': 'Vehicle license plate number for identification',
                'device_id': 'ID of GPS tracking device installed in this vehicle',
                'organization_id': 'Organization that owns this vehicle',
                'vin': 'Vehicle Identification Number (17-character unique identifier)',
                'make': 'Vehicle manufacturer (Toyota, Ford, etc.)',
                'model': 'Vehicle model name',
                'year': 'Manufacturing year of the vehicle',
                'status': 'Vehicle status (active/inactive)',
                'driver_id': 'Currently assigned driver for this vehicle',
                'score': 'Overall vehicle safety/performance score (0-100)',
                'total_distance': 'Total distance traveled by this vehicle',
                'total_harsh_braking_count': 'Count of harsh braking incidents',
                'total_over_speeding_count': 'Count of speeding violations'
            },
            'organizations': {
                'org_code': 'Unique organization identifier code',
                'name': 'Organization/company name',
                'dot_number': 'Department of Transportation number for commercial fleets',
                'garage_address': 'Physical address of fleet garage/depot',
                'fleet_score': 'Overall safety and performance score for the organization',
                'miles_last_30_days': 'Total miles driven by organization fleet in last 30 days'
            },
            'devices': {
                'device_id': 'Unique identifier for GPS tracking device',
                'imei': 'International Mobile Equipment Identity - unique device identifier',
                'vehicle_id': 'ID of vehicle this device is installed in',
                'organization_id': 'Organization that owns this device',
                'last_ping_ms': 'Timestamp of last communication from device',
                'last_ping_lat': 'Latest GPS latitude from device',
                'last_ping_lng': 'Latest GPS longitude from device',
                'last_speed': 'Last recorded speed from this device',
                'status': 'Device operational status (active/inactive)'
            },
            'trips': {
                'trip_code': 'Unique identifier for individual trip/journey',
                'vehicle_id': 'Vehicle that made this trip',
                'driver_id': 'Driver who made this trip',
                'device_id': 'GPS device that recorded this trip',
                'start_date': 'When the trip started',
                'end_date': 'When the trip ended',
                'start_latitude': 'GPS latitude where trip began',
                'start_longitude': 'GPS longitude where trip began',
                'end_latitude': 'GPS latitude where trip ended',
                'end_longitude': 'GPS longitude where trip ended',
                'trip_distance_miles': 'Total distance traveled during trip in miles',
                'trip_duration_seconds': 'Total trip time in seconds',
                'trip_score': 'Safety score for this trip (0-100)',
                'organization_id': 'Organization that owns the vehicle for this trip'
            },
            'livetrack': {
                'imei': 'Device IMEI that recorded this tracking data',
                'latitude': 'GPS latitude coordinate',
                'longitude': 'GPS longitude coordinate',
                'speed': 'Vehicle speed at time of recording',
                'heading': 'Direction vehicle was traveling (degrees)',
                'ts_in_str': 'Timestamp when this data was recorded',
                'event_type': 'Type of tracking event — use EXACT values from KNOWN COLUMN VALUES section',
                'speed_limit_mph': 'Posted speed limit at this location'
            },
            'allevents': {
                'imei': 'Device IMEI that detected this event',
                'event_type': 'Type of safety/operational event — use EXACT values from KNOWN COLUMN VALUES section',
                'latitude': 'GPS latitude where event occurred',
                'longitude': 'GPS longitude where event occurred',
                'speed': 'Vehicle speed when event occurred',
                'ts_in_str': 'Timestamp when event was detected',
                'driver_id': 'Driver associated with this event',
                'trip_id': 'Trip during which this event occurred'
            }
        }
        
        # Check table-specific descriptions first
        if table_name in table_specific_descriptions:
            table_columns = table_specific_descriptions[table_name]
            if column_name in table_columns:
                return table_columns[column_name]
        
        # Common column patterns (fallback)
        if column_name.endswith('_id') and column_name != 'device_id':
            return f"Unique identifier for {column_name[:-3]} records"
        elif column_name in ['lat', 'latitude']:
            return "GPS latitude coordinate"
        elif column_name in ['lng', 'longitude']:
            return "GPS longitude coordinate"
        elif column_name in ['speed']:
            return "Vehicle speed in kilometers per hour"
        elif column_name in ['created_at', 'updated_at']:
            return "Date/time when record was created or updated"
        elif column_name in ['status']:
            return "Current operational status"
        elif 'driver' in column_name.lower():
            return "Driver name or identifier"
        elif 'vehicle' in column_name.lower():
            return "Vehicle identifier or information"
        elif 'address' in column_name.lower():
            return "Physical address or location description"
        elif 'phone' in column_name.lower():
            return "Contact phone number"
        elif 'distance' in column_name.lower():
            return "Distance measurement in kilometers or miles"
        elif 'score' in column_name.lower():
            return "Performance or safety score (0-100 scale)"
        else:
            return f"Field: {column_name}"

    def _get_enum_column_values(self):
        """Query distinct values for enum-like columns so the LLM uses real values, not guesses."""
        
        # Define which table.column pairs have a small, finite set of categorical values
        enum_columns = {
            'allevents': ['event_type'],
            'livetrack': ['event_type'],
            'trips':     ['trip_status'],
            'vehicles':  ['status'],
            'devices':   ['status'],
        }
        
        enum_values = {}
        
        for table, columns in enum_columns.items():
            for col in columns:
                try:
                    query = f"SELECT DISTINCT {col} FROM public.{table} WHERE {col} IS NOT NULL ORDER BY {col} LIMIT 50;"
                    results = self.db_manager.execute_query(query)
                    values = [row[col] if isinstance(row, dict) else row[0] for row in results if row]
                    if values:
                        key = f"{table}.{col}"
                        enum_values[key] = values
                        print(f"[SCHEMA] Enum values for {key}: {values}")
                except Exception as e:
                    print(f"[SCHEMA] Could not load enum values for {table}.{col}: {e}")
        
        return enum_values

    def load_schema(self):
        """Load and return schema cache and date context"""
        self._load_database_schema()
        return self.schema_cache, self.mock_today
