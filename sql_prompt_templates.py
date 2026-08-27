"""
SQL Prompt Templates Module
Provides category-specific SQL query hints for the LLM to generate better PostgreSQL queries.
"""

class SQLPromptTemplates:
    """Manages SQL prompt templates and hints for different query categories"""
    
    def __init__(self):
        """Initialize the SQL prompt templates with category-specific hints"""
        self.hints = {
            "vehicle_location": """
### Vehicle Location Query Hints:
- For current location: Use devices.last_ping_lat, devices.last_ping_lng, devices.last_location
- For historical location: JOIN trips or livetrack tables
- For currently moving vehicles: Use devices.last_speed > 0
- Example: SELECT v.vehicle_id, v.license_plate_number, d.last_location, d.last_speed
          FROM vehicles v
          JOIN devices d ON v.device_id = d.id
          WHERE d.last_speed > 0;
""",
            
            "vehicle_status": """
### Vehicle Status Query Hints:
- Check vehicles.status for operational status
- Check devices.device_status for device connectivity
- For idle vehicles: devices.last_speed = 0
- For moving vehicles: devices.last_speed > 0
- Example: SELECT v.vehicle_id, v.status, d.device_status, d.last_speed
          FROM vehicles v
          JOIN devices d ON v.device_id = d.id
          WHERE v.status = 'active';
""",
            
            "trip_history": """
### Trip History Query Hints:
- Use trips table for journey data
- JOIN with vehicles for vehicle details
- Use start_time and end_time for time ranges
- Calculate duration: end_time - start_time
- Example: SELECT t.id, v.vehicle_id, t.start_location, t.end_location, t.distance_km, t.start_time
          FROM trips t
          JOIN vehicles v ON t.vehicle_id = v.id
          WHERE t.start_time >= CURRENT_DATE - INTERVAL '7 days'
          ORDER BY t.start_time DESC;
""",
            
            "safety_events": """
### Safety Events Query Hints:
- Use allevents table for incidents
- Filter by event_type for specific events (harsh_braking, overspeeding, etc.)
- JOIN with vehicles via devices.imei for vehicle context
- Example: SELECT ae.event_type, ae.ts_in_str, v.vehicle_id, ae.location
          FROM allevents ae
          JOIN devices d ON ae.imei = d.imei
          JOIN vehicles v ON d.id = v.device_id
          WHERE ae.event_type IN ('harsh_braking', 'overspeeding')
          ORDER BY ae.ts_in_str DESC;
""",
            
            "organization": """
### Organization Query Hints:
- Use ILIKE for organization name searches: o.name ILIKE '%search_term%'
- JOIN vehicles via organization_id
- Example: SELECT o.name, COUNT(v.id) as vehicle_count
          FROM organizations o
          LEFT JOIN vehicles v ON o.id = v.organization_id
          WHERE o.name ILIKE '%company%'
          GROUP BY o.name;
""",
            
            "device_tracking": """
### Device Tracking Query Hints:
- devices table for GPS hardware info
- livetrack for real-time GPS positions
- Use devices.imei as the key identifier
- Example: SELECT d.imei, d.device_status, d.last_ping_time, v.vehicle_id
          FROM devices d
          LEFT JOIN vehicles v ON d.id = v.device_id
          WHERE d.device_status = 'online';
""",
            
            "statistics": """
### Statistics Query Hints:
- Use aggregate functions: COUNT(), AVG(), SUM(), MAX(), MIN()
- GROUP BY for categorization
- Example: SELECT v.make, v.model, COUNT(*) as count, AVG(d.last_speed) as avg_speed
          FROM vehicles v
          JOIN devices d ON v.device_id = d.id
          GROUP BY v.make, v.model
          ORDER BY count DESC;
""",
            
            "default": """
### General Query Hints:
- Start with vehicles table for most queries
- JOIN devices for location/speed data
- JOIN trips for journey history
- JOIN allevents for safety incidents
- JOIN organizations for company info
- Use appropriate WHERE clauses to filter data
- Always ORDER BY timestamp columns for chronological data
- Apply LIMIT for large result sets
"""
        }
    
    def get_hints(self, query_category: str) -> str:
        """
        Get SQL hints for a specific query category
        
        Args:
            query_category: The category of query (e.g., 'vehicle_location', 'trip_history')
            
        Returns:
            String containing SQL hints for the category
        """
        return self.hints.get(query_category, self.hints["default"])
    
    def get_full_hints(self) -> str:
        """
        Get all SQL hints as a comprehensive fallback
        
        Returns:
            String containing all SQL hints combined
        """
        full_hints = "### Comprehensive SQL Query Hints:\n\n"
        
        # Combine all hints in a logical order
        hint_order = [
            "vehicle_location",
            "vehicle_status", 
            "trip_history",
            "safety_events",
            "organization",
            "device_tracking",
            "statistics",
            "default"
        ]
        
        for category in hint_order:
            if category in self.hints:
                full_hints += self.hints[category] + "\n"
        
        return full_hints
    
    def get_available_categories(self) -> list:
        """
        Get list of available hint categories
        
        Returns:
            List of category names
        """
        return list(self.hints.keys())
