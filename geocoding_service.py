import requests
from functools import lru_cache

class GeocodingService:
    def __init__(self):
        # No dependencies needed - this service is completely independent!
        pass

    @lru_cache(maxsize=1000)
    def reverse_geocode(self, latitude, longitude):
        """Convert latitude/longitude to human-readable location using BigDataCloud API"""
        try:
            # Round coordinates to 4 decimal places for caching efficiency (about 11m accuracy)
            lat_rounded = round(float(latitude), 4)
            lng_rounded = round(float(longitude), 4)
            
            # BigDataCloud reverse geocoding API (free, no API key required)
            url = f"https://api.bigdatacloud.net/data/reverse-geocode-client?latitude={lat_rounded}&longitude={lng_rounded}&localityLanguage=en"
            
            print(f"[GEOCODING] Requesting location for {lat_rounded}, {lng_rounded}")
            
            response = requests.get(url, timeout=5)  # 5 second timeout
            response.raise_for_status()
            
            data = response.json()
            
            # Extract location components
            location_info = {
                "city": data.get("city", ""),
                "locality": data.get("locality", ""),
                "principalSubdivision": data.get("principalSubdivision", ""),  # State/Province
                "countryName": data.get("countryName", ""),
                "countryCode": data.get("countryCode", "")
            }
            
            # Build human-readable address
            address_parts = []
            
            # Add locality (neighborhood/district) if different from city
            if location_info["locality"] and location_info["locality"] != location_info["city"]:
                address_parts.append(location_info["locality"])
            
            # Add city
            if location_info["city"]:
                address_parts.append(location_info["city"])
            
            # Add state/province
            if location_info["principalSubdivision"]:
                address_parts.append(location_info["principalSubdivision"])
            
            # Add country if not US (assume most fleet operations are US-based)
            if location_info["countryCode"] and location_info["countryCode"] != "US":
                address_parts.append(location_info["countryName"])
            
            formatted_address = ", ".join(address_parts) if address_parts else "Unknown Location"
            
            print(f"[GEOCODING] Location resolved: {formatted_address}")
            
            return {
                "formatted_address": formatted_address,
                "components": location_info,
                "success": True
            }
            
        except requests.exceptions.Timeout:
            print(f"[GEOCODING] Timeout for coordinates {latitude}, {longitude}")
            return {
                "formatted_address": f"Near {latitude}, {longitude}",
                "components": {},
                "success": False,
                "error": "timeout"
            }
        except requests.exceptions.RequestException as e:
            print(f"[GEOCODING] API error for {latitude}, {longitude}: {e}")
            return {
                "formatted_address": f"Coordinates {latitude}, {longitude}",
                "components": {},
                "success": False,
                "error": str(e)
            }
        except Exception as e:
            print(f"[GEOCODING] Unexpected error for {latitude}, {longitude}: {e}")
            return {
                "formatted_address": f"Location {latitude}, {longitude}",
                "components": {},
                "success": False,
                "error": str(e)
            }

    def enhance_location_data(self, data_records):
        """Enhance database records with human-readable locations"""
        enhanced_records = []
        
        for record in data_records:
            enhanced_record = record.copy()
            
            # Check if record has GPS coordinates
            lat_field = None
            lng_field = None
            
            # Find latitude field (multiple possible names)
            for field in ['last_ping_lat', 'latitude', 'lat', 'start_latitude', 'end_latitude']:
                if field in record and record[field] is not None:
                    lat_field = field
                    break
            
            # Find longitude field (multiple possible names)  
            for field in ['last_ping_lng', 'longitude', 'lng', 'start_longitude', 'end_longitude']:
                if field in record and record[field] is not None:
                    lng_field = field
                    break
            
            # If we have both coordinates, get the location
            if lat_field and lng_field:
                latitude = record[lat_field]
                longitude = record[lng_field]
                
                # Only geocode if coordinates are valid (not 0,0 or null)
                if latitude and longitude and (latitude != 0 or longitude != 0):
                    location_data = self.reverse_geocode(latitude, longitude)
                    
                    # Add location information to the record
                    enhanced_record["formatted_address"] = location_data["formatted_address"]
                    enhanced_record["location_components"] = location_data["components"]
                    enhanced_record["geocoding_success"] = location_data["success"]
                else:
                    enhanced_record["formatted_address"] = "Location unavailable"
                    enhanced_record["geocoding_success"] = False
            else:
                # No coordinates available
                enhanced_record["formatted_address"] = "GPS coordinates not available"
                enhanced_record["geocoding_success"] = False
            
            enhanced_records.append(enhanced_record)
        
        return enhanced_records
