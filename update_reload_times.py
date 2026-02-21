import json
import os
from pathlib import Path

# Get all JSON files in the weapons directory
weapons_dir = Path("weapons")
json_files = list(weapons_dir.glob("*.json"))

print(f"Found {len(json_files)} weapon files")

# Process each file
for json_file in json_files:
    try:
        # Read the JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if reload_times exists and is a list
        if "reload_times" in data and isinstance(data["reload_times"], list):
            reload_times = data["reload_times"]
            
            # Set index 1 to null if it exists
            if len(reload_times) > 1:
                reload_times[1] = None
            
            # Set index 2 to null if it exists
            if len(reload_times) > 2:
                reload_times[2] = None
            
            # Write the updated JSON back to the file
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            
            print(f"✓ Updated {json_file.name}")
        else:
            print(f"⚠ Skipped {json_file.name} - no reload_times array found")
    
    except Exception as e:
        print(f"✗ Error processing {json_file.name}: {e}")

print("\nAll files processed!")
