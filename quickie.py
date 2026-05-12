import json

INPUT_FILE = "Training Data/plant_diagnostics.json"
OUTPUT_FILE = "Training Data/plant_diagnostics_final.json"   

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    item["type"] = "plant_care"

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Updated {len(data)} entries. Saved to {OUTPUT_FILE}")