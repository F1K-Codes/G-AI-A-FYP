import json
from pathlib import Path
from difflib import get_close_matches

# ---------- CONFIGURATION ----------
RAW_FILE = "Training Data/plant_diagnostics_raw.json"
FINAL_FILE = "Training Data/plant_diagnostics.json"
PLANT_LIST_FILE = "Training Data/list_of_plants.txt"


# ---------- LOAD DATA ----------
with open(RAW_FILE, "r", encoding="utf-8") as f:
    pairs = json.load(f)

# Load all known plant names (lowercase)
with open(PLANT_LIST_FILE, "r", encoding="utf-8") as f:
    known_plants = [line.strip().lower() for line in f if line.strip()]

print(f"Loaded {len(pairs)} raw pairs.")

# ---------- CLEANING ----------
seen_instructions = set()
cleaned = []

for pair in pairs:
    instr = pair["instruction"].strip()
    output = pair["output"].strip()

    # 1. Skip empty or very short answers
    if len(output) < 20:
        continue

    # 2. Remove exact duplicate instructions
    if instr.lower() in seen_instructions:
        continue
    seen_instructions.add(instr.lower())

    # 3. Check that some plant name is mentioned (fuzzy)
    instr_lower = instr.lower()
    plant_found = any(plant in instr_lower for plant in known_plants)
    if not plant_found:
        # fuzzy fallback
        matches = get_close_matches(instr_lower, known_plants, n=1, cutoff=0.6)
        if not matches:
            print(f"Warning: No plant found in instruction: {instr[:60]}...")
            # Still keep it, but you might remove it if you want strictness
        else:
            plant_found = True

    # 4. Add plant_name field for later use (optional)
    # We'll extract the first matching plant name
    for plant in known_plants:
        if plant in instr_lower:
            pair["plant_name"] = plant
            break
    if "plant_name" not in pair:
        pair["plant_name"] = "unknown"

    cleaned.append({"instruction": instr, "output": output})

print(f"After cleaning: {len(cleaned)} pairs.")

# ---------- SAVE FINAL FILE ----------
Path(FINAL_FILE).parent.mkdir(parents=True, exist_ok=True)
with open(FINAL_FILE, "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=2, ensure_ascii=False)

print(f"Final dataset saved to {FINAL_FILE}")