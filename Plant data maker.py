import json
import time
import openai
from pathlib import Path

# ---------- CONFIGURATION ----------
PLANT_LIST_FILE = "Training Data/list_of_plants.txt"
OUTPUT_RAW_FILE = "Training Data/plant_diagnostics_raw.json"
PAIRS_PER_PLANT = 8
TEMPERATURE = 0.5                       # lower = more deterministic
MODEL = "gpt-4o-mini"

Path(OUTPUT_RAW_FILE).parent.mkdir(parents=True, exist_ok=True)

# ---------- CORRECTED SYSTEM PROMPT ----------
SYSTEM_PROMPT = (
    "You are a knowledgeable plant pathologist and gardening assistant.\n"
    f"For the given plant name, create exactly {PAIRS_PER_PLANT} diagnostic question-answer pairs.\n"
    "Each pair must have an 'instruction' (a realistic user question about a common disease, pest, or care problem for that plant) "
    "and an 'output' (a concise, helpful answer, 2-4 sentences).\n"
    "Your response must be a valid JSON object with a single key 'pairs' containing an array of those objects.\n"
    "Example response:\n"
    '{\n'
    '  "pairs": [\n'
    '    {\n'
    '      "instruction": "Why are the lower leaves of my tomato plant turning yellow?",\n'
    '      "output": "Lower leaves yellowing often indicates nitrogen deficiency or overwatering..."\n'
    '    },\n'
    '    ... (exactly 5 objects)\n'
    '  ]\n'
    '}\n'
    "Do not include any other text, markdown, or explanation. Only the JSON object."
)

# ---------- LOAD PLANT NAMES ----------
with open(PLANT_LIST_FILE, "r", encoding="utf-8") as f:
    plant_names = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(plant_names)} plant names.")

# ---------- INITIALIZE OPENAI CLIENT ----------
client = openai.OpenAI()

# ---------- COLLECT ALL PAIRS ----------
all_pairs = []
failed_plants = []

for i, plant in enumerate(plant_names, 1):
    print(f"[{i}/{len(plant_names)}] Processing: {plant}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": plant}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content

        # ---- DEBUG (optional) ----
        print(f"\n--- RAW RESPONSE for {plant} ---")
        print(content)
        print("--- END RAW ---\n")

        parsed = json.loads(content)

        # The expected structure is a dict with key "pairs"
        if isinstance(parsed, dict) and "pairs" in parsed and isinstance(parsed["pairs"], list):
            pairs = parsed["pairs"]
        else:
            # Fallback: look for any list
            pairs = None
            if isinstance(parsed, list):
                pairs = parsed
            elif isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        pairs = v
                        break
            if not pairs:
                raise ValueError("Could not find 'pairs' array in response.")

        # Clean and validate
        clean_pairs = []
        for p in pairs:
            if isinstance(p, dict) and "instruction" in p and "output" in p:
                clean_pairs.append({
                    "instruction": p["instruction"].strip(),
                    "output": p["output"].strip()
                })
        if len(clean_pairs) != PAIRS_PER_PLANT:
            print(f"  ⚠️ Expected {PAIRS_PER_PLANT} pairs, got {len(clean_pairs)}")
        all_pairs.extend(clean_pairs)
        print(f"  ✅ Successfully added {len(clean_pairs)} pairs.")

    except Exception as e:
        print(f"  ❌ ERROR for {plant}: {e}")
        failed_plants.append(plant)

    time.sleep(0.3)   # avoid rate limits
    
    

# ---------- SAVE RAW DATA ----------
with open(OUTPUT_RAW_FILE, "w", encoding="utf-8") as f:
    json.dump(all_pairs, f, indent=2, ensure_ascii=False)

print(f"\nTotal generated pairs: {len(all_pairs)}")
print(f"Failed plants: {failed_plants if failed_plants else 'None'}")
print(f"Raw data saved to {OUTPUT_RAW_FILE}")