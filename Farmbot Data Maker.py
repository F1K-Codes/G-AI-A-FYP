import json
import random
from typing import Dict, List, Any, Union, Callable, Tuple


# Configuration
OUTPUT_FILE = "farmbot_commands_final_p1.json"
TOTAL_EXAMPLES = 2000        # Base number per action type

# Bed dimensions (in mm) – adjust to your FarmBot setup
BED_X_MIN, BED_X_MAX = 0, 5700
BED_Y_MIN, BED_Y_MAX = 0, 20000
Z_SAFE = 0
Z_SOIL = -20
Z_DEEP = -50

TOOL_INDICES = [1, 2, 3]

FIXED_COORDS = [
    {"x": 100, "y": 500, "z": 0},
    {"x": 500, "y": 2000, "z": -20},
    {"x": 1200, "y": 4500, "z": -50},
    {"x": 2500, "y": 8000, "z": 0},
    {"x": 3200, "y": 12000, "z": -20},
    {"x": 4500, "y": 15000, "z": -50},
    {"x": 5200, "y": 18000, "z": 0},
    {"x": 600, "y": 3000, "z": -20},
    {"x": 1800, "y": 7000, "z": -50},
    {"x": 3000, "y": 10000, "z": 0},
    {"x": 4000, "y": 14000, "z": -20},
    {"x": 5000, "y": 17000, "z": -50},
    {"x": 200, "y": 900, "z": 0},
    {"x": 1500, "y": 3500, "z": -20},
    {"x": 3500, "y": 11000, "z": -50},
]

def random_coord() -> Dict[str, int]:
    return random.choice(FIXED_COORDS)

def random_duration() -> int:
    return random.choice([5, 15, 30])

def random_axis() -> str:
    return random.choice(["x", "y", "z"])

# ----------------------------------------------------------------------
# Template definitions: each is a tuple (template string, param_func)
# ----------------------------------------------------------------------
TEMPLATES = {
    "move_to_absolute": [
        ("move to x={x} y={y} z={z}", random_coord),
        ("go to coordinates {x} {y} {z}", random_coord),
        ("set position to x={x} y={y} z={z}", random_coord),
        ("drive to X={x} Y={y} Z={z}", random_coord),
        ("go to location {x} {y} at depth {z}", random_coord),
        ("move the toolhead to ({x}, {y}, {z})", random_coord),
        ("position the arm at {x} mm across and {y} mm forward", random_coord),
        ("navigate to x={x} y={y} z={z}", random_coord),
        ("send gantry to {x} {y} {z}", random_coord),
        ("I want to go to X={x} Y={y} Z={z}", random_coord),
        ("please move to {x} {y} {z}", random_coord),
        ("can you go to {x} {y} {z}?", random_coord),
        ("take the tool to {x}, {y}, {z}", random_coord),
        ("set coordinates to {x} {y} {z}", random_coord),
        ("go to point {x} {y} {z}", random_coord),
        ("move to absolute position {x} {y} {z}", random_coord),
        ("go to x equals {x}, y equals {y}, z equals {z}", random_coord),
        ("position at {x} {y} {z}", random_coord),
        ("relocate to {x} {y} {z}", random_coord),
        ("travel to coordinates {x} {y} {z}", random_coord),
    ],
    "move_increment": [
    ("move forward", lambda: {"direction": "forward"}),
    ("move backward", lambda: {"direction": "backward"}),
    ("move left", lambda: {"direction": "left"}),
    ("move right", lambda: {"direction": "right"}),
    ("go forward", lambda: {"direction": "forward"}),
    ("go backward", lambda: {"direction": "backward"}),
    ("go left", lambda: {"direction": "left"}),
    ("go right", lambda: {"direction": "right"}),
    ("step forward", lambda: {"direction": "forward"}),
    ("step backward", lambda: {"direction": "backward"}),
    ("step left", lambda: {"direction": "left"}),
    ("step right", lambda: {"direction": "right"}),
],
    "go_to_home": [
        ("go home", lambda: {}),
        ("return to home", lambda: {}),
        ("move to home position", lambda: {}),
        ("go back to the starting point", lambda: {}),
        ("return to base", lambda: {}),
        ("home the gantry", lambda: {}),
        ("send it home", lambda: {}),
        ("go to home location", lambda: {}),
        ("please go home", lambda: {}),
        ("I want you to go home", lambda: {}),
        ("can you go home?", lambda: {}),
        ("head home", lambda: {}),
        ("back to home", lambda: {}),
        ("take me home", lambda: {}),  # metaphorical
        ("go to the home position", lambda: {}),
        ("reset position to home", lambda: {}),
        ("move back to home", lambda: {}),
        ("go to zero zero zero", lambda: {}),
        ("go to origin", lambda: {}),
        ("go to the start", lambda: {}),
    ],
    "emergency_stop": [
        ("stop", lambda: {}),
        ("emergency stop", lambda: {}),
        ("halt", lambda: {}),
        ("stop everything", lambda: {}),
        ("stop right now", lambda: {}),
        ("cancel", lambda: {}),
        ("abort", lambda: {}),
        ("abort mission", lambda: {}),
        ("e-stop", lambda: {}),
        ("make it stop", lambda: {}),
        ("oh no stop", lambda: {}),
        ("help stop", lambda: {}),
        ("FarmBot stop", lambda: {}),
        ("stop the robot", lambda: {}),
        ("cease movement", lambda: {}),
        ("freeze", lambda: {}),
        ("don't move", lambda: {}),
        ("stop immediately", lambda: {}),
        ("kill the motors", lambda: {}),
        ("hit the brakes", lambda: {}),
    ],
    "reset_estop": [
        ("reset", lambda: {}),
        ("reset estop", lambda: {}),
        ("clear emergency stop", lambda: {}),
        ("resume", lambda: {}),
        ("continue", lambda: {}),
        ("unlock", lambda: {}),
        ("release estop", lambda: {}),
        ("clear estop", lambda: {}),
        ("enable motors", lambda: {}),
        ("re-enable", lambda: {}),
        ("restart after stop", lambda: {}),
        ("clear the stop", lambda: {}),
        ("reset the emergency", lambda: {}),
        ("you can move again", lambda: {}),
        ("resume operation", lambda: {}),
        ("continue after estop", lambda: {}),
        ("reset safety", lambda: {}),
        ("clear the halt", lambda: {}),
        ("estop off", lambda: {}),
        ("release the brake", lambda: {}),
    ],
    "calibrate_axis": [
        ("calibrate {axis}", lambda: {"axis": random_axis()}),
        ("calibrate the {axis} axis", lambda: {"axis": random_axis()}),
        ("run calibration for {axis}", lambda: {"axis": random_axis()}),
        ("find {axis} home", lambda: {"axis": random_axis()}),
        ("calibrate axis {axis}", lambda: {"axis": random_axis()}),
        ("please calibrate {axis}", lambda: {"axis": random_axis()}),
        ("I need to calibrate {axis}", lambda: {"axis": random_axis()}),
        ("can you calibrate {axis}?", lambda: {"axis": random_axis()}),
        ("perform {axis} calibration", lambda: {"axis": random_axis()}),
        ("set {axis} zero", lambda: {"axis": random_axis()}),
        ("zero the {axis} axis", lambda: {"axis": random_axis()}),
        ("home the {axis} axis", lambda: {"axis": random_axis()}),
        ("calibrate just {axis}", lambda: {"axis": random_axis()}),
        ("do a calibration on {axis}", lambda: {"axis": random_axis()}),
        ("run the {axis} axis calibration", lambda: {"axis": random_axis()}),
        ("find the {axis} limit", lambda: {"axis": random_axis()}),
        ("set {axis} home position", lambda: {"axis": random_axis()}),
        ("align {axis}", lambda: {"axis": random_axis()}),
        ("calibrate {axis} motor", lambda: {"axis": random_axis()}),
        ("calibrate {axis} stepper", lambda: {"axis": random_axis()}),
    ],
    "mount_tool": [
        ("mount tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("attach tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("pick up tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("grab tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("use tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("switch to tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("change to tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("load tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("I need tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("can you mount tool {tool_index}?", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("please attach tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("put on tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("get tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("activate tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("select tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("mount the number {tool_index} tool", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("attach tool number {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("go to tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("pick tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("fetch tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
    ],
    "unmount_tool": [
        ("unmount tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("detach tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("drop tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("put tool {tool_index} back", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("release tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("stow tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("return tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("dismount tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("put away tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("let go of tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("unload tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("eject tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("remove tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("take off tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("park tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("store tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("disengage tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("detach the current tool {tool_index}", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("can you unmount tool {tool_index}?", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
        ("please put tool {tool_index} back", lambda: {"tool_index": random.choice(TOOL_INDICES)}),
],
    "water_plants": [
        ("water the plants", lambda: {}),
        ("water the garden", lambda: {}),
        ("irrigate now", lambda: {}),
        ("give the plants a drink", lambda: {}),
        ("water for {seconds} seconds", lambda: {"seconds": random_duration()}),
        ("irrigate for {seconds} seconds", lambda: {"seconds": random_duration()}),
        ("water the bed for {seconds} sec", lambda: {"seconds": random_duration()}),
        ("run watering for {seconds} seconds", lambda: {"seconds": random_duration()}),
        ("start watering sequence", lambda: {}),
        ("moisten the soil", lambda: {}),
        ("water all plants", lambda: {}),
        ("give a quick water", lambda: {"seconds": 5}),
        ("soak the bed", lambda: {"seconds": 60}),
        ("just a sprinkle", lambda: {"seconds": 5}),
        ("water based on moisture", lambda: {}),  # default P_5
        ("water only dry plants", lambda: {}),
        ("water everything", lambda: {"all_plants": True}),
        ("water regardless of moisture", lambda: {"all_plants": True}),
        ("do a full watering", lambda: {}),
        ("water the crops", lambda: {}),
    ],
}

# ----------------------------------------------------------------------
# Entry creation (same format as your existing dataset)
# ----------------------------------------------------------------------
def create_entry(action: str, template_str: str, param_func: Callable) -> Dict[str, Any]:
    params = param_func()
    instruction = template_str
    for key, value in params.items():
        instruction = instruction.replace("{" + key + "}", str(value))
    output = {
        "action": action,
        "parameters": params
    }
    return {
        "instruction": instruction,
        "input": "",
        "output": json.dumps(output),
        "action": action,
        "parameters": params,
        "type": "farmbot_command"
    }

# ----------------------------------------------------------------------
# Main generation
# ----------------------------------------------------------------------
def main():
    all_entries = []

    # Calculate how many per action to reach ~4000 total
    num_actions = len(TEMPLATES)
    base_per_action = TOTAL_EXAMPLES // num_actions
    remainder = TOTAL_EXAMPLES % num_actions

    for idx, (action, templates) in enumerate(TEMPLATES.items()):
        count = base_per_action + (1 if idx < remainder else 0)
        for _ in range(count):
            template_str, param_func = random.choice(templates)
            entry = create_entry(action, template_str, param_func)
            all_entries.append(entry)


    random.shuffle(all_entries)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_entries, f, indent=2)

    print(f"Generated {len(all_entries)} training examples.")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()