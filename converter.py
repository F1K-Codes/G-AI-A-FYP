import json

with open("farmbot_commands_final_p1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

new_data = []

for entry in data:
    if entry.get("type") != "farmbot_command":
        continue

    action = entry.get("action")
    params = entry.get("parameters", {})

    output_string = None

    # ---- Map each action ----
    if action == "go_to_home":
        output_string = "H_0"

    elif action == "emergency_stop":
        output_string = "e"

    elif action == "reset_estop":
        output_string = "E"

    elif action == "calibrate_axis":
        axis = params.get("axis", "").lower()
        if axis in ("x", "y", "z"):
            output_string = f"H_2 {axis.upper()}"
        else:
            continue   # invalid axis, skip

    elif action == "move_to_absolute":
        # x, y, z are always present in valid move_to_absolute entries
        x = int(params["x"])
        y = int(params["y"])
        z = int(params["z"])
        output_string = f"M {x} {y} {z}"

    elif action == "move_increment":
        direction = params.get("direction")
        if direction == "forward":
            output_string = "w"
        elif direction == "backward":
            output_string = "s"
        elif direction == "left":
            output_string = "a"
        elif direction == "right":
            output_string = "d"
        else:
            continue

    elif action == "mount_tool":
        tool_index = params.get("tool_index")
        if tool_index is None:
            continue          # skip if tool_index missing
        idx = int(tool_index)
        output_string = f"T_{idx}_1"

    elif action == "unmount_tool":
        tool_index = params.get("tool_index")
        if tool_index is None:
            continue          # skip unmount without tool index
        idx = int(tool_index)
        output_string = f"T_{idx}_2"

    elif action == "water_plants":
        all_plants = params.get("all_plants", False)
        output_string = "P_4" if all_plants else "P_5"

    # If no mapping, skip (e.g., toggle_water_pump, toggle_led)
    if output_string is None:
        continue

    new_entry = {
        "instruction": entry["instruction"],
        "input": entry.get("input", ""),
        "output": output_string
    }
    new_data.append(new_entry)bcauiokcba
    

with open("farmbot_commands_final_p2.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, indent=2)

print(f"Created {len(new_data)} entries in farmbot_commands_final_p2.json")