import os

delimiter = ";"
original_dir = "original_data"
extrapolated_dir = "extrapolated_data"

if not os.path.exists(original_dir):
	raise Exception(f"Folder '{original_dir}' is missing in '{os.getcwd()}'.")
if not os.path.exists(extrapolated_dir):
	raise Exception(f"Folder '{extrapolated_dir}' is missing in '{os.getcwd()}'.")


file_names = [file_name for file_name in os.listdir(original_dir) if os.path.splitext(file_name)[1] == ".csv"]

for file_name in file_names:
	original_file_path = os.path.join(original_dir, file_name)
	extrapolated_file_path = os.path.join(extrapolated_dir, file_name)

	with open(original_file_path, "r") as original_file:
		original_content = original_file.read()

	original_lines = original_content.splitlines()
	if len(original_lines) != 2:
		raise Exception(f"'{original_file_path}' must have exactly 2 lines but has {len(original_lines)}.")
	
	original_distance = original_lines[0].strip(delimiter).split(delimiter)
	original_damage = original_lines[1].strip(delimiter).split(delimiter)

	if len(original_distance) != len(original_damage):
		raise Exception(f"'{original_file_path}' must have as many distance values as damage values but has {len(original_distance)} and {len(original_damage)}.")
	
	n = len(original_distance)

	extrapolated_distance = [int(distance) for distance in original_distance]
	extrapolated_damage = [int(damage) for damage in original_damage]
	
	previous_damage = 0
	previous_was_extrapolated = False

	for i in range(n):
		if i != 0:
			if extrapolated_distance[i] != extrapolated_distance[i-1] + 1:	# uncontinous distance (e.g. skipped distance 23m)
				raise Exception(f"Uncontinuous distance information for '{original_file_path}'.")

		if extrapolated_damage[i] != 0:
			if previous_was_extrapolated == True and extrapolated_damage[i] != previous_damage:	# unequal damage values before and after missing damage data
				raise Exception(f"Tried to extrapolate between two unequal damage values in '{original_file_path}'.")

			if extrapolated_damage[i] > extrapolated_damage[i-1] and extrapolated_damage[i-1] != 0:	# detected damage increase. should have been decrease or stagnation
				raise Exception(f"Detected damage increase from '{extrapolated_damage[i-1]}' to '{extrapolated_damage[i]}' at {extrapolated_distance[i-1]}m-{extrapolated_distance[i]}m in '{original_file_path}'.")

			previous_was_extrapolated = False
			previous_damage = extrapolated_damage[i]

		else:
			extrapolated_damage[i] = previous_damage
			
	extrapolated_content = ""
	for i in range(n):
		extrapolated_content += str(extrapolated_distance[i]) + delimiter
	extrapolated_content += "\n"
	for i in range(n):
		extrapolated_content += str(extrapolated_damage[i]) + delimiter
	extrapolated_content += "\n"

	with open(extrapolated_file_path, "x") as extrapolated_file:
		extrapolated_file.write(extrapolated_content)

	pass



    