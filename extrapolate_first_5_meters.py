import os, sys, traceback

def show_exception_and_exit(exc_type, exc_value, tb):
    traceback.print_exception(exc_type, exc_value, tb)
    input()
    sys.exit(-1)
sys.excepthook = show_exception_and_exit

delimiter = ";"
extrapolated_dir = "extrapolated_data"
extrapolated_dir_with_5 = "extrapolated_data_with_first_5_meters"

if not os.path.isdir(extrapolated_dir):
	raise Exception(f"Folder '{extrapolated_dir}' is missing in '{os.getcwd()}'.")
if not os.path.isdir(extrapolated_dir_with_5):
	raise Exception(f"Folder '{extrapolated_dir_with_5}' is missing in '{os.getcwd()}'.")

file_names = [file_name for file_name in os.listdir(extrapolated_dir) if os.path.splitext(file_name)[1] == ".csv"]

for file_name in file_names:
	extrapolated_file_path = os.path.join(extrapolated_dir, file_name)
	extrapolated_with_5_file_path = os.path.join(extrapolated_dir_with_5, file_name)

	with open(extrapolated_file_path, "r") as extrapolated_file:
		extrapolated_content = extrapolated_file.read()

	extrapolated_lines = extrapolated_content.splitlines()
	if len(extrapolated_lines) != 2:
		raise Exception(f"'{extrapolated_file_path}' must have exactly 2 lines but has {len(extrapolated_lines)}.")

	extrapolated_distance = extrapolated_lines[0].strip(delimiter).split(delimiter)
	extrapolated_damage = extrapolated_lines[1].strip(delimiter).split(delimiter)

	if len(extrapolated_distance) != len(extrapolated_damage):
		raise Exception(f"'{extrapolated_file_path}' must have as many distance values as damage values but has {len(extrapolated_distance)} and {len(extrapolated_damage)}.")

	n = len(extrapolated_distance)

	extrapolated_with_5_distance = [int(distance) for distance in extrapolated_distance]
	extrapolated_with_5_damage = [int(damage) for damage in extrapolated_damage]
	
	index_5m = extrapolated_with_5_distance.index(5)
	index_6m = extrapolated_with_5_distance.index(6)
	index_7m = extrapolated_with_5_distance.index(7)

	if index_5m != 5 or index_6m != index_5m + 1 or index_7m != index_6m + 1:
		raise Exception(f"Falsly aligned distance values in '{extrapolated_file_path}'.")

	if extrapolated_with_5_damage[index_5m] == extrapolated_with_5_damage[index_6m] == extrapolated_with_5_damage[index_7m]:
		extrapolated_with_5_damage[0] = extrapolated_with_5_damage[index_5m]
		extrapolated_with_5_damage[1] = extrapolated_with_5_damage[index_5m]
		extrapolated_with_5_damage[2] = extrapolated_with_5_damage[index_5m]
		extrapolated_with_5_damage[3] = extrapolated_with_5_damage[index_5m]
		extrapolated_with_5_damage[4] = extrapolated_with_5_damage[index_5m]
	else:
		print(f"Warning: Can't extrapolate first 5 meters for '{extrapolated_file_path}'.")

	extrapolated_with_5_content = ""
	for i in range(n):
		extrapolated_with_5_content += str(extrapolated_with_5_distance[i]) + delimiter
	extrapolated_with_5_content += "\n"
	for i in range(n):
		extrapolated_with_5_content += str(extrapolated_with_5_damage[i]) + delimiter
	extrapolated_with_5_content += "\n"

	with open(extrapolated_with_5_file_path, "x") as extrapolated_with_5_file:
		extrapolated_with_5_file.write(extrapolated_with_5_content)

	pass

input("Completed!")