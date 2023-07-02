import sys, os, traceback


def show_exception_and_exit(exc_type, exc_value, tb):
    traceback.print_exception(exc_type, exc_value, tb)
    input()
    sys.exit(-1)
sys.excepthook = show_exception_and_exit

delimiter = ";"
args = sys.argv[1:]

if len(args) == 0:
	raise Exception("This python program expects a directory as argument. Pass it through the console or per drag and drop.")
elif len(args) > 1:
	raise Exception(f"This python program expects exactly one directory as argument. It got {len(args)}.")

data_dir = args[0]

if not os.path.isdir(data_dir):
	raise Exception(f"'{data_dir}' is not a valid directory.")

file_names = [file_name for file_name in os.listdir(data_dir) if os.path.splitext(file_name)[1] == ".csv"]


distances : list[str] = []
damages_per_weapon : dict[str, list[str]] = {}

i = 0
for file_name in file_names:
	file_path = os.path.join(data_dir, file_name)
	weapon_name = os.path.splitext(file_name)[0]

	with open(file_path, "r") as file:
		data = file.read()

	data_lines = data.splitlines()
	if len(data_lines) != 2:
		raise Exception(f"'{file_path}' must have exactly 2 lines but has {len(data_lines)}.")

	data_distance = data_lines[0].strip(delimiter).split(delimiter)
	data_damage = data_lines[1].strip(delimiter).split(delimiter)

	if i == 0:
		distances = [distance for distance in data_distance]
	elif len(data_distance) != len(distances):
		raise Exception(f"'{file_path}' must have {len(distances)} but has {len(data_distance)} distance values.")

	if len(data_distance) != len(data_damage):
		raise Exception(f"'{file_path}' must have as many distance values as damage values but has {len(data_distance)} and {len(data_damage)}.")

	damages_per_weapon[weapon_name] = [damage for damage in data_damage]

	i += 1


content : str = "m" + delimiter

for distance in distances:
	content += distance + delimiter
content += "\n"

weapon_names = sorted(damages_per_weapon.keys())
for weapon_name in weapon_names:
	damages = damages_per_weapon[weapon_name]
	if len(damages) != len(distances):
		raise Exception("how??")

	content += weapon_name + delimiter
	for damage in damages:
		content += damage + delimiter
	content += "\n"


target_file_name = f"merged_{os.path.basename(data_dir)}.csv"
with open(target_file_name, "x") as target_file:
	target_file.write(content)

input("Completed!")
