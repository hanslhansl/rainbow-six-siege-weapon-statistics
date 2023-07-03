

###################################################
# settings
###################################################

# the delimiter used in the csv files
csv_delimiter = ";"

# the file containing all weapon names
weapon_names_file_name = "weapon_names.csv"
# the file containing the type of every weapon
weapon_types_file_name = "weapon_types.csv"
# the file containing the fire rate of every weapon
weapon_firerates_file_name = "weapon_firerates.csv"
# the file containing the weapons each operator has access to
operator_weapons_file_name = "operator_weapons.csv"

# the directory containing the weapon damage files
damage_data_dir = "damage_data"

# the distance the weapon damage stats start at (usually either 0 or 1)
first_distance = 0
# the distance the weapon damage stats end at (usually 40 because the of the Shooting Range limit)
last_distance = 40 

###################################################
# settings end
# don't edit from here on
###################################################


#imports
import os, sys, traceback

# install exception catcher
def show_exception_and_exit(exc_type, exc_value, tb):
    traceback.print_exception(exc_type, exc_value, tb)
    input()
    sys.exit(-1)
sys.excepthook = show_exception_and_exit


# check if the settings are correct
if not os.path.isfile(weapon_names_file_name):
	raise Exception(f"'{weapon_names_file_name}' is not a valid file path.")
if not os.path.isfile(weapon_types_file_name):
	raise Exception(f"'{weapon_types_file_name}' is not a valid file path.")
if not os.path.isfile(weapon_firerates_file_name):
	raise Exception(f"'{weapon_firerates_file_name}' is not a valid file path.")
if not os.path.isfile(operator_weapons_file_name):
	raise Exception(f"'{operator_weapons_file_name}' is not a valid file path.")

if not os.path.isdir(damage_data_dir):
	raise Exception(f"'{damage_data_dir}' is not a valid directory.")

if not 0 <= first_distance:
	raise Exception(f"'first_distance' must be >=0 but is {first_distance}.")
if not first_distance <= last_distance:
	raise Exception(f"'last_distance' must be >='first_distance'={first_distance} but is {last_distance}.")




class Weapon:
	types : list[str] = []
	operators : list[str] = []
	distances : list[int] = [i for i in range(first_distance, last_distance+1)]

	def __init__(self, name : str):
		self.name = name

		self.typeIndex : int
		self.firerate : int
		self.operatorIndices : list[int]
		self.damages : list[int]


def get_weapon_types(weapons : dict[str, Weapon], file_name : str):
	with open(file_name, "r") as file:
		weapon_types_lines = file.read().splitlines()

	if not len(weapon_types_lines) >= 1:
		raise Exception(f"{file_name} must have at least one line.")

	header_line = weapon_types_lines[0]
	if header_line[0] != csv_delimiter:
		raise Exception(f"{file_name} is of wrong format.")
	Weapon.types = header_line.strip(csv_delimiter).split(csv_delimiter)

	content_splitted = [line.split(csv_delimiter) for line in weapon_types_lines[1:]]
	content_dict = {splitted[0] : splitted[1:] for splitted in content_splitted}

	for weapon_name in weapons:
		try:
			type_list = content_dict[weapon_name]
		except KeyError as err:
			print(f"'{file_name}' is missing '{weapon_name}'.")
			raise err

		if len(type_list) > len(Weapon.types):
			raise Exception(f"Too many columns for '{weapon_name}' in '{file_name}'")

		try:
			index = type_list.index("x")
		except ValueError as err:
			print(f"No weapon type for '{weapon_name}' in '{file_name}'.")
			raise err

		weapons[weapon_name].typeIndex = index

	return

def get_weapon_firerates(weapons : dict[str, Weapon], file_name : str):
	for weapon_name in weapons:
		weapons[weapon_name].firerate = -1

	"""with open(file_name, "r") as file:
		weapon_firerates_lines = file.read().splitlines()

	print(weapon_firerates_lines)

	content_splitted = [line.split(csv_delimiter) for line in weapon_firerates_lines]
	content_dict = {splitted[0] : splitted[1] for splitted in content_splitted}

	for weapon_name in weapons:
		try:
			firerate = content_dict[weapon_name]
		except KeyError as err:
			print(f"'{file_name}' is missing '{weapon_name}'.")
			raise err

		try:
			firerate = int(firerate)
		except ValueError as err:
			print(f"Error in '{file_name}'.")
			raise err

		weapons[weapon_name].firerate = firerate"""

	return

def get_operator_weapons(weapons : dict[str, Weapon], file_name : str):
	with open(file_name, "r") as file:
		operator_weapons_lines = file.read().splitlines()

	operator_weapons_entries = [x.split(csv_delimiter) for x in operator_weapons_lines]

	all_operators : list[str] = []
	for line in operator_weapons_lines:
		operator = line.split(csv_delimiter)[0]
		if operator != "":
			all_operators.append(operator)
	Weapon.operators = sorted(all_operators)

	weapon_operatorIndex_dict : dict[str, list[int]] = {}
	current_operator = ""
	for line in operator_weapons_lines:
		try:
			operator, weapon = line.split(csv_delimiter)
		except ValueError as err:
			print(f"Error in '{file_name}'.")
			raise err

		if operator == "":
			if current_operator == "":
				raise Exception(f"{file_name} is of wrong format.")
			else:
				operator = current_operator
		else:
			current_operator = operator

		operatorIndex = Weapon.operators.index(operator)

		if weapon in weapon_operatorIndex_dict:
			weapon_operatorIndex_dict[weapon].append(operatorIndex)
		else:
			weapon_operatorIndex_dict[weapon] = [operatorIndex]

	for weapon_name in weapons:
		try:
			operatorIndices = weapon_operatorIndex_dict[weapon_name]
		except KeyError as err:
			print(f"'{file_name}' is missing '{weapon_name}'.")
			raise err

		weapons[weapon_name].operatorIndices = operatorIndices
		
	return

def get_damages_per_weapon(weapons : dict[str, Weapon], data_dir : str, distances : list[int]):
	N = len(distances)

	for weapon_name in weapons:
		data_file_name = weapon_name + ".csv"
		data_file_path = os.path.join(data_dir, data_file_name)

		if not os.path.isfile(data_file_path):
			raise Exception(f"Directory '{data_dir}' is missing file '{weapon_name}.csv'.")

		with open(data_file_path, "r") as data_file:
			data = data_file.read()

		lines = data.splitlines()
		if len(lines) != 2:
			raise Exception(f"'{data_file_path}' must have exactly 2 lines but has {len(lines)}.")
	
		distanceStrings = lines[0].strip(csv_delimiter).split(csv_delimiter)
		if distances != [int(distance) for distance in distanceStrings]:
			raise Exception(f"The distance values in '{data_file_path}' are not correct.")

		damageStrings = lines[1].strip(csv_delimiter).split(csv_delimiter)
		if len(damageStrings) != N:
			raise Exception(f"'{data_file_path}' must have {N} damage values but has {len(damageStrings)}.")
	
		try:
			damages = [int(damage) for damage in damageStrings]
		except ValueError as err:
			print(f"Error in '{data_file_path}'.")
			raise err
	
		# interpolate gaps. damages will be continuous in [5;40]
		previous_damage = 0
		previous_was_interpolated = False
		for i in range(N):
			if damages[i] != 0:
				if previous_was_interpolated == True and damages[i] != previous_damage:	# unequal damage values before and after missing damage data
					raise Exception(f"Tried to interpolate between two unequal damage values in '{data_file_path}'.")

				if damages[i] > damages[i-1] and damages[i-1] != 0:	# detected damage increase. should have been decrease or stagnation
					raise Exception(f"Detected damage increase from '{damages[i-1]}' to '{damages[i]}' at {distances[i-1]}m-{distances[i]}m in '{data_file_path}'.")

				previous_was_interpolated = False
				previous_damage = damages[i]

			else:
				damages[i] = previous_damage
			
		# extrapolate first 5 meters. damages will be continuous in [0;4]
		first_nonzero_index = next((i for i, damage in enumerate(damages) if damage != 0), None)

		if first_nonzero_index == 0:
			pass	# no extrapolation needed
		elif first_nonzero_index == None:
			raise Exception(f"This exception should not be triggerd. '{data_file_path}' has to be corrupted in a strange way.")
		else:

			if damages[first_nonzero_index] == damages[first_nonzero_index+1] == damages[first_nonzero_index+2]:
				for i in range(first_nonzero_index):
					damages[i] = damages[first_nonzero_index]
			else:
				print(f"Warning: Can't extrapolate first {first_nonzero_index} meters for '{data_file_path}'.")

		# write this weapons stats to the weapons dict
		weapons[weapon_name].damages = damages

		pass

	return

def main():
	
	# get all distances
	distances = [i for i in range(first_distance, last_distance+1)]

	# get all weapon names
	with open(weapon_names_file_name, "r") as file:
		weapon_names = file.read().splitlines()

	weapons : dict[str, Weapon]= {}
	for weapon_name in weapon_names:
		if weapon_name.startswith("#"):
			print(f"Warning: Excluding weapon '{weapon_name}' because of #.")
		else:
			weapons[weapon_name] = Weapon(weapon_name)


	# get all weapon types
	get_weapon_types(weapons, weapon_types_file_name)

	# get all weapon fire rates
	get_weapon_firerates(weapons, weapon_firerates_file_name)

	# get all operator weapons
	get_operator_weapons(weapons, operator_weapons_file_name)

	# get all weapon damages
	get_damages_per_weapon(weapons, damage_data_dir, distances)


	"""# combine the weapon stats into a single string
	content : str = "m" + csv_delimiter

	for distance in distances:
		content += str(distance) + csv_delimiter
	content += "\n"

	weapon_names = sorted(damages_per_weapon.keys())
	for weapon_name in weapon_names:
		content += weapon_name + csv_delimiter
	
		if len(damages_per_weapon[weapon_name]) != N:
			raise Exception("how??")
		for damage in damages_per_weapon[weapon_name]:
			content += str(damage) + csv_delimiter

		content += "\n"


	# write the string to file
	target_file_name = f"merged_{os.path.basename(damage_data_dir)}.csv"
	with open(target_file_name, "x") as target_file:
		target_file.write(content)"""

	return




main()

input("Completed!")