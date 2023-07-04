

###################################################
# settings
###################################################

# the delimiter used in the csv files
csv_delimiter = ";"

# the file containing all weapon names
weapon_names_file_name = "weapon_names.json"
# the file containing the type of every weapon
weapon_types_file_name = "weapon_types.json"
# the file containing the fire rate of every weapon
weapon_firerates_file_name = "weapon_firerates.json"
# the file containing the weapons each operator has access to
operator_weapons_file_name = "operator_weapons.json"
# the file containing the pellet count for every weapon
weapon_pellet_counts_file_name = "weapon_pellet_counts.json"
# the file containing the ads time for every weapon
weapon_ads_times_file_name = "weapon_ads_times.json"
# the file containing the reload times for every weapon
weapon_reload_times_file_name = "weapon_reload_times.json"

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
import os, sys, traceback, numpy as np, json, typing

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
if not os.path.isfile(weapon_pellet_counts_file_name):
	raise Exception(f"'{weapon_pellet_counts_file_name}' is not a valid file path.")
if not os.path.isfile(weapon_ads_times_file_name):
	raise Exception(f"'{weapon_ads_times_file_name}' is not a valid file path.")
if not os.path.isfile(weapon_reload_times_file_name):
	raise Exception(f"'{weapon_reload_times_file_name}' is not a valid file path.")


if not os.path.isdir(damage_data_dir):
	raise Exception(f"'{damage_data_dir}' is not a valid directory.")

if not 0 <= first_distance:
	raise Exception(f"'first_distance' must be >=0 but is {first_distance}.")
if not first_distance <= last_distance:
	raise Exception(f"'last_distance' must be >='first_distance'={first_distance} but is {last_distance}.")


class Weapon:
	types : tuple[str,...]
	operators : tuple[str,...]
	distances = np.array([i for i in range(first_distance, last_distance+1)], np.int32)

	def __init__(self, name : str):
		self.name : str = name

		self.typeIndex : int
		self.operatorIndices : tuple[int,...]

		self.firerate : int	#rpm
		self.damages : np.ndarray
		self.reloadTime : tuple[float, float]
		self.adsTime : float
		self.pelletCount : int
		

	def getRPM(self):
		return self.firerate
	def getRPS(self):
		return float(self.firerate) / 60.
	def getRPMS(self):
		return float(self.firerate) / 60000.

	def getDPS(self):
		return self.damages * self.getRPS()

	def getSTDOK(self, hp : int):
		return np.ceil(hp / self.damages).astype(np.int32)

	def getTTDOK(self, hp : int):
		return self.getSTDOK(hp) / self.getRPS()

def deserialize_json(file_name : str):
	with open(file_name, "r") as file:
		try:
			content = json.load(file)
		except json.JSONDecodeError:
			raise Exception(f"The json deserialization of file '{file_name}' failed.")
	return content

def get_weapon_types(weapons : dict[str, Weapon], file_name : str):
	json_content = deserialize_json(file_name)

	if type(json_content) != dict:
		raise Exception(f"File '{file_name}' doesn't deserialize to a weapon type list and weapon type indices.")

	try:
		weapon_type_list = json_content["types"]
	except KeyError:
		raise Exception(f"File '{file_name}' is missing the weapon type list.") from None
	if type(weapon_type_list) != list:
		raise Exception(f"File '{file_name}' is missing the weapon type list.")
	if not all(isinstance(name, str) for name in weapon_type_list):
		raise Exception(f"The weapon type list in file '{file_name}' doesn't deserialize to a list of strings.")
	weapon_type_list = typing.cast(list[str], weapon_type_list)

	try:
		weapon_type_indices = json_content["indices"]
	except KeyError:
		raise Exception(f"File '{file_name}' is missing the weapon type indices.") from None
	if type(weapon_type_indices) != dict:
		raise Exception(f"File '{file_name}' is missing the weapon type indices.")
	if not all(isinstance(weapon, str) for weapon in weapon_type_indices):
		raise Exception(f"The weapons in file '{file_name}' don't deserialize to strings.")
	if not all(isinstance(type_index, int) for type_index in weapon_type_indices.values()):
		raise Exception(f"The weapons type indices in file '{file_name}' don't deserialize integers.")
	weapon_type_indices = typing.cast(dict[str, int], weapon_type_indices)

	Weapon.types = tuple(weapon_type_list)

	for weapon_name in weapons:
		try:
			type_index = weapon_type_indices[weapon_name]
		except KeyError:
			raise Exception(f"File '{file_name}' is missing weapon '{weapon_name}'.") from None

		if not 0 <= type_index < len(Weapon.types):
			raise Exception(f"Weapon type index for weapon '{weapon_name}' is out of bounds in file '{file_name}'.")

		weapons[weapon_name].typeIndex = type_index

	for fake_weapon_name in weapon_type_indices.keys() - weapons.keys():
		print(f"Warning: Weapon '{fake_weapon_name}' found in file '{file_name}' is not an actual weapon.")

	return

def get_weapon_firerates(weapons : dict[str, Weapon], file_name : str):
	json_content = deserialize_json(file_name)
	if type(json_content) != dict:
		raise Exception(f"File '{file_name}' doesn't deserialize to a dict of weapons and fire rates.")

	if not all(isinstance(weapon, str) for weapon in json_content):
		raise Exception(f"The weapons in file '{file_name}' don't deserialize to strings.")
	if not all(isinstance(firerate, int) for firerate in json_content.values()):
		raise Exception(f"The fire rates in file '{file_name}' don't deserialize to integers.")
	weapon_firerates = typing.cast(dict[str, int], json_content)
	
	for weapon_name in weapons:
		try:
			firerate = weapon_firerates[weapon_name]
		except KeyError:
			raise Exception(f"File '{file_name}' is missing weapon '{weapon_name}'.") from None

		weapons[weapon_name].firerate = firerate

	for fake_weapon_name in weapon_firerates.keys() - weapons.keys():
		print(f"Warning: Weapon '{fake_weapon_name}' found in file '{file_name}' is not an actual weapon.")

	return

def get_weapon_pellet_counts(weapons : dict[str, Weapon], file_name : str):
	json_content = deserialize_json(file_name)
	if type(json_content) != dict:
		raise Exception(f"File '{file_name}' doesn't deserialize to a dict of weapons and pellet counts.")

	if not all(isinstance(weapon, str) for weapon in json_content):
		raise Exception(f"The weapons in file '{file_name}' don't deserialize to strings.")
	if not all(isinstance(pellet_count, int) for pellet_count in json_content.values()):
		raise Exception(f"The pellet counts in file '{file_name}' don't deserialize to integers.")
	weapon_pellet_counts = typing.cast(dict[str, int], json_content)

	for weapon_name in weapons:
		try:
			pelletCount = weapon_pellet_counts[weapon_name]
		except KeyError:
			raise Exception(f"File '{file_name}' is missing weapon '{weapon_name}'.") from None

		weapons[weapon_name].pelletCount = pelletCount

	for fake_weapon_name in weapon_pellet_counts.keys() - weapons.keys():
		print(f"Warning: Weapon '{fake_weapon_name}' found in file '{file_name}' is not an actual weapon.")

	return

def get_weapon_ads_times(weapons : dict[str, Weapon], file_name : str):
	json_content = deserialize_json(file_name)
	if type(json_content) != dict:
		raise Exception(f"File '{file_name}' doesn't deserialize to a dict of weapons and ads times.")

	if not all(isinstance(weapon, str) for weapon in json_content):
		raise Exception(f"The weapons in file '{file_name}' don't deserialize to strings.")
	if not all(isinstance(ads_time, float) for ads_time in json_content.values()):
		raise Exception(f"The ads times in file '{file_name}' don't deserialize to floats.")
	weapon_ads_times = typing.cast(dict[str, float], json_content)

	for weapon_name in weapons:
		try:
			ads_time = weapon_ads_times[weapon_name]
		except KeyError:
			raise Exception(f"File '{file_name}' is missing weapon '{weapon_name}'.") from None

		weapons[weapon_name].adsTime = ads_time

	for fake_weapon_name in weapon_ads_times.keys() - weapons.keys():
		print(f"Warning: Weapon '{fake_weapon_name}' found in file '{file_name}' is not an actual weapon.")

	return

def get_weapon_reload_times(weapons : dict[str, Weapon], file_name : str):
	json_content = deserialize_json(file_name)
	if type(json_content) != dict:
		raise Exception(f"File '{file_name}' doesn't deserialize to a dict of weapons and ads times.")

	if not all(isinstance(weapon, str) for weapon in json_content):
		raise Exception(f"The weapons and in file '{file_name}' don't deserialize to strings.")
	if not all(isinstance(reload_times, list) for reload_times in json_content.values()):
		raise Exception(f"The reload times in file '{file_name}' don't deserialize to lists.")
	if not all(len(reload_times) == 2 and all(isinstance(reload_time, float) for reload_time in reload_times) for reload_times in json_content.values()):
		raise Exception(f"The reload times in file '{file_name}' don't deserialize to lists of two floats.")
	json_content = typing.cast(dict[str, list[float]], json_content)
	weapon_reload_times = {weapon : (reload_times[0], reload_times[0]) for weapon, reload_times in json_content.items()}

	for weapon_name in weapons:
		try:
			reload_times = weapon_reload_times[weapon_name]
		except KeyError:
			raise Exception(f"File '{file_name}' is missing weapon '{weapon_name}'.") from None

		weapons[weapon_name].reloadTime = reload_times

	for fake_weapon_name in weapon_reload_times.keys() - weapons.keys():
		print(f"Warning: Weapon '{fake_weapon_name}' found in file '{file_name}' is not an actual weapon.")

	return

def get_operator_weapons(weapons : dict[str, Weapon], file_name : str):
	json_content = deserialize_json(file_name)
	if type(json_content) != dict:
		raise Exception(f"File '{file_name}' doesn't deserialize to a dict of operators and weapons lists.")

	if not all(isinstance(operator, str) for operator in json_content):
		raise Exception(f"The operators in file '{file_name}' don't deserialize to strings.")
	if not all(isinstance(weapon_list, list) for weapon_list in json_content.values()):
		raise Exception(f"The weapon lists in file '{file_name}' don't deserialize to lists.")
	if not all(all(isinstance(weapon, str) for weapon in weapon_list) for weapon_list in json_content.values()):
		raise Exception(f"The weapon lists in file '{file_name}' don't deserialize to lists of strings.")
	operator_weapons = typing.cast(dict[str, list[str]], json_content)

	Weapon.operators = tuple(sorted(operator_weapons.keys()))

	weapon_operatorIndex_dict : dict[str, list[int]] = {}
	for operator, weapon_list in operator_weapons.items():

		operatorIndex = Weapon.operators.index(operator)

		for weapon in weapon_list:
			if weapon in weapon_operatorIndex_dict:
				weapon_operatorIndex_dict[weapon].append(operatorIndex)
			else:
				weapon_operatorIndex_dict[weapon] = [operatorIndex]

		pass

	for weapon_name in weapons:
		try:
			operatorIndices = weapon_operatorIndex_dict[weapon_name]
		except KeyError:
			raise Exception(f"File '{file_name}' is missing weapon '{weapon_name}'.") from None

		weapons[weapon_name].operatorIndices = tuple(operatorIndices)

	for fake_weapon_name in weapon_operatorIndex_dict.keys() - weapons.keys():
		print(f"Warning: Weapon '{fake_weapon_name}' found in file '{file_name}' is not an actual weapon.")
		
	return

def get_weapon_damages(weapons : dict[str, Weapon], data_dir : str):
	weapon_damages : dict[str, list[int]]= {}

	for data_file in os.listdir(data_dir):
		if data_file.endswith(".json"):
			file_name = os.path.join(data_dir, data_file)
			weapon = os.path.splitext(data_file)[0]

			json_content = deserialize_json(file_name)
			if type(json_content) != dict:
				raise Exception(f"File '{file_name}' doesn't deserialize to a dict of distance and damage values.")

			if not all(isinstance(distance, str) for distance in json_content):
				raise Exception(f"The distance values and in file '{file_name}' don't deserialize to strings.")
			if not all(isinstance(damage, int) for damage in json_content.values()):
				raise Exception(f"The distance values and in file '{file_name}' don't deserialize to strings.")
			json_content = typing.cast(dict[str, int], json_content)

			distances = [int(distance) for distance in json_content]
			damages = [damage for damage in json_content.values()]

			
			#if Weapon.distances != distances:
			if not np.array_equal(Weapon.distances, distances):
				raise Exception(f"The distance values in file '{file_name}' are not correct.")
			
			# interpolate gaps. damages will be continuous in [5;40]
			previous_real_damage = 0
			previous_was_interpolated = False
			for i in range(len(Weapon.distances)):
				if damages[i] != 0:	# value is given

					if previous_was_interpolated == True and damages[i] != previous_real_damage and previous_real_damage != 0:	# unequal damage values before and after missing damage data
						raise Exception(f"Tried to interpolate between two unequal damage values '{previous_real_damage}' and '{damages[i]}' at {Weapon.distances[i]}m in '{data_file}'.")

					if damages[i] > damages[i-1] and previous_real_damage != 0:	# detected damage increase. should have been decrease or stagnation
						raise Exception(f"Detected damage increase from '{damages[i-1]}' to '{damages[i]}' at {Weapon.distances[i-1]}m-{Weapon.distances[i]}m in '{data_file}'.")

					previous_was_interpolated = False
					previous_real_damage = damages[i]

				else:	# value needs to be interpolated
					previous_was_interpolated = True
					damages[i] = previous_real_damage

			# get index to first non-zero damage
			first_nonzero_index = next((i for i, damage in enumerate(damages) if damage != 0), -1)

			# extrapolate first 5 meters. damages will be continuous in [0;4]
			if first_nonzero_index == 0:
				pass	# no extrapolation needed
			elif first_nonzero_index == -1:
				raise Exception(f"This exception should not be triggerd. '{file_name}' has to be corrupted in a strange way.")
			else:
				if damages[first_nonzero_index] == damages[first_nonzero_index+1] == damages[first_nonzero_index+2]:
					for i in range(first_nonzero_index):
						damages[i] = damages[first_nonzero_index]
				else:
					print(f"Warning: Can't extrapolate first {first_nonzero_index} meters for '{file_name}'.")

			# write this weapons stats to the weapons dict
			weapon_damages[weapon] = damages

	for weapon_name in weapons:
		try:
			damages = weapon_damages[weapon_name]
		except KeyError:
			raise Exception(f"Directory '{data_dir}' is missing weapon '{weapon_name}'.") from None

		weapons[weapon_name].damages = np.array(damages)

	for fake_weapon_name in weapon_damages.keys() - weapons.keys():
		print(f"Warning: Weapon '{fake_weapon_name}' found in directory '{data_dir}' is not an actual weapon.")

	return

def main() -> None:
	# get all weapon names
	with open(weapon_names_file_name, "r") as file:
		try:
			weapon_names = json.load(file)
		except json.JSONDecodeError:
			raise Exception(f"The json deserialization of file '{weapon_names_file_name}' failed.")
	if type(weapon_names) != list:
		raise Exception(f"File '{weapon_names_file_name}' doesn't deserialize to a list of weapon names.")
	if not all(isinstance(name, str) for name in weapon_names):
		raise Exception(f"File '{weapon_names_file_name}' doesn't deserialize to a list of weapon names.")
	weapon_names = typing.cast(list[str], weapon_names)

	# create the dict containing all weapon object
	weapons = {weapon_name : Weapon(weapon_name) for weapon_name in weapon_names if not weapon_name.startswith("#") or print(f"Warning: Excluding weapon '{weapon_name}' because of #.")}

	# get all weapon types
	get_weapon_types(weapons, weapon_types_file_name)

	# get all weapon fire rates
	get_weapon_firerates(weapons, weapon_firerates_file_name)

	# get all weapon pellet counts
	get_weapon_pellet_counts(weapons, weapon_pellet_counts_file_name)

	# get all weapon ads times
	get_weapon_ads_times(weapons, weapon_ads_times_file_name)

	# get all weapon reload times
	get_weapon_reload_times(weapons, weapon_reload_times_file_name)

	# get all operator weapons
	get_operator_weapons(weapons, operator_weapons_file_name)

	# get all weapon damages
	get_weapon_damages(weapons, damage_data_dir)
	


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