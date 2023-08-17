

###################################################
# settings
###################################################

# the delimiter used in csv files
csv_delimiter = ";"

# the file containing the weapons each operator has access to
operator_weapons_file_name = "operator_weapons.json"

# the directory containing the weapon damage files
weapon_data_dir = "weapons"

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
from re import S

# install exception catcher
def show_exception_and_exit(exc_type, exc_value, tb):
    traceback.print_exception(exc_type, exc_value, tb)
    input()
    sys.exit(-1)
sys.excepthook = show_exception_and_exit


# check if the settings are correct
if not os.path.isfile(operator_weapons_file_name):
	raise Exception(f"'{operator_weapons_file_name}' is not a valid file path.")

if not os.path.isdir(weapon_data_dir):
	raise Exception(f"'{weapon_data_dir}' is not a valid directory.")

if not 0 <= first_distance:
	raise Exception(f"'first_distance' must be >=0 but is {first_distance}.")
if not first_distance <= last_distance:
	raise Exception(f"'last_distance' must be >='first_distance'={first_distance} but is {last_distance}.")


class Weapon:
	types = ("AR", "SMG", "LMG", "DMR", "SG", "Pistol", "MP", "Else")
	operators : tuple[str,...]
	distances = np.array([i for i in range(first_distance, last_distance+1)], np.int32)

	def __init__(self, name_ : str, json_content):
		self.name = name_

		self.operator_indices : tuple[int,...]

		self.__type : int
		self.__rpm : int
		self.damages : np.ndarray
		self.reloadTimes : tuple[float, float]
		self.ads : float
		self.pellets : int
		
		if type(json_content) != dict:
			raise Exception(f"Weapon '{self.name}' doesn't deserialize to a dict.") from None
		
		# get weapon type
		if "type" not in json_content:
			raise Exception(f"Weapon '{self.name}' is missing a type.") from None
		if type(json_content["type"]) != str:
			raise Exception(f"Weapon '{self.name}' has a type that doesn't deserialize to a string.") from None
		if json_content["type"] not in self.types:
			raise Exception(f"Weapon '{self.name}' has an invalid type.") from None
		self.__type = self.types.index(json_content["type"])
		
		# get weapon fire rate
		if "rpm" not in json_content:
			raise Exception(f"Weapon '{self.name}' is missing a fire rate.") from None
		if type(json_content["rpm"]) != int:
			raise Exception(f"Weapon '{self.name}' has a fire rate that doesn't deserialize to an int.") from None
		self.__rpm = json_content["rpm"]
		
		# get weapon ads time
		if "ads" not in json_content:
			raise Exception(f"Weapon '{self.name}' is missing an ads time.") from None
		if type(json_content["ads"]) != float:
			raise Exception(f"Weapon '{self.name}' has an ads time that doesn't deserialize to a float.") from None
		self.ads = json_content["ads"]

		# get weapon pellet count
		if "pellets" not in json_content:
			raise Exception(f"Weapon '{self.name}' is missing a pellet count.") from None
		if type(json_content["pellets"]) != int:
			raise Exception(f"Weapon '{self.name}' has a pellet count that doesn't deserialize to an int.") from None
		self.pellets = json_content["pellets"]
		
		# get weapon reload times
		if "reloadTimes" not in json_content:
			raise Exception(f"Weapon '{self.name}' is missing reload times.") from None
		if type(json_content["reloadTimes"]) != list:
			raise Exception(f"Weapon '{self.name}' has reload times that don't deserialize to a list.") from None
		if len(json_content["reloadTimes"]) != 2:
			raise Exception(f"Weapon '{self.name}' doesn't have exactly 2 reload times.") from None
		if type(json_content["reloadTimes"][0]) != float or type(json_content["reloadTimes"][1]) != float:
			raise Exception(f"Weapon '{self.name}' has reload times that don't deserialize to floats.") from None
		self.reloadTimes = (json_content["reloadTimes"][0], json_content["reloadTimes"][1])
		
		# get weapon damages
		if "damages" not in json_content:
			raise Exception(f"Weapon '{self.name}' is missing damage values.") from None
		if type(json_content["damages"]) != dict:
			raise Exception(f"Weapon '{self.name}' has damage values that don't deserialize to a dict.") from None
		if not all(isinstance(distance, str) for distance in json_content["damages"]):
			raise Exception(f"Weapon '{self.name}' has distance values that don't deserialize to strings.") from None
		if not all(isinstance(damage, int) for damage in json_content["damages"].values()):
			raise Exception(f"Weapon '{self.name}' has damage values that don't deserialize to integers.") from None
		dist_dam_dict = typing.cast(dict[str, int], json_content["damages"])	

		for distance in self.distances:
			if str(distance) not in dist_dam_dict:
				dist_dam_dict[str(distance)] = 0
			
		distances = [int(distance) for distance in dist_dam_dict]
		damages = [damage for damage in dist_dam_dict.values()]
		
		#if Weapon.distances != distances:
		if not np.array_equal(Weapon.distances, distances):
			raise Exception(f"Weapon '{self.name}' has distance values that are not correct.")

		# interpolate gaps. damages will be continuous in [5;40]
		previous_real_damage = 0
		previous_was_interpolated = False
		for i in range(len(Weapon.distances)):
			if damages[i] != 0:	# value is given

				if previous_was_interpolated == True and damages[i] != previous_real_damage and previous_real_damage != 0:	# unequal damage values before and after missing damage data
					raise Exception(f"Tried to interpolate between two unequal damage values '{previous_real_damage}' and '{damages[i]}' at {Weapon.distances[i]}m for weapon '{self.name}'.") from None

				if damages[i] > damages[i-1] and previous_real_damage != 0:	# detected damage increase. should have been decrease or stagnation
					raise Exception(f"Detected damage increase from '{damages[i-1]}' to '{damages[i]}' at {Weapon.distances[i-1]}m-{Weapon.distances[i]}m for weapon '{self.name}'.") from None

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
			raise Exception(f"This exception should not be triggerd. '{self.name}' has to be corrupted in a strange way.") from None
		else:
			if damages[first_nonzero_index] == damages[first_nonzero_index+1] == damages[first_nonzero_index+2]:
				for i in range(first_nonzero_index):
					damages[i] = damages[first_nonzero_index]
			else:
				print(f"Warning: Can't extrapolate first {first_nonzero_index} meters for weapon '{self.name}'.")

		# save the damage stats
		self.damages = np.array(damages)
		return

	def getOperators(self):
		return tuple([self.operators[opIndex] for opIndex in self.operator_indices])

	def getType(self):
		return self.types[self.__type]

	def getRPM(self):
		return self.__rpm
	def getRPS(self):
		return float(self.__rpm) / 60.
	def getRPMS(self):
		return float(self.__rpm) / 60000.

	def getDPS(self):
		return self.damages * self.getRPS()

	def getSTDOK(self, hp : int):
		return np.ceil(hp / self.damages).astype(np.int32)

	def getTTDOK(self, hp : int):
		return self.getSTDOK(hp) / self.getRPMS()
	

def deserialize_json(file_name : str):
	with open(file_name, "r") as file:
		try:
			content = json.load(file)
		except json.JSONDecodeError:
			raise Exception(f"The json deserialization of file '{file_name}' failed.") from None
	return content

def get_operator_weapons(weapons : dict[str, Weapon], file_name : str):
	json_content = deserialize_json(file_name)
	if type(json_content) != dict:
		raise Exception(f"File '{file_name}' doesn't deserialize to a dict of operators and weapons lists.")

	if not all(isinstance(operator, str) for operator in json_content):
		raise Exception(f"The operators in file '{file_name}' don't deserialize to strings.") from None
	if not all(isinstance(weapon_list, list) for weapon_list in json_content.values()):
		raise Exception(f"The weapon lists in file '{file_name}' don't deserialize to lists.") from None
	if not all(all(isinstance(weapon, str) for weapon in weapon_list) for weapon_list in json_content.values()):
		raise Exception(f"The weapon lists in file '{file_name}' don't deserialize to lists of strings.") from None
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

		weapons[weapon_name].operator_indices = tuple(sorted(operatorIndices))

	for fake_weapon_name in weapon_operatorIndex_dict.keys() - weapons.keys():
		print(f"Warning: Weapon '{fake_weapon_name}' found in file '{file_name}' is not an actual weapon.")
		
	return

def get_weapons_dict():
	weapons : dict[str, Weapon] = {}	

	for file_name in os.listdir(weapon_data_dir):
		file_path = os.path.join(weapon_data_dir, file_name)		

		name, extension = os.path.splitext(file_name);		
		if not extension == ".json":
			continue
		if name.startswith("_"):
			print(f"Warning: Excluding weapon '{name}' because of _.")
			continue
		
		weapons[name] = Weapon(name, deserialize_json(file_path))
		
	# get all operator weapons
	get_operator_weapons(weapons, operator_weapons_file_name)
	
	return weapons

def safe_to_csv_file(weapons : dict[str, Weapon]):
	weapons_list = sorted(weapons.values(), key=lambda x: x.getType(), reverse=False)

	# combine the weapon stats into a single string
	content : str = "Damage over distance\n"
	content += "Distance" + csv_delimiter + csv_delimiter.join([str(distance) for distance in Weapon.distances]) + csv_delimiter + csv_delimiter + "RPM" + "\n"
	for weapon in weapons_list:
		content += weapon.name + csv_delimiter + csv_delimiter.join([str(damage) for damage in  weapon.damages]) + csv_delimiter + csv_delimiter + str(weapon.getRPM()) + "\n"
		
	content += "\nDamage per second\n"
	content += "Distance" + csv_delimiter + csv_delimiter.join([str(distance) for distance in Weapon.distances]) + csv_delimiter + csv_delimiter + "RPM" + "\n"
	for weapon in weapons_list:
		content += weapon.name + csv_delimiter + csv_delimiter.join([str(round(dps)) for dps in weapon.getDPS()]) + csv_delimiter + csv_delimiter + str(weapon.getRPM()) + "\n"
		
	# write the string to file
	target_file_name = f"R6S-Weapon-Statistics.csv"
	with open(target_file_name, "x") as target_file:
		target_file.write(content)

	return


weapons = get_weapons_dict()
safe_to_csv_file(weapons)

input("\nCompleted!")