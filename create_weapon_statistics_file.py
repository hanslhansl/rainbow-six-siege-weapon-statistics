

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

# weapon types
weapon_types = ("AR", "SMG", "LMG", "DMR", "SG", "Pistol", "MP", "Else")

# weapon type background colors
background_colors = ("A4C2F4", "D5A6BD", "B4A7D6", "B6D7A8", "D0E0E3", "FFE599", "EA9999", "B7B7B7")

###################################################
# settings end
# don't edit from here on
###################################################

# install exception catcher
import sys, traceback

def show_exception_and_exit(exc_type, exc_value, tb):
    traceback.print_exception(exc_type, exc_value, tb)
    input("\nAbort")
    sys.exit(-1)
sys.excepthook = show_exception_and_exit

#imports
from calendar import c
import os, numpy, json, typing, math
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Border, Alignment, NamedStyle, Side
from openpyxl.utils import get_column_letter
from win32com.client import Dispatch

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
	types = weapon_types
	operators : tuple[str,...]
	distances = numpy.array([i for i in range(first_distance, last_distance+1)], numpy.int32)
	
	alignment = Alignment("center", "center")
	border = Border(left=Side(border_style='thin',
                          color='FF8CA5D0'),
                right=Side(border_style='thin',
                           color='FF8CA5D0'),
                top=Side(border_style='thin',
                         color='FF8CA5D0'),
                bottom=Side(border_style='thin',
                            color='FF8CA5D0'))
	fills = [PatternFill(fgColor=background_colors[i], fill_type = "solid") for i in range(len(types))]
	stylesABF = (lambda t=types, a=alignment, b=border, f=fills: [NamedStyle(name=t[i] + " ABF", alignment = a, border=b, fill=f[i]) for i in range(len(t))])()
	stylesBF = (lambda t=types, b=border, f=fills: [NamedStyle(name=t[i] + " BF", border=b, fill=f[i]) for i in range(len(t))])()
	stylesAB = (lambda t=types, a=alignment, b=border: [NamedStyle(name=t[i] + " AB", alignment = a, border=b) for i in range(len(t))])()

	def __init__(self, name_ : str, json_content):
		self.name = name_

		self.operator_indices : tuple[int,...]

		self.type_index : int
		self.rpm : int	# rounds per minute
		self.damages : tuple[int,...]
		self.reloadTimes : tuple[float, float]	# time in seconds
		self.ads : float	# time in seconds
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
		self.type_index = self.types.index(json_content["type"])
		
		# get weapon fire rate
		if "rpm" not in json_content:
			raise Exception(f"Weapon '{self.name}' is missing a fire rate.") from None
		if type(json_content["rpm"]) != int:
			raise Exception(f"Weapon '{self.name}' has a fire rate that doesn't deserialize to an int.") from None
		self.rpm = json_content["rpm"]
		
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
		distance_damage_dict = {int(distance) : int(damage) for distance, damage in json_content["damages"].items()}

		# insert missing distances with damage = 0
		for distance in Weapon.distances:
			if distance not in distance_damage_dict:
				distance_damage_dict[distance] = 0
				
		# sort distance_damage_dict in ascending order by distance
		distance_damage_dict = dict(sorted(distance_damage_dict.items()))

		distances = list(distance_damage_dict.keys())
		damages = list(distance_damage_dict.values())
		
		#if Weapon.distances != distances:
		if not numpy.array_equal(Weapon.distances, distances):
			raise Exception(f"Weapon '{self.name}' has distance values that are not correct.")

		# make sure damages only stagnates or decreases and zeros are surrounded by equal non-zero damages
		# interpolate gaps. damages will be continuous in [5;40]
		previous_real_damage = 0
		previous_was_interpolated = False
		for i in range(len(Weapon.distances)):
			if damages[i] == 0:	# this damage value needs to be interpolated
				damages[i] = previous_real_damage
				previous_was_interpolated = True
				
			else:	# this damage value is given
				if damages[i] > previous_real_damage and previous_real_damage != 0:
					raise Exception(f"Weapon '{self.name}' has a damage increase from '{previous_real_damage}' to '{damages[i]}' at {Weapon.distances[i]}m.")
				if previous_real_damage != 0 and previous_was_interpolated == True and damages[i] != previous_real_damage:
					raise Exception(f"Tried to interpolate between two unequal damage values '{previous_real_damage}' and '{damages[i]}' at {Weapon.distances[i]}m for weapon '{self.name}'.") from None
				
				previous_real_damage = damages[i]
				previous_was_interpolated = False

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
		self.damages = tuple(damages)
		return


	def getName(self):
		return self.name, self.getStyleBF()
	def getType(self):
		return self.types[self.type_index]
	def getRPM(self):
		style = self.getStyleABF()
		if self.rpm == 0:
			return "", style
		else:
			return str(self.rpm), style
	def getRPS(self):
		style = self.getStyleABF()
		if self.rpm == 0:
			return "", style
		else:
			return str(self.rpm / 60.), style
	def getRPMS(self):
		style = self.getStyleABF()
		if self.rpm == 0:
			return "", style
		else:
			return str(self.rpm / 60000.), style
	def getDamage(self, index : int):
		style = self.getStyle(index)
		if self.damages[index] == 0:
			return "", style
		else:
			return str(self.damages[index]), style
	def getDPS(self, index : int):
		style = self.getStyle(index)
		if self.damages[index] == 0 or self.rpm == 0:
			return "", style
		else:
			return str(round(self.damages[index] * self.rpm / 60.)), style
	def getSTDOK(self, index : int, hp : int):
		return math.ceil(hp / self.damages[index])
	def getTTDOK(self, index : int, hp : int):
		return self.getSTDOK(index, hp) / self.getRPMS()[0]
	
	def getOperators(self):
		return tuple([self.operators[opIndex] for opIndex in self.operator_indices])

	def getDamageDropoffBorders(self):
		lastInitialDamageIndex = -1
		firstEndDamageIndex = -1
		
		for i in range(len(self.damages)):
			if self.damages[i] == self.damages[0]:
				lastInitialDamageIndex = i
			elif self.damages[i] == self.damages[-1]:
				firstEndDamageIndex = i
				break

		return (lastInitialDamageIndex, firstEndDamageIndex)

	def getStyle(self, index : int):
		lastInitialDamageIndex, firstEndDamageIndex = self.getDamageDropoffBorders()
		if index <= lastInitialDamageIndex:
			return self.getStyleABF()
		elif index >= firstEndDamageIndex:
			return self.getStyleABF()
		else:
			return self.getStyleAB()
	def getStyleABF(self):
	    return self.stylesABF[self.type_index]
	def getStyleBF(self):
	    return self.stylesBF[self.type_index]
	def getStyleAB(self):
	    return self.stylesAB[self.type_index]

def deserialize_json(file_name : str):
	with open(file_name, "r") as file:
		try:
			content = json.load(file)
		except json.JSONDecodeError:
			raise Exception(f"The json deserialization of file '{file_name}' failed.") from None
	return content

def get_operator_weapons(weapons : list[Weapon], file_name : str) -> None:
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

		for weapon_name in weapon_list:
			if weapon_name in weapon_operatorIndex_dict:
				weapon_operatorIndex_dict[weapon_name].append(operatorIndex)
			else:
				weapon_operatorIndex_dict[weapon_name] = [operatorIndex]
		pass

	for weapon in weapons:
		try:
			operator_indices = weapon_operatorIndex_dict[weapon.name]
		except KeyError:
			raise Exception(f"File '{file_name}' is missing weapon '{weapon.name}'.") from None

		weapon.operator_indices = tuple(sorted(operator_indices))
		del weapon_operatorIndex_dict[weapon.name]

	for fake_weapon_name in weapon_operatorIndex_dict:
		print(f"Warning: Weapon '{fake_weapon_name}' found in file '{file_name}' is not an actual weapon.")
		
	return

def get_weapons_dict() -> list[Weapon]:
	weapons : list[Weapon] = []

	for file_name in os.listdir(weapon_data_dir):
		file_path = os.path.join(weapon_data_dir, file_name)		

		name, extension = os.path.splitext(file_name);		
		if not extension == ".json":
			continue
		if name.startswith("_"):
			print(f"Warning: Excluding weapon '{name}' because of _.")
			continue
		
		weapons.append(Weapon(name, deserialize_json(file_path)))
		
	# get all operator weapons
	get_operator_weapons(weapons, operator_weapons_file_name)
	
	return weapons

def safe_to_csv_file(weapons : list[Weapon]) -> None:
	weapons_list = sorted(weapons, key=lambda x: x.getType(), reverse=False)

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

def safe_to_xlsx_file(weapons : list[Weapon]):
	""" https://openpyxl.readthedocs.io/en/stable/ """
	file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "R6S-Weapon-Statistics.xlsx")
	worksheet_title = "Operation Dread Factor"

	weapons = sorted(weapons, key=lambda x: x.getType(), reverse=False)	

	# create the workbook
	workbook = Workbook()

	# get active the
	worksheet = workbook.active
	
	# set the worksheet title
	worksheet.title = worksheet_title
	#workbook.create_sheet("Charts")

	row = 1
	worksheet.merge_cells(start_row=row, end_row=row, start_column=1, end_column=1 + len(Weapon.distances) + 2)
	worksheet.cell(row=row, column=1).value = "created by hanslhansl"
	
	row += 1
	worksheet.merge_cells(start_row=row, end_row=row, start_column=1, end_column=1 + len(Weapon.distances) + 2)
	worksheet.cell(row=row, column=1).value = "For a detailed explanation see https://github.com/hanslhansl/R6S-Weapon-Statistics"

	row += 2
	worksheet.merge_cells(start_row=row, end_row=row, start_column=1, end_column=1 + len(Weapon.distances) + 2)
	worksheet.cell(row=row, column=1).value = "Damage over distance"
	row += 1
	worksheet.cell(row=row, column=1).value = "Distance"
	for col in range(len(Weapon.distances)):
		c = worksheet.cell(row=row, column=col + 2)
		c.value = Weapon.distances[col]
		c.alignment = Weapon.alignment
	c = worksheet.cell(row=row, column=col + 4)
	c.value = "RPM"
	c.alignment = Weapon.alignment
	for weapon in weapons:
		damages = weapon.damages
		row += 1
		c = worksheet.cell(row=row, column=1)
		c.value, c.style = weapon.getName()
		for col in range(len(damages)):
			c = worksheet.cell(row=row, column=col + 2)
			c.value, c.style = weapon.getDamage(col)			

		c = worksheet.cell(row=row, column=col + 4)
		c.value, c.style = weapon.getRPM()

	row += 2
	worksheet.merge_cells(start_row=row, end_row=row, start_column=1, end_column=1 + len(Weapon.distances) + 2)
	worksheet.cell(row=row, column=1).value = "Damage per second"
	row += 1
	worksheet.cell(row=row, column=1).value = "Distance"
	for col in range(len(Weapon.distances)):
		c = worksheet.cell(row=row, column=col + 2)
		c.value = Weapon.distances[col]
		c.alignment = Weapon.alignment
	c = worksheet.cell(row=row, column=col + 4)
	c.value = "RPM"
	c.alignment = Weapon.alignment
	for weapon in weapons:
		row += 1
		c = worksheet.cell(row=row, column=1)
		c.value, c.style = weapon.getName()
		for col in range(len(Weapon.distances)):
			c = worksheet.cell(row=row, column=col + 2)
			c.value, c.style = weapon.getDPS(col)
		c = worksheet.cell(row=row, column=col + 4)
		c.value, c.style = weapon.getRPM()

	# resize columns
	worksheet.column_dimensions[get_column_letter(1)].width = 18
	for i in range(2, len(Weapon.distances) + 4):
		worksheet.column_dimensions[get_column_letter(i)].width = 5
	#worksheet.column_dimensions[get_column_letter(len(Weapon.distances) + 3)].width = 18
		
	# save to file
	workbook.save(file_path)
	
	# resize columns
	"""excel = Dispatch('Excel.Application')
	wb = excel.Workbooks.Open(file_path)
	excel.Worksheets(1).Activate()
	excel.ActiveSheet.Columns.AutoFit()
	wb.Save()
	wb.Close()
	excel.Quit()"""

	return


weapons = get_weapons_dict()
safe_to_xlsx_file(weapons)

input("\nCompleted!")
