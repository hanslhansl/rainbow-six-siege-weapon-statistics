

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
background_colors = ("84ADF0", "C482A3", "8EB4BC", "A08FCB", "94C37F", "FFD45B", "B8B8B8", "F79F57")


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
import os, numpy, json, typing, math, ctypes
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Border, Alignment, NamedStyle, Side
from openpyxl.formatting.rule import ColorScaleRule
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

kernel32 = ctypes.windll.kernel32
kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
message = "\x1b[38;2;83;141;213mMessage:\033[0m"
warning = "\x1b[38;2;255;255;0mWarning:\033[0m"
exception = "\x1b[38;2;255;0;0mException:\033[0m"

def color_to_border_color(s : str):
	r, g, b = int(s[0:2], 16) / 0xFF, int(s[2:4], 16) / 0xFF, int(s[4:6], 16) / 0xFF
    
	r = pow(r, 2.2)
	g = pow(g, 2.2)
	b = pow(b, 2.2)

	mult = 0.65

	r = r * mult
	g = g * mult
	b = b * mult

	r = pow(r, 1/2.2)
	g = pow(g, 1/2.2)
	b = pow(b, 1/2.2)

	r = int(r * 0xFF)
	g = int(g * 0xFF)
	b = int(b * 0xFF)

	return hex(r)[2:] + hex(g)[2:] + hex(b)[2:]

border_style = "thin"

class Weapon:
	types = weapon_types
	operators : tuple[str,...]
	distances = numpy.array([i for i in range(first_distance, last_distance+1)], numpy.int32)
	
	alignment = Alignment("center", "center")
	border_color = "FF8CA5D0"

	borders = [Border(
				left = Side(border_style=border_style, color=color_to_border_color(background_colors[i])),
                right=Side(border_style=border_style, color=color_to_border_color(background_colors[i])),
                top=Side(border_style=border_style, color=color_to_border_color(background_colors[i])),
                bottom=Side(border_style=border_style, color=color_to_border_color(background_colors[i]))
				) for i in range(len(types))]
	fills = [PatternFill(fgColor=background_colors[i], fill_type = "solid") for i in range(len(types))]
	
	stylesABF = (lambda t=types, a=alignment, b=borders, f=fills: [NamedStyle(name=t[i] + " ABF", alignment=a, border=b[i], fill=f[i]) for i in range(len(t))])()
	stylesBF = (lambda t=types, b=borders, f=fills: [NamedStyle(name=t[i] + " BF", border=b[i], fill=f[i]) for i in range(len(t))])()
	stylesAB = (lambda t=types, a=alignment, b=borders: [NamedStyle(name=t[i] + " AB", alignment=a, border=b[i]) for i in range(len(t))])()
	stylesA = NamedStyle(name="AB", alignment=alignment)
	
	start_values : dict[int, dict[str, typing.Any]] = {}
	end_values : dict[int, dict[str, typing.Any]] = {}

	default_rpm = 0
	default_ads = 0.
	default_pellets = 0
	default_reloadTimes = (0., 0.)
	default_capacity = (0, 0)

	def __init__(self, name_ : str, json_content):
		self.name = name_

		self.operator_indices : tuple[int,...]

		self.type_index : int
		self.rpm : int	# rounds per minute
		self.damages : tuple[int,...]
		self.reloadTimes : tuple[float, float]	# time in seconds
		self.ads : float	# time in seconds
		self.pellets : int
		self.capacity : tuple[int, int]	# (magazine, chamber)

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
		if "rpm" in json_content:
			if type(json_content["rpm"]) != int:
				raise Exception(f"Weapon '{self.name}' has a fire rate that doesn't deserialize to an int.") from None
			self.rpm = json_content["rpm"]
		else:
			print(f"{warning} Weapon '{self.name}' is missing a fire rate. Using default value instead.")
			self.rpm = self.default_rpm
		
		# get weapon ads time
		if "ads" in json_content:
			if type(json_content["ads"]) != float:
				raise Exception(f"Weapon '{self.name}' has an ads time that doesn't deserialize to a float.") from None
			self.ads = json_content["ads"]
		else:
			print(f"{warning} Weapon '{self.name}' is missing an ads time. Using default value instead.")
			self.ads = self.default_ads

		# get weapon pellet count
		if "pellets" in json_content:
			if type(json_content["pellets"]) != int:
				raise Exception(f"Weapon '{self.name}' has a pellet count that doesn't deserialize to an integer.") from None
			self.pellets = json_content["pellets"]
		else:
			print(f"{warning} Weapon '{self.name}' is missing a pellet count. Using default value instead.")
			self.pellets = self.default_pellets
		
		# get weapon reload times
		if "reloadTimes" in json_content:
			if type(json_content["reloadTimes"]) != list:
				raise Exception(f"Weapon '{self.name}' has reload times that don't deserialize to a list.") from None
			if len(json_content["reloadTimes"]) != 2:
				raise Exception(f"Weapon '{self.name}' doesn't have exactly 2 reload times.") from None
			if type(json_content["reloadTimes"][0]) != float or type(json_content["reloadTimes"][1]) != float:
				raise Exception(f"Weapon '{self.name}' has reload times that don't deserialize to floats.") from None
			self.reloadTimes = (json_content["reloadTimes"][0], json_content["reloadTimes"][1])
		else:
			print(f"{warning} Weapon '{self.name}' is missing the reload times. Using default value instead.")
			self.reloadTimes = self.default_reloadTimes

		# get weapon magazine capacity
		if "capacity" in json_content:
			if type(json_content["capacity"]) != list:
				raise Exception(f"Weapon '{self.name}' has a magazine capacity that doesn't deserialize to a list.") from None
			if len(json_content["capacity"]) != 2:
				raise Exception(f"Weapon '{self.name}' doesn't have exactly 2 magazine capacity values.") from None
			if type(json_content["capacity"][0]) != int or type(json_content["capacity"][1]) != int:
				raise Exception(f"Weapon '{self.name}' has magazine capacities that don't deserialize to integers.") from None
			self.capacity = (json_content["capacity"][0], json_content["capacity"][1])
		else:
			print(f"{warning} Weapon '{self.name}' is missing the magazine capacity. Using default value instead.")
			self.capacity = self.default_capacity

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
			raise Exception(f"Weapon '{self.name}' has incorrect distance values.")

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
					raise Exception(f"Weapon '{self.name}' has a damage increase from '{previous_real_damage}' to '{damages[i]}' at {Weapon.distances[i]}m.") from None
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
			raise Exception(f"Weapon '{self.name}' has no damage values at all.") from None
		else:
			if self.type_index == 4:	# special treatment for shotguns
				if first_nonzero_index <= 5:
					for i in range(first_nonzero_index):
						damages[i] = damages[first_nonzero_index]
				else:
					raise Exception(f"Can't extrapolate first {first_nonzero_index} meters for shotgun '{self.name}'.") from None
			else:
				if damages[first_nonzero_index] == damages[first_nonzero_index+1] == damages[first_nonzero_index+2]:
					for i in range(first_nonzero_index):
						damages[i] = damages[first_nonzero_index]
				else:
					raise Exception(f"Can't extrapolate first {first_nonzero_index} meters for weapon '{self.name}'.") from None

		# save the damage stats
		self.damages = tuple(damages)
		
		if self.type_index not in Weapon.start_values:
			Weapon.start_values[self.type_index] = {}
		if self.type_index not in Weapon.end_values:
			Weapon.end_values[self.type_index] = {}

		stat_name = "Damage per bullet"
		if stat_name not in Weapon.start_values[self.type_index]:
			Weapon.start_values[self.type_index][stat_name] = min(self.damages)
		else:
			Weapon.start_values[self.type_index][stat_name] = min(Weapon.start_values[self.type_index][stat_name], min(self.damages))
		if stat_name not in Weapon.end_values[self.type_index]:
			Weapon.end_values[self.type_index][stat_name] = max(self.damages)
		else:
			Weapon.end_values[self.type_index][stat_name] = max(Weapon.end_values[self.type_index][stat_name], max(self.damages))
			
		stat_name = "Damage per bullet per second"
		DPS = tuple([self.getDPS(i)[0] for i in range(len(Weapon.distances))])
		if stat_name not in Weapon.start_values[self.type_index]:
			Weapon.start_values[self.type_index][stat_name] = min(DPS)
		else:
			Weapon.start_values[self.type_index][stat_name] = min(Weapon.start_values[self.type_index][stat_name], min(DPS))
		if stat_name not in Weapon.end_values[self.type_index]:
			Weapon.end_values[self.type_index][stat_name] = max(DPS)
		else:
			Weapon.end_values[self.type_index][stat_name] = max(Weapon.end_values[self.type_index][stat_name], max(DPS))
			
		stat_name = "Damage per shot (relevant for shotguns)"
		DamagePerShot = tuple([self.getDamagePerShot(i)[0] for i in range(len(Weapon.distances))])
		if stat_name not in Weapon.start_values[self.type_index]:
			Weapon.start_values[self.type_index][stat_name] = min(DamagePerShot)
		else:
			Weapon.start_values[self.type_index][stat_name] = min(Weapon.start_values[self.type_index][stat_name], min(DamagePerShot))
		if stat_name not in Weapon.end_values[self.type_index]:
			Weapon.end_values[self.type_index][stat_name] = max(DamagePerShot)
		else:
			Weapon.end_values[self.type_index][stat_name] = max(Weapon.end_values[self.type_index][stat_name], max(DamagePerShot))
			
		stat_name = "Damage per shot per second (relevant for shotguns)"
		DamagePerShotPerSecond = tuple([self.getDamagePerShotPerSecond(i)[0] for i in range(len(Weapon.distances))])
		if stat_name not in Weapon.start_values[self.type_index]:
			Weapon.start_values[self.type_index][stat_name] = min(DamagePerShotPerSecond)
		else:
			Weapon.start_values[self.type_index][stat_name] = min(Weapon.start_values[self.type_index][stat_name], min(DamagePerShotPerSecond))
		if stat_name not in Weapon.end_values[self.type_index]:
			Weapon.end_values[self.type_index][stat_name] = max(DamagePerShotPerSecond)
		else:
			Weapon.end_values[self.type_index][stat_name] = max(Weapon.end_values[self.type_index][stat_name], max(DamagePerShotPerSecond))
		
		return


	def getName(self):
		return self.name, self.getStyleBF()
	def getType(self):
		return self.types[self.type_index], self.getStyleABF()
	def getRPM(self):
		return self.rpm, self.getStyleABF()
	def getRPS(self):
		return self.rpm / 60., self.getStyleABF()
	def getRPMS(self):
		return self.rpm / 60000., self.getStyleABF()
	def getDamage(self, index : int):
		return self.damages[index], self.getStyle(index)
	def getDPS(self, index : int):
		return round(self.damages[index] * self.rpm / 60.), self.getStyleAB()
	def getSTDOK(self, index : int, hp : int):
		return math.ceil(hp / self.damages[index])
	def getTTDOK(self, index : int, hp : int):
		return self.getSTDOK(index, hp) / self.getRPMS()[0]
	def getCapacity(self):
		return str(self.capacity[0]) + "+" + str(self.capacity[1]), self.getStyleABF()
	def getReloadTimes(self):
		return str(self.reloadTimes[0]), str(self.reloadTimes[1]), self.getStyleABF()
	def getPellets(self):
		if self.pellets == 1:
			return "", self.getStyleA()
		else:
			return self.pellets, self.getStyleABF()
	def getDamagePerShot(self, index : int):
		return self.damages[index] * self.pellets, self.getStyle(index)
	def getDamagePerShotPerSecond(self, index : int):
		dps, _ = self.getDPS(index)
		return dps * self.pellets, self.getStyleAB()
	def getADSTime(self):
		return str(self.ads), self.getStyleABF()

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
			return self.getStyleA()
	def getStyleABF(self):
		return self.stylesABF[self.type_index]
	def getStyleAB(self):
		return self.stylesAB[self.type_index]
	def getStyleBF(self):
		return self.stylesBF[self.type_index]
	def getStyleA(self):
		return self.stylesA


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
		print(f"{warning} Weapon '{fake_weapon_name}' found in file '{file_name}' is not an actual weapon.")
		
	return

def get_weapons_dict() -> list[Weapon]:
	weapons : list[Weapon] = []

	for file_name in os.listdir(weapon_data_dir):
		file_path = os.path.join(weapon_data_dir, file_name)		

		name, extension = os.path.splitext(file_name);		
		if not extension == ".json":
			continue
		if name.startswith("_"):
			print(f"{message} Excluding weapon '{name}' because of _.")
			continue
		
		weapons.append(Weapon(name, deserialize_json(file_path)))
		
	# get all operator weapons
	get_operator_weapons(weapons, operator_weapons_file_name)
	
	return weapons

def add_stat_to_worksheet(worksheet, weapons : list[Weapon], stat_name, stat_method, row):
	worksheet.merge_cells(start_row=row, end_row=row, start_column=1, end_column=1 + len(Weapon.distances))
	worksheet.cell(row=row, column=1).value = stat_name
	row += 1
	worksheet.cell(row=row, column=1).value = "Distance"
	for col in range(2, len(Weapon.distances) + 2):
		c = worksheet.cell(row=row, column=col)
		c.value = Weapon.distances[col - 2]
		c.alignment = Weapon.alignment
		
	col += 2
	c = worksheet.cell(row=row, column=col)
	c.value = "Type"
	c.alignment = Weapon.alignment

	col += 2
	c = worksheet.cell(row=row, column=col)
	c.value = "RPM"
	c.alignment = Weapon.alignment

	col += 1
	c = worksheet.cell(row=row, column=col)
	c.value = "Capacity"
	c.alignment = Weapon.alignment
	
	col += 1
	c = worksheet.cell(row=row, column=col)
	c.value = "Pellets"
	c.alignment = Weapon.alignment
	
	col += 2
	c = worksheet.cell(row=row, column=col)
	c.value = "ADS time"
	c.alignment = Weapon.alignment	

	# col += 1
	# worksheet.merge_cells(start_row=row-1, end_row=row-1, start_column=col, end_column=col + 1)
	# c = worksheet.cell(row=row-1, column=col)
	# c.value = "Reload times"
	# c.alignment = Weapon.alignment
	
	# c = worksheet.cell(row=row, column=col)
	# c.value = "Tactical"
	# c.alignment = Weapon.alignment
	
	# c = worksheet.cell(row=row, column=col + 1)
	# c.value = "Full"
	# c.alignment = Weapon.alignment
	
	for weapon in weapons:
		row += 1

		c = worksheet.cell(row=row, column=1)
		c.value, c.style = weapon.getName()
		
		for col in range(2, len(Weapon.distances) + 2):
			c = worksheet.cell(row=row, column=col)
			c.value, c.style = stat_method(weapon, col - 2)

		if stat_name in ("Damage per bullet per second", "Damage per shot per second (relevant for shotguns)"):
			end_color = background_colors[weapon.type_index]
			start_color = "FFFFFF"

			start_value = Weapon.start_values[weapon.type_index][stat_name]
			end_value = Weapon.end_values[weapon.type_index][stat_name]
			mid_value = (start_value + end_value) / 2
			color_rule = ColorScaleRule(start_type="num", start_value=start_value, start_color=start_color,
										end_type="num", end_value=end_value, end_color=end_color)
			worksheet.conditional_formatting.add(f"{get_column_letter(2)}{row}:{get_column_letter(len(Weapon.distances)+2)}{row}", color_rule)

		col += 2
		c = worksheet.cell(row=row, column=col)
		c.value, c.style = weapon.getType()
		
		col += 2
		c = worksheet.cell(row=row, column=col)
		c.value, c.style = weapon.getRPM()
		
		col += 1
		c = worksheet.cell(row=row, column=col)
		c.value, c.style = weapon.getCapacity()

		col += 1
		c = worksheet.cell(row=row, column=col)
		c.value, c.style = weapon.getPellets()
		
		col += 2
		c = worksheet.cell(row=row, column=col)
		c.value, c.style = weapon.getADSTime()

		# col += 1
		# c1 = worksheet.cell(row=row, column=col)
		# col += 1
		# c2 = worksheet.cell(row=row, column=col)
		# c1.value, c2.value, c1.style = weapon.getReloadTimes()
		# c2.style = c1.style
		
	return row

def safe_to_xlsx_file(weapons):
	""" https://openpyxl.readthedocs.io/en/stable/ """
	file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "R6S-Weapon-Statistics.xlsx")
	worksheet_title = "Operation Dread Factor"

	weapons : list[Weapon] = sorted(weapons, key=lambda x: x.type_index, reverse=False)	

	# create the workbook
	workbook = Workbook()

	# get active the
	worksheet = workbook.active
	
	# set the worksheet title
	worksheet.title = worksheet_title
	#workbook.create_sheet("Charts")

	row = 1
	worksheet.merge_cells(start_row=row, end_row=row, start_column=1, end_column=1 + len(Weapon.distances))
	worksheet.cell(row=row, column=1).value = "created by hanslhansl"
	
	row += 1
	worksheet.merge_cells(start_row=row, end_row=row, start_column=1, end_column=1 + len(Weapon.distances))
	worksheet.cell(row=row, column=1).value = "For a detailed explanation see https://github.com/hanslhansl/R6S-Weapon-Statistics"

	row += 2
	row = add_stat_to_worksheet(worksheet, weapons, "Damage per bullet", Weapon.getDamage, row)

	row += 2
	row = add_stat_to_worksheet(worksheet, weapons, "Damage per bullet per second", Weapon.getDPS, row)

	row += 2
	row = add_stat_to_worksheet(worksheet, weapons, "Damage per shot (relevant for shotguns)", Weapon.getDamagePerShot, row)

	row += 2
	row = add_stat_to_worksheet(worksheet, weapons, "Damage per shot per second (relevant for shotguns)", Weapon.getDamagePerShotPerSecond, row)
	
	# save to file
	workbook.save(file_path)

	return


weapons = get_weapons_dict()
safe_to_xlsx_file(weapons)

print("\nCompleted!")
