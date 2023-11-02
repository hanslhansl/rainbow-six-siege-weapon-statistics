

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
weapon_types = ("AR", "SMG", "MP", "LMG", "DMR", "SG", "Pistol", "Else")

# weapon type background colors
background_colors = ("5083EA", "B6668E", "76A5AE", "8771BD", "7CB563", "FFBC01", "A3A3A3", "F48020")
background_colors = ("5083EA", "B6668E", "76A5AE", "8771BD", "7CB563", "FFD609", "A3A3A3", "F47220")


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
import os, numpy, json, typing, math, ctypes, copy
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Border, Alignment, NamedStyle, Side
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter

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

def warning(s):
	return f"\x1b[38;2;255;255;0m{s}\033[0m"
def message(s):
	return f"\x1b[38;2;83;141;213m{s}\033[0m"

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

	ret = hex(r)[2:].zfill(2) + hex(g)[2:].zfill(2) + hex(b)[2:].zfill(2)
	return ret

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
	stylesA = NamedStyle(name="A", alignment=alignment)
	
	lowest_damage : dict[int, int] = {}
	highest_damage : dict[int, int] = {}

	lowest_dps : dict[int, int] = {}
	highest_dps : dict[int, int] = {}

	default_rpm = 0
	default_ads = 0.
	default_pellets = 0
	default_reload_times = (0., 0.)
	default_capacity = (0, 0)
	default_extended_barrel = False

	extended_barrel_weapon_name = "+ extended barrel"
	extended_barrel_damage_multiplier = 1.1

	def __init__(self, name_ : str, json_content_):
		self.name = name_
		self.json_content =  json_content_

		self.operator_indices : tuple[int,...]
		
		self._damages = None
		self._type_index = None
		self._rpm = None	# rounds per minute
		self._reload_times = None	# time in seconds
		self._ads = None	# time in seconds
		self._pellets = None	# number of pellets
		self._capacity = None	# (magazine, chamber)
		self._extended_barrel = None # whether the weapon has an extended barrel attachment

		if type(self.json_content) != dict:
			raise Exception(f"Weapon '{self.name}' doesn't deserialize to a dict.")
		
		if self.type_index not in Weapon.lowest_damage:
			Weapon.lowest_damage[self.type_index] = min(self.damages)
		else:
			Weapon.lowest_damage[self.type_index] = min(Weapon.lowest_damage[self.type_index], min(self.damages))
		if self.type_index not in Weapon.highest_damage:
			Weapon.highest_damage[self.type_index] = max(self.damages)
		else:
			Weapon.highest_damage[self.type_index] = max(Weapon.highest_damage[self.type_index], max(self.damages))

		DPS = tuple([self.getDPS(i)[0] for i in range(len(Weapon.distances))])
		if self.type_index not in Weapon.lowest_dps:
			Weapon.lowest_dps[self.type_index] = min(DPS)
		else:
			Weapon.lowest_dps[self.type_index] = min(Weapon.lowest_dps[self.type_index], min(DPS))
		if self.type_index not in Weapon.highest_dps:
			Weapon.highest_dps[self.type_index] = max(DPS)
		else:
			Weapon.highest_dps[self.type_index] = max(Weapon.highest_dps[self.type_index], max(DPS))

		return

	@property
	def type_index(self) -> int:
		if self._type_index == None:
			# get weapon type
			if "type" not in self.json_content:
				raise Exception(f"Weapon '{self.name}' is missing a type.")
			if type(self.json_content["type"]) != str:
				raise Exception(f"Weapon '{self.name}' has a type that doesn't deserialize to a string.")
			if self.json_content["type"] not in self.types:
				raise Exception(f"Weapon '{self.name}' has an invalid type.")
			self._type_index = self.types.index(self.json_content["type"])
		
		return self._type_index
	@property
	def rpm(self) -> int:
		if self._rpm == None:
			# get weapon fire rate
			if "rpm" in self.json_content:
				if type(self.json_content["rpm"]) != int:
					raise Exception(f"Weapon '{self.name}' has a fire rate that doesn't deserialize to an int.")
				self._rpm = self.json_content["rpm"]
			else:
				print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing a {warning('fire rate')}. Using default value ({self.default_rpm}) instead.")
				self._rpm = self.default_rpm
				
		return self._rpm
	@property
	def ads(self) -> float:
		if self._ads == None:
			# get weapon ads time
			if "ads" in self.json_content:
				if type(self.json_content["ads"]) != float:
					raise Exception(f"Weapon '{self.name}' has an ads time that doesn't deserialize to a float.")
				self._ads = self.json_content["ads"]
			else:
				print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing an {warning('ads time')}. Using default value ({self.default_ads}) instead.")
				self._ads = self.default_ads
			
		return self._ads
	@property
	def pellets(self) -> int:
		if self._pellets == None:
			# get weapon pellet count
			if "pellets" in self.json_content:
				if type(self.json_content["pellets"]) != int:
					raise Exception(f"Weapon '{self.name}' has a pellet count that doesn't deserialize to an integer.")
				self._pellets = self.json_content["pellets"]
			else:
				print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing a {warning('pellet count')}. Using default value ({self.default_pellets}) instead.")
				self._pellets = self.default_pellets
				
		return self._pellets
	@property
	def reload_times(self) -> tuple[float, float]:
		if self._reload_times == None:
			# get weapon reload times
			if "reload_times" in self.json_content:
				if type(self.json_content["reload_times"]) != list:
					raise Exception(f"Weapon '{self.name}' has reload times that don't deserialize to a list.")
				if len(self.json_content["reload_times"]) != 2:
					raise Exception(f"Weapon '{self.name}' doesn't have exactly 2 reload times.")
				if type(self.json_content["reload_times"][0]) != float or type(self.json_content["reload_times"][1]) != float:
					raise Exception(f"Weapon '{self.name}' has reload times that don't deserialize to floats.")
				self._reload_times = (self.json_content["reload_times"][0], self.json_content["reload_times"][1])
			else:
				print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing the {warning('reload times')}. Using default value ({self.default_reload_times}) instead.")
				self._reload_times = self.default_reload_times
				
		return self._reload_times
	@property
	def capacity(self) -> tuple[int, int]:
		if self._capacity == None:
			# get weapon magazine capacity
			if "capacity" in self.json_content:
				if type(self.json_content["capacity"]) != list:
					raise Exception(f"Weapon '{self.name}' has a magazine capacity that doesn't deserialize to a list.")
				if len(self.json_content["capacity"]) != 2:
					raise Exception(f"Weapon '{self.name}' doesn't have exactly 2 magazine capacity values.")
				if type(self.json_content["capacity"][0]) != int or type(self.json_content["capacity"][1]) != int:
					raise Exception(f"Weapon '{self.name}' has magazine capacities that don't deserialize to integers.")
				self._capacity = (self.json_content["capacity"][0], self.json_content["capacity"][1])
			else:
				print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing the {warning('magazine capacity')}. Using default value ({self.default_capacity}) instead.")
				self._capacity = self.default_capacity
				
		return self._capacity
	@property
	def damages(self) -> tuple[int,...]:
		if self._damages == None:
			# get weapon damages
			if "damages" not in self.json_content:
				raise Exception(f"Weapon '{self.name}' is missing damage values.")
			if type(self.json_content["damages"]) != dict:
				raise Exception(f"Weapon '{self.name}' has damage values that don't deserialize to a dict.")
			if not all(isinstance(distance, str) for distance in self.json_content["damages"]):
				raise Exception(f"Weapon '{self.name}' has distance values that don't deserialize to strings.")
			if not all(isinstance(damage, int) for damage in self.json_content["damages"].values()):
				raise Exception(f"Weapon '{self.name}' has damage values that don't deserialize to integers.")
			distance_damage_dict = {int(distance) : int(damage) for distance, damage in self.json_content["damages"].items()}

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
						raise Exception(f"Weapon '{self.name}' has a damage increase from '{previous_real_damage}' to '{damages[i]}' at {Weapon.distances[i]}m.")
					if previous_real_damage != 0 and previous_was_interpolated == True and damages[i] != previous_real_damage:
						raise Exception(f"Tried to interpolate between two unequal damage values '{previous_real_damage}' and '{damages[i]}' at {Weapon.distances[i]}m for weapon '{self.name}'.")
				
					previous_real_damage = damages[i]
					previous_was_interpolated = False

			# get index to first non-zero damage
			first_nonzero_index = next((i for i, damage in enumerate(damages) if damage != 0), -1)

			# extrapolate first 5 meters. damages will be continuous in [0;4]
			if first_nonzero_index == 0:
				pass	# no extrapolation needed
			elif first_nonzero_index == -1:
				raise Exception(f"Weapon '{self.name}' has no damage values at all.")
			else:
				if self.types[self.type_index] == "SG":	# special treatment for shotguns
					if first_nonzero_index <= 5:
						for i in range(first_nonzero_index):
							damages[i] = damages[first_nonzero_index]
					else:
						raise Exception(f"Can't extrapolate first {first_nonzero_index} meters for shotgun '{self.name}'.")
				else:
					if damages[first_nonzero_index] == damages[first_nonzero_index+1] == damages[first_nonzero_index+2]:
						for i in range(first_nonzero_index):
							damages[i] = damages[first_nonzero_index]
					else:
						raise Exception(f"Can't extrapolate first {first_nonzero_index} meters for weapon '{self.name}'.")

			# save the damage stats
			self._damages = tuple(damages)

		return self._damages
	@property
	def extended_barrel(self) -> bool:
		if self._extended_barrel == None:
			# get weapon extended barrel
			if "extended_barrel" in self.json_content:
				if type(self.json_content["extended_barrel"]) != bool:
					raise Exception(f"Weapon '{self.name}' has an extended barrel value that doesn't deserialize to a bool.")
				self._extended_barrel = self.json_content["extended_barrel"]
			else:
				print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing an {warning('extended barrel')} value. Using default value ({self.default_extended_barrel}) instead.")
				self._extended_barrel = self.default_extended_barrel
				
		return self._extended_barrel

	def getStartEndValue(self, statName : str):
		if statName == "DPS":
			return self.lowest_dps[self.type_index], self.highest_dps[self.type_index]
		elif statName == "DmgPerShot":
			return min(self.damages) * self.pellets, max(self.damages) * self.pellets

	def getName(self):
		if self.name == self.extended_barrel_weapon_name:
			return self.name, "Normal"
		else:
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
		return round(self.getDamagePerShot(index)[0] * self.rpm / 60.), self.getStyleAB()
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
		return self.damages[index] * self.pellets, self.getStyleAB()
	def getADSTime(self):
		return str(self.ads), self.getStyleABF()
	def getDamageToBaseDamagePercentage(self, index : int):
		val = round(self.damages[index] / max(self.damages), 2)
		if val == 1:
			return 1, self.getStyleAB()
		else:
			return val, self.getStyleAB()
			return str(val)[1:], self.getStyleAB()

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

	def getExtendedBarrelWeapon(self):
		retVar = copy.deepcopy(self)
		
		retVar.name = self.extended_barrel_weapon_name
		retVar.json_content = None

		retVar._damages = tuple(math.ceil(dmg * self.extended_barrel_damage_multiplier) for dmg in self.damages)
		retVar._rpm = self.rpm
		retVar._pellets = self.pellets
		retVar._reload_times = None
		retVar._ads = None
		retVar._capacity = None
		retVar._extended_barrel = False

		return retVar

def deserialize_json(file_name : str):
	with open(file_name, "r") as file:
		try:
			content = json.load(file)
		except json.JSONDecodeError:
			raise Exception(f"The json deserialization of file '{file_name}' failed.")
	return content

def get_operator_weapons(weapons : list[Weapon], file_name : str) -> None:
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
			raise Exception(f"File '{file_name}' is missing weapon '{weapon.name}'.")

		weapon.operator_indices = tuple(sorted(operator_indices))
		del weapon_operatorIndex_dict[weapon.name]

	for fake_weapon_name in weapon_operatorIndex_dict:
		print(f"{warning('Warning:')} Weapon '{warning(fake_weapon_name)}' found in file '{file_name}' is {warning('not an actual weapon')}.")
		
	return

def get_weapons_dict() -> list[Weapon]:
	weapons : list[Weapon] = []

	for file_name in os.listdir(weapon_data_dir):
		file_path = os.path.join(weapon_data_dir, file_name)		

		name, extension = os.path.splitext(file_name);		
		if not extension == ".json":
			continue
		if name.startswith("_"):
			print(f"{message('Message: Excluding')} weapon '{message(name)}' because of _.")
			continue
		
		weapons.append(Weapon(name, deserialize_json(file_path)))
		
	# get all operator weapons
	get_operator_weapons(weapons, operator_weapons_file_name)
	
	return weapons

def add_weapon_to_stat_worksheet(worksheet, weapon : Weapon, stat_name : str, stat_method : str, row : int):
	c = worksheet.cell(row=row, column=1)
	c.value, c.style = weapon.getName()
		
	for col in range(2, len(Weapon.distances) + 2):
		c = worksheet.cell(row=row, column=col)
		c.value, c.style = stat_method(weapon, col - 2)

	if stat_name in ("DPS", "DmgPerShot"):
		end_color = background_colors[weapon.type_index]
		start_color = "FFFFFF"
		start_value, end_value = weapon.getStartEndValue(stat_name)

		color_rule = ColorScaleRule(start_type="num", start_value=start_value, start_color=start_color,
									end_type="num", end_value=end_value, end_color=end_color)
		worksheet.conditional_formatting.add(f"{get_column_letter(2)}{row}:{get_column_letter(len(Weapon.distances)+2)}{row}", color_rule)

	if weapon.getName()[0] != weapon.extended_barrel_weapon_name:
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
	return

def add_stat_worksheet(workbook, weapons : list[Weapon], stat_name : str, stat_display_name : str, stat_link : str, description : str, stat_method):
	worksheet = workbook.create_sheet(stat_display_name)

	row = 1
	worksheet.merge_cells(start_row=row, end_row=row, start_column=1, end_column=1 + len(Weapon.distances))
	worksheet.cell(row=row, column=1).value = "created by hanslhansl"
	
	row += 1
	worksheet.merge_cells(start_row=row, end_row=row, start_column=1, end_column=1 + len(Weapon.distances))
	worksheet.cell(row=row, column=1).value = '=HYPERLINK("https://github.com/hanslhansl/R6S-Weapon-Statistics/", "A detailed explanation can be found here")'
	
	row += 2
	worksheet.merge_cells(start_row=row, end_row=row, start_column=1, end_column=1 + len(Weapon.distances))
	worksheet.cell(row=row, column=1).value = f'=HYPERLINK("https://github.com/hanslhansl/R6S-Weapon-Statistics/#{stat_link}", "{stat_display_name}")'
	
	row += 1
	worksheet.merge_cells(start_row=row, end_row=row, start_column=1, end_column=1 + len(Weapon.distances))
	worksheet.cell(row=row, column=1).value = description

	row += 1
	worksheet.cell(row=row, column=1).value = "Distance"
	for col in range(2, len(Weapon.distances) + 2):
		c = worksheet.cell(row=row, column=col)
		c.value = Weapon.distances[col - 2]
		c.alignment = Weapon.alignment

	space_mult = 6

	col += 1
	c = worksheet.cell(row=row, column=col)
	c.value = " " * space_mult

	col += 1
	c = worksheet.cell(row=row, column=col)
	c.value = "Type"
	c.alignment = Weapon.alignment

	col += 1
	c = worksheet.cell(row=row, column=col)
	c.value = " " * space_mult
	
	col += 1
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
	
	col += 1
	c = worksheet.cell(row=row, column=col)
	c.value = " " * space_mult
	
	col += 1
	c = worksheet.cell(row=row, column=col)
	c.value = "ADS time"
	c.alignment = Weapon.alignment	

	col += 1
	c = worksheet.cell(row=row, column=col)
	c.value = " " * space_mult

	col += 1
	worksheet.merge_cells(start_row=row-1, end_row=row-1, start_column=col, end_column=col + 1)
	c = worksheet.cell(row=row-1, column=col)
	c.value = "Reload times"
	c.alignment = Weapon.alignment
	
	c = worksheet.cell(row=row, column=col)
	c.value = "Tactical"
	c.alignment = Weapon.alignment
	
	c = worksheet.cell(row=row, column=col + 1)
	c.value = "Full"
	c.alignment = Weapon.alignment
	
	for weapon in weapons:
		row += 1
		add_weapon_to_stat_worksheet(worksheet, weapon, stat_name, stat_method, row)
		
		if (weapon.extended_barrel):
			row += 1
			extended_barrel_weapon = weapon.getExtendedBarrelWeapon()
			add_weapon_to_stat_worksheet(worksheet, extended_barrel_weapon, stat_name, stat_method, row)
		
	return

def safe_to_xlsx_file(weapons : list[Weapon]):
	""" https://openpyxl.readthedocs.io/en/stable/ """
	file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Rainbow-Six-Siege-Weapon-Statistics.xlsx")

	weapons = sorted(weapons, key=lambda x: x.type_index, reverse=False)

	# create the workbook
	workbook = Workbook()

	workbook.remove(workbook.active)

	add_stat_worksheet(workbook, weapons, "DMG", "Damage per bullet", "damage-per-bullet",
		       "The colored areas represent steady damage, the colorless areas represent decreasing damage.", Weapon.getDamage)

	add_stat_worksheet(workbook, weapons, "DmgPerShot", "Damage per shot", "damage-per-shot",
		       "The color gradient illustrates the damage compared to the weapon's base damage.", Weapon.getDamagePerShot)

	add_stat_worksheet(workbook, weapons, "DPS", "Damage per second", "damage-per-second---dps",
		       "The color gradient illustrates the DPS compared to the highest DPS of the weapon's type (excluding extended barrel stats).", Weapon.getDPS)

	# save to file
	workbook.save(file_path)

	# resize columns
	import win32com.client
	excel = win32com.client.Dispatch('Excel.Application')
	wb = excel.Workbooks.Open(file_path)
	for i in range(1, excel.Worksheets.Count + 1):
		excel.Worksheets(i).Activate()
		excel.ActiveSheet.Columns.AutoFit()
	excel.Worksheets(1).Activate()
	wb.Save()
	wb.Close()
	excel.Quit()

	return


weapons = get_weapons_dict()
safe_to_xlsx_file(weapons)

input("\nCompleted!")
