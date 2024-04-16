

###################################################
# settings
###################################################

# the file containing the weapons each operator has access to
operators_file_name = "operators.json"

# the file containing the attachment overview
attachment_overview_file_name = "attachment_overview.json"

# the file name of the html output file
html_output_file_name = "rainbow-six-siege-weapon-statistics.html"

# the file name of the xlsx output file
xlsx_output_file_name = "rainbow-six-siege-weapon-statistics.xlsx"

# the directory containing the weapon damage files
weapon_data_dir = "weapons"

# the distance the weapon damage stats start at (usually 0)
first_distance = 0
# the distance the weapon damage stats end at (usually 40 because the of the Shooting Range limit)
last_distance = 40 

# weapon types
weapon_classes = ("AR", "SMG", "MP", "LMG", "DMR", "SR", "SG", "Slug SG", "Handgun", "Revolver", "Hand Canon")

# weapon type background colors
background_colors = {"AR":"5083EA", "SMG":"B6668E", "MP":"76A5AE", "LMG":"8771BD", "DMR":"7CB563", "SR":"DE2A00", "SG":"FFBC01", "Slug SG":"A64E06", "Handgun":"A3A3A3", "Revolver":"F48020", "Hand Canon":"FFFFFF"}


###################################################
# settings end
# don't edit from here on
###################################################

# install exception catcher
#from re import S
import sys, traceback

def show_exception_and_exit(exc_type, exc_value, tb):
	traceback.print_exception(exc_type, exc_value, tb)
	input("\nAbort")
	sys.exit(-1)
sys.excepthook = show_exception_and_exit

#imports
import os, numpy, json, typing, math, ctypes, copy
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Border, Alignment, NamedStyle, Side, Font
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter

# check if the settings are correct
if not os.path.isfile(operators_file_name):
	raise Exception(f"'{operators_file_name}' is not a valid file path.")

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
	classes = weapon_classes
	operator_tuple : tuple[str,...]
	distances = numpy.array([i for i in range(first_distance, last_distance+1)], numpy.int32)
	
	alignment = Alignment("center", "center", wrapText=True)
	border_color = "FF8CA5D0"

	borders = {type_ : Border(
				left = Side(border_style=border_style, color=color_to_border_color(background_colors[type_])),
				right=Side(border_style=border_style, color=color_to_border_color(background_colors[type_])),
				top=Side(border_style=border_style, color=color_to_border_color(background_colors[type_])),
				bottom=Side(border_style=border_style, color=color_to_border_color(background_colors[type_]))
				) for type_ in classes}
	
	fills = {type_ : PatternFill(fgColor=background_colors[type_], fill_type = "solid") for type_ in classes}
	
	stylesABF = (lambda t=classes, a=alignment, b=borders, f=fills: {type_ : NamedStyle(name=type_ + " ABF", alignment=a, border=b[type_], fill=f[type_]) for type_ in t})()
	stylesBF = (lambda t=classes, b=borders, f=fills: {type_ : NamedStyle(name=type_ + " BF", border=b[type_], fill=f[type_]) for type_ in t})()
	stylesAB = (lambda t=classes, a=alignment, b=borders: {type_ : NamedStyle(name=type_ + " AB", alignment=a, border=b[type_]) for type_ in t})()
	stylesA = NamedStyle(name="A", alignment=alignment)

	default_rpm = 0
	default_ads = 0.
	default_pellets = 1
	default_reload_times = (0., 0.)
	default_capacity = (0, 0)
	default_extra_ammo = 0
	default_extended_barrel = False
	default_grip = False
	default_has_laser = False

	extended_barrel_weapon_name = "+ extended barrel"
	extended_barrel_damage_multiplier = 0.0
	laser_ads_speed_multiplier = 0.0;
	angled_grip_reload_speed_multiplier = 0.0;
	
	tdok_hps = (100, 110, 125, 120, 130, 145)

	lowest_highest_dps : dict[str, tuple[int, int]] = {}	# type : (lowest dps, highest dps)
	lowest_highest_ttdok : dict[int, dict[str, tuple[int, int]]] = {}	# hp : {type : (lowest ttdok, highest ttdok)}
	
	DPSColorScaleRule : dict[str, typing.Any] = {}	# type : ColorScaleRule
	TTDOKColorScaleRules : dict[int, dict[str, typing.Any]] = {}	# hp : {type : ColorScaleRule}

	def __init__(self, json_content_):
		self.json_content =  json_content_

		self.operator_indices : tuple[int,...]
		
		self._name = None
		self._damages = None
		self._class = None	# type string
		self._rpm = None	# rounds per minute
		self._reload_times = None	# time in seconds
		self._ads = None	# time in seconds
		self._pellets = None	# number of pellets
		self._capacity = None	# (magazine, chamber)
		self._extra_ammo = None	# extra ammo
		self._grip = None	# whether the weapon has access to a grip attachment
		self._has_laser = None	# whether the weapon has access to a laser attachment
		
		self._has_extended_barrel = None # whether the weapon has an extended barrel attachment
		self.extended_barrel_weapon : Weapon = None	# the extended barrel version of this weapon
		self.is_extended_barrel = False	# whether this is an extended barrel version
		self.extended_barrel_parent : None | Weapon = None	# the base weapon object if this is an extended barrel version
		
		self.DmgPerShotColorScaleRule = None

		if type(self.json_content) != dict:
			raise Exception(f"Weapon '{self.name}' doesn't deserialize to a dict.")
		
		DPS = tuple([int(self.dps(i) + 0.5) for i in range(len(Weapon.distances))])
		if self.class_ not in Weapon.lowest_highest_dps:
			Weapon.lowest_highest_dps[self.class_] = min(DPS), max(DPS)
		else:
			Weapon.lowest_highest_dps[self.class_] = min(Weapon.lowest_highest_dps[self.class_][0], min(DPS)), max(Weapon.lowest_highest_dps[self.class_][1], max(DPS))

		for hp in self.tdok_hps:
			if hp not in Weapon.lowest_highest_ttdok:
				Weapon.lowest_highest_ttdok[hp] = {}
				
			TTDOK = tuple([int(self.ttdok(i, hp) + 0.5) for i in range(len(Weapon.distances))])
			if self.class_ not in Weapon.lowest_highest_ttdok[hp]:
				Weapon.lowest_highest_ttdok[hp][self.class_] = min(TTDOK), max(TTDOK)
			else:
				Weapon.lowest_highest_ttdok[hp][self.class_] = min(Weapon.lowest_highest_ttdok[hp][self.class_][0], min(TTDOK)), max(Weapon.lowest_highest_ttdok[hp][self.class_][1], max(TTDOK))

		return

	def validate_damages(self, distance_damage_dict : dict[int, int]):
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

		# make sure the last damage value is given. otherwise the extrapolation will be wrong
		if damages[-1] == 0:
			raise Exception(f"Weapon '{self.name}' is missing a damage value at {distances[-1]}m.")

		# make sure damages only stagnate or decrease and zeros are surrounded by identical non-zero damages
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
			if self.class_ == "SG":	# special treatment for shotguns
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

		# return the damage stats
		return tuple(damages)

	# primary properties
	@property
	def name(self) -> str:
		if self._name == None:
			# get weapon name
			if "name" not in self.json_content:
				raise Exception(f"Weapon is missing its name.")
			if type(self.json_content["name"]) != str:
				raise Exception(f"Weapon has a name that doesn't deserialize to a string.")
			self._name = self.json_content["name"]
		
		return self._name
	@property
	def class_(self) -> str:
		if self._class == None:
			# get weapon type
			if "class" not in self.json_content:
				raise Exception(f"Weapon '{self.name}' is missing a type.")
			if type(self.json_content["class"]) != str:
				raise Exception(f"Weapon '{self.name}' has a type that doesn't deserialize to a string.")
			if self.json_content["class"] not in self.classes:
				raise Exception(f"Weapon '{self.name}' has an invalid type.")
			self._class = self.json_content["class"]
		return self._class
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
	def ads_time(self) -> float:
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
	def extra_ammo(self) -> int:
		if self._extra_ammo == None:
			# get extra ammo
			if "extra_ammo" in self.json_content:
				if type(self.json_content["extra_ammo"]) != int:
					raise Exception(f"Weapon '{self.name}' has an extra ammo value that doesn't deserialize to an integer.")
				self._extra_ammo = self.json_content["extra_ammo"]
			else:
				print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing an {warning('extra ammo value')}. Using default value ({self.default_extra_ammo}) instead.")
				self._extra_ammo = self.default_extra_ammo
				
		return self._extra_ammo
	@property
	def damages(self) -> tuple[int,...]:
		if self._damages == None:
			# get weapon damages
			if "damages" not in self.json_content:
				raise Exception(f"Weapon '{self.name}' is missing damage values.")
			if type(self.json_content["damages"]) != dict:
				raise Exception(f"Weapon '{self.name}' has damage values that don't deserialize to a dict.")
			if not all(isinstance(damage, int) for damage in self.json_content["damages"].values()):
				raise Exception(f"Weapon '{self.name}' has damage values that don't deserialize to integers.")
			distance_damage_dict = {int(distance) : int(damage) for distance, damage in self.json_content["damages"].items()}
			
			self._damages = self.validate_damages(distance_damage_dict)

		return self._damages
	@property
	def has_extended_barrel(self) -> bool:
		if self._has_extended_barrel == None:
			potential_eb = copy.deepcopy(self)
		
			# get weapon extended barrel
			if "extended_barrel" in self.json_content:
				if type(self.json_content["extended_barrel"]) == bool:	# use default damage multiplier for extended barrel
					self._has_extended_barrel = self.json_content["extended_barrel"]
					if self._has_extended_barrel == True:
						potential_eb._damages = tuple(math.ceil(dmg * self.extended_barrel_damage_multiplier) for dmg in self.damages)
					
				elif type(self.json_content["extended_barrel"]) == dict:	# use custom damages for extended barrel
					if not all(isinstance(damage, int) for damage in self.json_content["extended_barrel"].values()):
						raise Exception(f"Weapon '{self.name}' has damage values that don't deserialize to integers.")
					distance_damage_dict = {int(distance) : int(damage) for distance, damage in self.json_content["extended_barrel"].items()}
					potential_eb._damages = self.validate_damages(distance_damage_dict)
					self._has_extended_barrel = True
				else:
					raise Exception(f"Weapon '{self.name}' has an extended barrel value that doesn't deserialize to a bool or a dict.")

			else:
				print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing an {warning('extended barrel')} value. Using default value ({self.default_has_extended_barrel}) instead.")
				self._has_extended_barrel = self.default_has_extended_barrel
				
			if self._has_extended_barrel == True:
				potential_eb.json_content = None
				
				potential_eb._name = self.extended_barrel_weapon_name

				potential_eb._has_extended_barrel = False
				potential_eb.extended_barrel_weapon = None
				potential_eb.is_extended_barrel = True
				potential_eb.extended_barrel_parent = self
		
				potential_eb.DmgPerShotColorScaleRule = None
				
				self.extended_barrel_weapon = potential_eb

		return self._has_extended_barrel
	@property
	def has_laser(self) -> bool:
		if self._has_laser == None:
			# get laser
			if "laser" in self.json_content:
				if type(self.json_content["laser"]) != bool:
					raise Exception(f"Weapon '{self.name}' has a laser value that doesn't deserialize to a bool.")
				self._has_laser = self.json_content["laser"]
			else:
				print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing a {warning('laser')} value. Using default value ({self.default_has_laser}) instead.")
				self._has_laser = self.default_has_laser
			pass
		return self._has_laser
	@property
	def has_grip(self) -> bool:
		if self._grip == None:
			# get weapon grip
			if "grip" in self.json_content:
				if type(self.json_content["grip"]) != bool:
					raise Exception(f"Weapon '{self.name}' has a grip value that doesn't deserialize to a bool.")
				self._grip = self.json_content["grip"]
			else:
				print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing a {warning('grip')} value. Using default value ({self.default_grip}) instead.")
				self._grip = self.default_grip
			pass
		return self._grip
	@property
	def operators(self):
		return tuple([Weapon.operator_tuple[opIndex] for opIndex in self.operator_indices])

	# derived properties
	@property
	def rps(self):
		return self.rpm / 60.
	@property
	def rpms(self):
		return self.rpm / 60000.
	@property
	def ads_time_with_laser(self):
		if self.has_laser == True:
			return self.ads_time / self.laser_ads_speed_multiplier
		else:
			raise Exception(f"Weapon '{self.name}' doesn't have a laser.")
	@property
	def reload_times_with_angled_grip(self):
		if self.has_grip == True:
			rt0 = self.reload_times[0]
			rt1 = self.reload_times[1]
			if rt0 != 0:
				rt0 /= self.angled_grip_reload_speed_multiplier
			if rt1 != 0:
				rt1 /= self.angled_grip_reload_speed_multiplier
			return rt0, rt1
		else:
			raise Exception(f"Weapon '{self.name}' doesn't have a grip.")
	def damage_per_shot(self, index : int):
		return self.damages[index] * self.pellets
	def dps(self, index : int):
		return self.damages[index] * self.pellets * self.rpm / 60.
	def stdok(self, index : int, hp : int):
		return math.ceil(hp / self.damages[index] / self.pellets)
	def ttdok(self, index : int, hp : int):
		return (self.stdok(index, hp) - 1) / self.rpms
	def how_useful_is_extended_barrel(self, hp : int):
		return self.extended_barrel_parent.stdok(0, hp) - self.stdok(0, hp), self.extended_barrel_parent.ttdok(0, hp) - self.ttdok(0, hp)

	# properties with excel styles
	def getName(self):
		if self.name == self.extended_barrel_weapon_name:
			return self.name, "Normal"
		else:
			return self.name, self.getStyleBF()
	def getClass(self):
		return self.class_, self.getStyleABF()
	def getRPM(self):
		return self.rpm, self.getStyleABF()
	def getRPS(self):
		return self.rps, self.getStyleABF()
	def getRPMS(self):
		return self.rpms, self.getStyleABF()
	def getDamage(self, index : int):
		return self.damages[index], self.getDamageStyle(index)
	def getDamagePerShot(self, index : int):
		return self.damage_per_shot(index), self.getStyleAB()
	def getDPS(self, index : int):
		return int(self.dps(index) + 0.5), self.getStyleAB()
	def getSTDOK(self, index : int, hp : int):
		return self.stdok(index, hp), self.getSTDOKStyle(index, hp)
	def getTTDOK(self, index : int, hp : int):
		return int(self.ttdok(index, hp) + 0.5), self.getStyleAB()
	def getCapacity(self):
		return str(self.capacity[0]) + "+" + str(self.capacity[1]), self.getStyleABF()
	def getExtraAmmo(self):
		return self.extra_ammo, self.getStyleABF()
	def getReloadTimes(self):
		return str(self.reload_times[0]), str(self.reload_times[1]), self.getStyleABF()
	def getReloadTimesWithAngledGrip(self):
		if self.has_grip == False:
			return "", "", self.getStyleA()
		else:
			return str(self.reload_times_with_angled_grip[0]), str(self.reload_times_with_angled_grip[1]), self.getStyleABF()
	def getPellets(self):
		if self.pellets == 1:
			return "", self.getStyleA()
		else:
			return self.pellets, self.getStyleABF()
	def getADSTime(self):
		return str(self.ads_time), self.getStyleABF()
	def getADSTimeWithLaser(self):
		if self.has_laser == False:
			return "", self.getStyleA()
		else:
			return str(round(self.ads_time_with_laser, 3)), self.getStyleABF()
	def getDamageToBaseDamagePercentage(self, index : int):
		val = round(self.damages[index] / max(self.damages), 2)
		if val == 1:
			return 1, self.getStyleAB()
		else:
			return val, self.getStyleAB()
			return str(val)[1:], self.getStyleAB()
	def getHowUsefulIsExtendedBarrel(self, hp : int):
		usefulness = self.how_useful_is_extended_barrel(hp)
		if usefulness[0] == 0:
			return 0, 0, self.getStyleA()
		else:
			return usefulness[0], int(usefulness[1] + 0.5), self.getStyleABF()

	# excel styles
	def getDamageStyle(self, index : int):
		a, b = self.getDamageDropoffBorders()
		if index <= a:
			return self.getStyleABF()
		elif index >= b:
			return self.getStyleABF()
		else:
			return self.getStyleA()
	def getSTDOKStyle(self, index : int, hp : int):
		# if this is the extended barrel version of a weapon
		if self.extended_barrel_parent != None:
			# if the stdok of the extended barrel version is different from the stdok of the normal version
			if self.stdok(index, hp) != self.extended_barrel_parent.stdok(index, hp):
				return self.getStyleABF()
			
		return self.getStyleA()
		
	def getStyleABF(self):
		return self.stylesABF[self.class_]
	def getStyleAB(self):
		return self.stylesAB[self.class_]
	def getStyleBF(self):
		return self.stylesBF[self.class_]
	def getStyleA(self):
		return self.stylesA
	
	def getDmgPerShotColorScaleRule(self):
		if self.DmgPerShotColorScaleRule == None:
			start_val, end_val = self.getDmgPerShotInterval()
			end_col = background_colors[self.class_]
			self.DmgPerShotColorScaleRule = ColorScaleRule(start_type="num", start_value=start_val, start_color="FFFFFF", end_type="num", end_value=end_val, end_color=end_col)

		return self.DmgPerShotColorScaleRule
	def getDPSColorScaleRule(self):
		if self.class_ not in Weapon.DPSColorScaleRule:
			start_val, end_val = self.getDPSInterval()
			end_col = background_colors[self.class_]
			Weapon.DPSColorScaleRule[self.class_] = ColorScaleRule(start_type="num", start_value=start_val, start_color="FFFFFF", end_type="num", end_value=end_val, end_color=end_col)

		return Weapon.DPSColorScaleRule[self.class_]
	def getTTDOKColorScaleRule(self, hp : int):
		if hp not in Weapon.TTDOKColorScaleRules:
			Weapon.TTDOKColorScaleRules[hp] = {}
			
		if self.class_ not in Weapon.TTDOKColorScaleRules[hp]:
			start_val, end_val = self.getTTDOKInterval(hp)
			start_col = background_colors[self.class_]
			Weapon.TTDOKColorScaleRules[hp][self.class_] = ColorScaleRule(start_type="num", start_value=start_val, start_color=start_col, end_type="num", end_value=end_val, end_color="FFFFFF")

		return Weapon.TTDOKColorScaleRules[hp][self.class_]
	
	# interval methods for excel conditional formatting
	def getDmgPerShotInterval(self):
		return min(self.damages) * self.pellets, max(self.damages) * self.pellets
	def getDPSInterval(self):
		return Weapon.lowest_highest_dps[self.class_]
	def getTTDOKInterval(self, hp : int):
		return Weapon.lowest_highest_ttdok[hp][self.class_]

	# other methods
	def getDamageDropoffBorders(self):
		lastInitialDamageIndex = -1
		firstEndDamageIndex = -1
		
		for i in range(1, len(self.damages)):
			if self.damages[i] != self.damages[i-1]:
				lastInitialDamageIndex = i-1
				break
			
		for i in range(len(self.damages)-2, -1, -1):
			if self.damages[i] != self.damages[i+1]:
				firstEndDamageIndex = i+1
				break

		return lastInitialDamageIndex, firstEndDamageIndex
	def getTDOKBorders(self, hp):
		lastInitialSTDOKIndex = -1
		firstEndSTDOKIndex = -1

		for i in range(1, len(self.damages)):
			if self.stdok(i, hp) != self.stdok(i-1, hp):
				lastInitialSTDOKIndex = i-1
				break
			
		for i in range(len(self.damages)-2, -1, -1):
			if self.stdok(i, hp) != self.stdok(i+1, hp):
				firstEndSTDOKIndex = i+1
				break

		return lastInitialSTDOKIndex, firstEndSTDOKIndex
	def getExtendedBarrelWeapon(self):
		if self.has_extended_barrel == True:
			return self.extended_barrel_weapon
		else:
			raise Exception(f"Weapon '{self.name}' doesn't have an extended barrel version.")


def deserialize_json(file_name : str):
	with open(file_name, "r") as file:
		try:
			content = json.load(file)
		except json.JSONDecodeError:
			raise Exception(f"The json deserialization of file '{file_name}' failed.")
	return content

def get_operators(weapons : list[Weapon], file_name : str) -> None:
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

	Weapon.operator_tuple = tuple(sorted(operator_weapons.keys()))

	weapon_operatorIndex_dict : dict[str, list[int]] = {}
	for operator, weapon_list in operator_weapons.items():
		operatorIndex = Weapon.operator_tuple.index(operator)

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
	attachment_categories = deserialize_json(attachment_overview_file_name)
	Weapon.extended_barrel_damage_multiplier = 1.0 + attachment_categories["Barrels"]["Extended barrel"]["damage bonus"]
	Weapon.laser_ads_speed_multiplier = 1.0 + attachment_categories["Under Barrel"]["Laser"]["ads speed bonus"]

	weapons : list[Weapon] = []
	for file_name in os.listdir(weapon_data_dir):
		file_path = os.path.join(weapon_data_dir, file_name)

		name, extension = os.path.splitext(file_name);		
		if not extension == ".json":
			continue
		if name.startswith("_"):
			print(f"{message('Message: Excluding')} weapon '{message(name)}' because of _.")
			continue
		
		weapons.append(Weapon(deserialize_json(file_path)))
		
	# get all operator weapons
	get_operators(weapons, operators_file_name)
	
	weapons = sorted(weapons, key=lambda weapon: weapon_classes.index(weapon.class_), reverse=False)

	return weapons

def add_weapon_to_worksheet(worksheet : typing.Any, weapon : Weapon, sub_name : None | str, stat_method : typing.Any, format_method : typing.Any,
							additional_param : None | typing.Any, row : int, cond_formats : dict[typing.Any, str]):
	if sub_name != None:
		c = worksheet.cell(row=row, column=1)
		c.value = sub_name
	
	if additional_param == None:
		for col in range(2, len(Weapon.distances) + 2):
			c = worksheet.cell(row=row, column=col)
			c.value, c.style = stat_method(weapon, col - 2)
		if format_method != None:
			cond_format = format_method(weapon)
			if cond_format in cond_formats:
				cond_formats[cond_format] += f" {get_column_letter(2)}{row}:{get_column_letter(len(Weapon.distances)+2)}{row}"
			else:
				cond_formats[cond_format] = f"{get_column_letter(2)}{row}:{get_column_letter(len(Weapon.distances)+2)}{row}"
		
	else:
		for col in range(2, len(Weapon.distances) + 2):
			c = worksheet.cell(row=row, column=col)
			c.value, c.style = stat_method(weapon, col - 2, additional_param)
		if format_method != None:
			cond_format = format_method(weapon, additional_param)
			if cond_format in cond_formats:
				cond_formats[cond_format] += f" {get_column_letter(2)}{row}:{get_column_letter(len(Weapon.distances)+2)}{row}"
			else:
				cond_formats[cond_format] = f"{get_column_letter(2)}{row}:{get_column_letter(len(Weapon.distances)+2)}{row}"
		
	return

def add_worksheet_header(workbook : typing.Any, worksheet_name : str, stat_name : str, stat_link : str | None, description : str | tuple[str,...], cols_inbetween : int):
	worksheet = workbook.create_sheet(worksheet_name)
	
	worksheet.column_dimensions[get_column_letter(1)].width = 20.5
	
	row = 1
	worksheet.merge_cells(start_row=row, end_row=row, start_column=2, end_column=1 + cols_inbetween)
	c = worksheet.cell(row=row, column=2)
	c.value = "created by hanslhansl"
	c.font = Font(bold=True)

	row += 1
	worksheet.merge_cells(start_row=row, end_row=row, start_column=2, end_column=1 + cols_inbetween)
	c = worksheet.cell(row=row, column=2)
	c.value = '=HYPERLINK("https://github.com/hanslhansl/Rainbow-Six-Siege-Weapon-Statistics/", "A detailed explanation can be found here")'
	c.font = Font(color = "FF0000FF")
	
	row += 2
	worksheet.merge_cells(start_row=row, end_row=row, start_column=2, end_column=1 + cols_inbetween)
	c = worksheet.cell(row=row, column=2)
	if type(stat_link) == str:
		c.value = f'=HYPERLINK("https://github.com/hanslhansl/Rainbow-Six-Siege-Weapon-Statistics/#{stat_link}", "{stat_name}")'
		c.font = Font(color = "FF0000FF")
	else:
		c.value = stat_name
	c.font = Font(bold=True)
	
	if type(description) == str:
		description = (description, )
	for desc in description:
		row += 1
		worksheet.merge_cells(start_row=row, end_row=row, start_column=2, end_column=1 + cols_inbetween)
		c = worksheet.cell(row=row, column=2)
		c.value = desc
	
	row += 1

	return worksheet, row

def add_secondary_weapon_stats_header(worksheet : typing.Any, row : int, col : int):
	worksheet.column_dimensions[get_column_letter(col)].width = 3

	col += 1
	c = worksheet.cell(row=row, column=col)
	c.value = "Class"
	c.alignment = Weapon.alignment
	worksheet.column_dimensions[get_column_letter(col)].width = 9

	col += 1
	worksheet.column_dimensions[get_column_letter(col)].width = 3
	
	col += 1
	c = worksheet.cell(row=row, column=col)
	c.value = "RPM"
	c.alignment = Weapon.alignment
	worksheet.column_dimensions[get_column_letter(col)].width = 5

	col += 1
	c = worksheet.cell(row=row, column=col)
	c.value = "Capacity"
	c.alignment = Weapon.alignment
	worksheet.column_dimensions[get_column_letter(col)].width = 9
	
	col += 1
	c = worksheet.cell(row=row, column=col)
	c.value = "Extra ammo"
	c.alignment = Weapon.alignment
	worksheet.column_dimensions[get_column_letter(col)].width = 7

	col += 1
	c = worksheet.cell(row=row, column=col)
	c.value = "Pellets"
	c.alignment = Weapon.alignment
	worksheet.column_dimensions[get_column_letter(col)].width = 7
	
	col += 1
	worksheet.column_dimensions[get_column_letter(col)].width = 3
	
	col += 1
	c = worksheet.cell(row=row, column=col)
	c.value = "ADS time"
	c.alignment = Weapon.alignment
	worksheet.column_dimensions[get_column_letter(col)].width = 6

	col += 1
	c = worksheet.cell(row=row, column=col)
	c.value = "+ Laser"
	c.alignment = Weapon.alignment
	worksheet.column_dimensions[get_column_letter(col)].width = 8

	col += 1
	worksheet.column_dimensions[get_column_letter(col)].width = 3

	col += 1
	c = worksheet.cell(row=row, column=col)
	c.value = "Tactical reload"
	c.alignment = Weapon.alignment
	worksheet.column_dimensions[get_column_letter(col)].width = 8
	
	col += 1
	c = worksheet.cell(row=row, column=col)
	c.value = "Full reload"
	c.alignment = Weapon.alignment
	worksheet.column_dimensions[get_column_letter(col)].width = 7
	
	col += 1
	worksheet.merge_cells(start_row=row-1, end_row=row-1, start_column=col, end_column=col + 1)
	c = worksheet.cell(row=row-1, column=col)
	c.value = "+ Angled grip"
	c.alignment = Weapon.alignment
	
	c = worksheet.cell(row=row, column=col)
	c.value = "Tactical reload"
	c.alignment = Weapon.alignment
	worksheet.column_dimensions[get_column_letter(col)].width = 8
	
	col += 1
	c = worksheet.cell(row=row, column=col)
	c.value = "Full reload"
	c.alignment = Weapon.alignment
	worksheet.column_dimensions[get_column_letter(col)].width = 7

def add_secondary_weapon_stats(worksheet : typing.Any, weapon : Weapon, row : int, col : int):
	c = worksheet.cell(row=row, column=col)
	c.value, c.style = weapon.getClass()
		
	col += 2
	c = worksheet.cell(row=row, column=col)
	c.value, c.style = weapon.getRPM()
		
	col += 1
	c = worksheet.cell(row=row, column=col)
	c.value, c.style = weapon.getCapacity()

	col += 1
	c = worksheet.cell(row=row, column=col)
	c.value, c.style = weapon.getExtraAmmo()

	col += 1
	c = worksheet.cell(row=row, column=col)
	c.value, c.style = weapon.getPellets()
		
	col += 2
	c = worksheet.cell(row=row, column=col)
	c.value, c.style = weapon.getADSTime()
	
	col += 1
	c = worksheet.cell(row=row, column=col)
	c.value, c.style = weapon.getADSTimeWithLaser()

	weapon.has_grip

	col += 2
	# c1 = worksheet.cell(row=row, column=col)
	col += 1
	# c2 = worksheet.cell(row=row, column=col)
	# c1.value, c2.value, c1.style = weapon.getReloadTimes()
	# c2.style = c1.style
	
	col += 1
	# c1 = worksheet.cell(row=row, column=col)
	col += 1
	# c2 = worksheet.cell(row=row, column=col)
	# c1.value, c2.value, c1.style = weapon.getReloadTimesWithAngledGrip()
	# c2.style = c1.style

	return

def add_stats_worksheet(workbook : typing.Any, weapons : list[Weapon], worksheet_name : str, stat_name : str, stat_link : str, description : str,
						sub_names : None | tuple[str,...], stat_method : typing.Any, format_method : typing.Any, additional_params : None | tuple[typing.Any]):
	
	worksheet, row = add_worksheet_header(workbook, worksheet_name, stat_name, stat_link, description, len(Weapon.distances))
	add_secondary_weapon_stats_header(worksheet, row, len(Weapon.distances) + 2)
	worksheet.freeze_panes = worksheet.cell(row=row+1, column=2)

	col = 1
	c = worksheet.cell(row=row, column=col)
	c.value = "Distance"
	al = copy.copy(Weapon.alignment)
	al.horizontal = "left"
	c.alignment = al

	for col in range(2, len(Weapon.distances) + 2):
		c = worksheet.cell(row=row, column=col)
		c.value = Weapon.distances[col - 2]
		c.alignment = Weapon.alignment
		worksheet.column_dimensions[get_column_letter(col)].width = 4.8

	cond_formats : dict[typing.Any, str] = {}

	row += 1
	for i in range(len(weapons)):
		weapon = weapons[i]

		c = worksheet.cell(row=row, column=1)
		c.value, c.style = weapon.getName()

		add_secondary_weapon_stats(worksheet, weapon, row, len(Weapon.distances) + 3)

		if sub_names == None and additional_params == None:
			add_weapon_to_worksheet(worksheet, weapon, None, stat_method, format_method, None, row, cond_formats)
			row += 1
			
			if (weapon.has_extended_barrel):
				extended_barrel_weapon = weapon.getExtendedBarrelWeapon()
				add_weapon_to_worksheet(worksheet, extended_barrel_weapon, Weapon.extended_barrel_weapon_name, stat_method, format_method, None, row, cond_formats)
				row += 1
				
		elif type(sub_names) == tuple and type(additional_params) == tuple: 
			if len(sub_names) != len(additional_params):
				raise Exception(f"Parameters are tuples of different length.")

			worksheet.merge_cells(start_row=row, end_row=row, start_column=2, end_column=1 + len(Weapon.distances))
			row += 1

			if (weapon.has_extended_barrel):
				extended_barrel_weapon = weapon.getExtendedBarrelWeapon()
			for sub_name, additional_param in zip(sub_names, additional_params):
				add_weapon_to_worksheet(worksheet, weapon, sub_name, stat_method, format_method, additional_param, row, cond_formats)
				row += 1
				if (weapon.has_extended_barrel):
					add_weapon_to_worksheet(worksheet, extended_barrel_weapon, Weapon.extended_barrel_weapon_name, stat_method, format_method, additional_param, row, cond_formats)
					row += 1
					
		else:
			raise Exception(f"Parameters aren't both None or tuples.")
		
	for cond_format, rng in cond_formats.items():
		worksheet.conditional_formatting.add(rng, cond_format)
	
	return

def add_extended_barrel_overview(worksheet : typing.Any, weapons : list[Weapon], row : int, col : int, with_secondary_weapon_stats : bool):
	col_names = ("1 (100)", "2 (110)", "3 (125)", "1R (120)", "2R (130)", "3R (145)")
	original_col = col

	worksheet.merge_cells(start_row=row, end_row=row, start_column=col+1, end_column=col+6)
	c = worksheet.cell(row=row, column=col+1)
	c.value = "STDOK"
	c.alignment = Weapon.alignment
	
	worksheet.merge_cells(start_row=row, end_row=row, start_column=col+8, end_column=col+13)
	c = worksheet.cell(row=row, column=col+8)
	c.value = "TTDOK"
	c.alignment = Weapon.alignment
	
	row += 1
	
	c = worksheet.cell(row=row, column=col)
	c.value = "Health rating"
	worksheet.column_dimensions[get_column_letter(col)].width = 20.5

	for col_name in col_names:
		col += 1
		c = worksheet.cell(row=row, column=col)
		c.value = col_name
		c.alignment = Weapon.alignment
		worksheet.column_dimensions[get_column_letter(col)].width = 8
		
		c = worksheet.cell(row=row, column=col + len(col_names) + 1)
		c.value = col_name
		c.alignment = Weapon.alignment
		worksheet.column_dimensions[get_column_letter(col)].width = 8
		
	row += 1

	for weapon in weapons:
		if weapon.has_extended_barrel == False:
			continue
		extended_weapon = weapon.getExtendedBarrelWeapon()
		
		col = original_col
		
		c = worksheet.cell(row=row, column=col)
		c.value, c.style = weapon.getName()

		if with_secondary_weapon_stats:
			add_secondary_weapon_stats(worksheet, weapon, row, col+15)

		for hp in Weapon.tdok_hps:
			col += 1
			c1 = worksheet.cell(row=row, column=col)
			c2 = worksheet.cell(row=row, column=col + len(col_names) + 1)
			c1.value, c2.value, c1.style = extended_weapon.getHowUsefulIsExtendedBarrel(hp)
			c2.style = c1.style
			
		row += 1

	return row

def add_attachment_overview(workbook : typing.Any, weapons : list[Weapon]):
	json_content = deserialize_json(attachment_overview_file_name)
	
	if not isinstance(json_content, dict):
		raise Exception(f"File '{attachment_overview_file_name}' doesn't deserialize to a dictionary.")
	attachment_categories : dict[str, typing.Any] = json_content

	worksheet, row = add_worksheet_header(workbook, "Attachments", "Attachment overview", None, "A short overview over all available attachments.", 19)
	worksheet.freeze_panes = worksheet.cell(row=row, column=2)

	for attachment_category, attachment_dict in attachment_categories.items():
		if not isinstance(attachment_dict, dict):
			raise Exception(f"An attachment category in file '{attachment_overview_file_name}' doesn't deserialize to a dictionary but to '{type(attachment_dict)}'.")
		attachment_dict : dict[str, typing.Any]

		c = worksheet.cell(row=row, column=1)
		c.value = attachment_category
		c.font = Font(bold=True)
		
		for attachment_name, attachment in attachment_dict.items():
			if not isinstance(attachment, dict):
				raise Exception(f"An attachment in file '{attachment_overview_file_name}' doesn't deserialize to a dictionary but to '{type(attachment)}'.")

			worksheet.merge_cells(start_row=row, end_row=row, start_column=2, end_column=1 + 19)
			c = worksheet.cell(row=row, column=2)
			c.value = attachment_name
			c.font = Font(bold=True)
			row += 1
			
			if "description" in attachment:
				description = attachment.pop("description").format(*attachment.values())
				worksheet.merge_cells(start_row=row, end_row=row, start_column=2, end_column=1 + 19)
				c = worksheet.cell(row=row, column=2)
				c.value = description
				row += 1
				
			if "damage bonus" in attachment:
				row += 1
				row = add_extended_barrel_overview(worksheet, weapons, row, 2, False)
				
			row += 1
			
		row += 1

def save_to_xlsx_file(weapons : list[Weapon], stat_names : tuple[str,...], stat_links : tuple[str,...]):
	""" https://openpyxl.readthedocs.io/en/stable/ """

	sheet_names = ("Damage per bullet", "Damage per shot", "DPS", "STDOK - old", "STDOK", "TTDOK")
	explanations = (
		"The colored areas represent steady damage, the white areas represent decreasing damage.",
		"The color gradient illustrates the damage compared to the weapon's base damage.",
		"The color gradient illustrates the DPS compared to the highest DPS of the weapon's type (excluding extended barrel stats).",
		"The colored areas represent steady STDOK, the white areas represent increasing STDOK.",
		"The colored areas show where the extended barrel attachment actually affects the STDOK.",
		"The color gradient illustrates the TTDOK compared to the lowest TTDOK of the weapon's type against the same armor rating (excluding extended barrel stats).")

	sub_names = ("1 armor (100 hp)", "2 armor (110 hp)", "3 armor (125 hp)",
	  "1 armor + Rook (120 hp)", "2 armor + Rook (130 hp)", "3 armor + Rook (145 hp)")
	
	# excel file
	workbook = Workbook()

	workbook.remove(workbook.active)
		
	add_stats_worksheet(workbook, weapons, sheet_names[0], stat_names[0], stat_links[0],
		explanations[0], None, Weapon.getDamage, None, None)

	add_stats_worksheet(workbook, weapons, sheet_names[1], stat_names[1], stat_links[1],
		explanations[1], None, Weapon.getDamagePerShot, Weapon.getDmgPerShotColorScaleRule, None)

	add_stats_worksheet(workbook, weapons, sheet_names[2], stat_names[2], stat_links[2],
		explanations[2], None, Weapon.getDPS, Weapon.getDPSColorScaleRule, None)

	add_stats_worksheet(workbook, weapons, sheet_names[4], stat_names[4], stat_links[4],
		explanations[4], sub_names, Weapon.getSTDOK, None, Weapon.tdok_hps)

	add_stats_worksheet(workbook, weapons, sheet_names[5], stat_names[5], stat_links[5],
		explanations[5], sub_names, Weapon.getTTDOK, Weapon.getTTDOKColorScaleRule, Weapon.tdok_hps)
	
	add_attachment_overview(workbook, weapons)

	# save to file
	workbook.save(xlsx_output_file_name)
	
	os.system("start " + xlsx_output_file_name)
	return

def save_to_html_file(weapons : list[Weapon], stat_names : tuple[str,...]):
	# html file
	string = """<!DOCTYPE html><html lang="en"><body>"""
	string += """<script> function toggledCheckbox(event)
{
	if (event.target.checked)
		document.getElementById("row_" + event.target.id).style.visibility = 'visible';
	else
		document.getElementById("row_" + event.target.id).style.visibility = 'collapse';
}
function changedStat() {
	let values = document.getElementById('values');
	let tbody = values.children[0];
	let data = document.getElementById('data');
	let data_tbody = data.children[0];
	
	let displayed_stats_tbody = document.getElementById('displayed stat').children[0];
	
	let row0 = displayed_stats_tbody.children[0];
	let dmgPerBullet = row0.children[0].children[0];
	let dmgPerShot = row0.children[1].children[0];
	let dmgPerSecond = row0.children[2].children[0];
	let STDOK = row0.children[3].children[0];
	let TTDOK = row0.children[4].children[0];
	
	let row1 = displayed_stats_tbody.children[1];
	let hp = row1.children[3].children[0].valueAsNumber;
	//let hp = 100;
	
	for (let i = 0; i < tbody.childElementCount; i++)
	{
		let row = tbody.children[i+2];
		let data_row = data_tbody.children[i];
		let pellets = parseInt(row.children[47].textContent);
		let rpm = parseInt(row.children[45].textContent);
		for (let j = 1; j < """ + f"{len(Weapon.distances) + 1}" + """; j++)
		{
			let cell = row.children[j];
			let data_cell = data_row.children[j-1];
		
			
			if (dmgPerBullet.checked == true)
				cell.textContent = data_cell.textContent;
			else if (dmgPerShot.checked == true)
				cell.textContent = parseInt(data_cell.textContent) * parseInt(pellets);
			else if (dmgPerSecond.checked == true)
				cell.textContent = Math.round(parseFloat(data_cell.textContent) * pellets * rpm / 60.);
			else if (STDOK.checked == true)
				cell.textContent = Math.ceil(hp / parseInt(data_cell.textContent));
			else if (TTDOK.checked == true)
				cell.textContent = Math.round((Math.ceil(hp / parseInt(data_cell.textContent)) - 1) * 60000 / rpm);
			
			/*if (event.target.id == 'Damage per bullet')
				cell.textContent = data_cell.textContent;
			else if (event.target.id == 'Damage per shot')
				cell.textContent = parseInt(data_cell.textContent) * parseInt(pellets);
			else if (event.target.id == 'Damage per second')
				cell.textContent = Math.round(parseFloat(data_cell.textContent) * pellets * rpm / 60.);
			else if (event.target.id == 'Shots to down or kill')
				cell.textContent = Math.ceil(hp / parseInt(data_cell.textContent));
			else if (event.target.id == 'Time to down or kill')
				cell.textContent = Math.round((Math.ceil(hp / parseInt(data_cell.textContent)) - 1) * 60000 / rpm);*/
		}
	}
}
</script>"""

	# weapon filter
	string += """<table><tr style="vertical-align:top">"""
	current_type = ""
	for weapon in weapons:
		if current_type != weapon.class_ or weapon == weapons[0]:
			if weapon != weapons[0]:
				current_type = weapon.class_
				string += f"</fieldset></td>"
			string += f"""<td><fieldset><legend>{weapon.class_}{"" if weapon.class_ == "Else" else "s"}:</legend>"""
			
		weapon_name = weapon.name
		string += f"""<input type="checkbox" id="{weapon_name}" value="{weapon_name}" onchange="toggledCheckbox(event)"><label for="{weapon_name}">{weapon_name}</label><br>"""

		if weapon == weapons[-1]:
			string += "</fieldset></td>"
		pass
	string += """</tr></table>"""
	
	# displayed stat
	string += """<fieldset><legend>Stat:</legend><form><table id="displayed stat"><tr>"""
	for stat in stat_names:
		string += f"""<td><input type="radio" id="{stat}" name="stat" onchange="changedStat()" {"checked" if stat == stat_names[0] else ""}><label for="{stat}">{stat}</label></td>"""
	string += f"""<tr><td></td><td></td><td></td><td colspan="2" align="center"><input type="number" id="hp" value="100" onchange="changedStat()"><label for="hp"> hp</label></td></tr>"""
	string += """</tr></table></form></fieldset>"""

	# values
	string += """<table id="values">"""
	string += "<tr>" + ("<td></td>" * 51) + """<th colspan="2">Reload time</th></tr>"""
	string += "<tr><th>Distance</th>"
	for distance in Weapon.distances:
		string += f"""<th>{distance}</th>"""
	string += """<th>&emsp;</th><th>Type</th><th>&emsp;</th><th>RPM</th><th>Capacity</th><th>Pellets</th><th>&emsp;</th><th>ADS time</th><th>&emsp;</th><td>Tactical</td><td>Full</td></tr>"""
		
	for weapon in weapons:
		bg = f"background-color:#{background_colors[weapon.class_]};"
		string += f"""<tr id="row_{weapon.name}" style="visibility:collapse;"><td style="{bg}">{weapon.name}</td>"""
		for i in range(len(Weapon.distances)):
			string += f"""<td>{weapon.damages[i]}</td>"""
		string += f"""<td></td><td style="{bg}">{weapon.class_}</td><td></td><td style="{bg}">{weapon.rpm}</td><td style="{bg}">{weapon.capacity[0]}+{weapon.capacity[1]}</td>"""
		string += f"""<td style="{bg}">{weapon.pellets}</td><td></td><td style="{bg}">{weapon.ads_time}</td><td></td><td>a</td><td>b</td></tr>"""
		string += "</tr>"
	string += "</table>"

	string += """<table id="data">"""
	for weapon in weapons:
		string += f"""<tr style="display:none">"""
		for i in range(len(Weapon.distances)):
			string += f"""<td>{weapon.damages[i]}</td>"""
		string += "</tr>"
	string += "</table>"	


	string += """</body></html>"""

	with open(html_output_file_name, "w") as file:
		file.write(string)
	os.system("start " + html_output_file_name)

def save_to_output_files(weapons : list[Weapon]):
	stat_names = ("Damage per bullet", "Damage per shot", "Damage per second", "Shots to down or kill - old", "Shots to down or kill", "Time to down or kill")
	stat_links = ("damage-per-bullet", "damage-per-shot", "damage-per-second---dps", "shots-to-down-or-kill---stdok", "shots-to-down-or-kill---stdok", "time-to-down-or-kill---ttdok")
	
	#save_to_html_file(weapons, stat_names)
	save_to_xlsx_file(weapons, stat_names, stat_links)
	return


weapons = get_weapons_dict()
save_to_output_files(weapons)

input("\nCompleted!")
