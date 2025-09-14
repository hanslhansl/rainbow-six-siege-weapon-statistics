

###################################################
# settings
###################################################

# the file containing the weapons each operator has access to
operators_file_name = "operators"

# the file containing the attachment overview
attachment_overview_file_name = "attachment_overview"

# the file name of the xlsx output file
xlsx_output_file_name = "rainbow-six-siege-weapon-statistics"

# the directory containing the weapon damage files
weapon_data_dir = "weapons"

# the distance the weapon damage stats start at (usually 0)
first_distance = 0
# the distance the weapon damage stats end at (usually 40 because the of the Shooting Range limit)
last_distance = 40 

# weapon types
weapon_classes = ("AR", "SMG", "MP", "LMG", "DMR", "SR", "SG", "Slug SG", "Handgun", "Revolver", "Hand Canon")

# weapon type background colors
background_colors = {"AR":"5083EA", "SMG":"B6668E", "MP":"76A5AE", "LMG":"8771BD", "DMR":"7CB563", "SR":"DE2A00",
					 "SG":"FFBC01", "Slug SG":"A64E06", "Handgun":"A3A3A3", "Revolver":"F48020", "Hand Canon":"948A54"}


###################################################
# settings end
# don't edit from here on
###################################################

#imports
from turtle import left
import os, json, typing, math, ctypes, copy, sys, itertools, colorama, pickle, sys
from openpyxl.cell.text import InlineFont
from openpyxl.cell.rich_text import TextBlock, CellRichText
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Border, Alignment, NamedStyle, Side, Font
from openpyxl.formatting.rule import ColorScaleRule, FormulaRule, CellIsRule
from openpyxl.utils import get_column_letter

def warning(s = "Warning"):
	return f"\x1b[38;2;255;255;0m{s}\033[0m"
def message(s = "Message"):
	return f"\x1b[38;2;83;141;213m{s}\033[0m"
def error(s = "Error"):
	return f"\x1b[38;2;255;0;0m{s}\033[0m"

colorama.just_fix_windows_console()
patch_version = sys.argv[1] if len(sys.argv) > 1 else "y_s_"

operators_file_name += ".json"
attachment_overview_file_name += ".json"
xlsx_output_file_name += ".xlsx"

# check if the settings are correct
if not os.path.isfile(operators_file_name):
	raise Exception(f"{error()}: '{operators_file_name}' is not a valid file path.")
if not os.path.isdir(weapon_data_dir):
	raise Exception(f"{error()}: '{weapon_data_dir}' is not a valid directory.")
if not 0 <= first_distance:
	raise Exception(f"{error()}: 'first_distance' must be >=0 but is {first_distance}.")
if not first_distance <= last_distance:
	raise Exception(f"{error()}: 'last_distance' must be >='first_distance'={first_distance} but is {last_distance}.")

def color_to_openpyxl_color(s : str):
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

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
def lighten_color(hex_color, ratio):
    r, g, b = hex_to_rgb(hex_color)
    r = int(r + (255 - r) * (1 - ratio))
    g = int(g + (255 - g) * (1 - ratio))
    b = int(b + (255 - b) * (1 - ratio))
    return f"{r:02X}{g:02X}{b:02X}"

def normalize(min_val, max_val, value):
    # Clamp value within range
    value = max(min_val, min(max_val, value))
    
    # Normalize to [0, 1]
    top = value - min_val
    bottom = max_val - min_val
    if top == bottom:
        return 1.
    return top / bottom
def get_non_stagnant_intervals(data) -> tuple[tuple[int, int],...]:
    intervals = []
    start = None
    for i in range(len(data) - 1):
        if data[i] > data[i + 1]:
            if start is None:
                start = i
        else:
            if start is not None:
                intervals.append((start, i))
                start = None
    if start is not None:
        intervals.append((start, len(data) - 1))
    return tuple(intervals)
def is_index_in_intervals(index : int, intervals : tuple[tuple[int, int],...]):
	for start, end in intervals:
		if start <= index <= end:
			return start, end
	return None

border_style = "thin"
center_alignment = Alignment("center", wrapText=True)
left_alignment = Alignment("left", wrapText=True)

class Weapon:
	classes = weapon_classes
	distances = list(range(first_distance, last_distance+1))

	default_rpm = 0
	default_ads = 0.
	default_pellets = 1
	default_reload_times = (0., 0.)
	default_capacity = (0, 0)
	default_extra_ammo = 0
	default_has_extended_barrel = False
	default_has_grip = False
	default_has_laser = False

	extended_barrel_weapon_name = "+ extended barrel"
	extended_barrel_damage_multiplier = 0.0
	laser_ads_speed_multiplier = 0.0;
	angled_grip_reload_speed_multiplier = 0.0;
	
	with_rook = (False, False, False, True, True, True)
	hp_levels = (100, 110, 125, 120, 130, 145)

	lowest_highest_base_dps : dict[str, tuple[int, int]] = {}	# class : (lowest dps, highest dps)
	lowest_highest_base_ttdok : dict[int, dict[str, tuple[int, int]]] = {}	# hp : {class : (lowest ttdok, highest ttdok)}
	
	# excel stuff

	ex_borders = {class_ : Border(
				left=  Side(border_style=border_style, color=color_to_openpyxl_color(background_colors[class_])),
				right= Side(border_style=border_style, color=color_to_openpyxl_color(background_colors[class_])),
				top=   Side(border_style=border_style, color=color_to_openpyxl_color(background_colors[class_])),
				bottom=Side(border_style=border_style, color=color_to_openpyxl_color(background_colors[class_]))
				) for class_ in classes}
	
	ex_empty_fill = PatternFill()
	ex_fills = {class_ : PatternFill(start_color=background_colors[class_], end_color=background_colors[class_], fill_type="solid") for class_ in classes}

	ex_styles_bf = (lambda c=classes, b=ex_borders, f=ex_fills: {class_ : NamedStyle(name=class_+" BF", border=b[class_], fill=f[class_]) for class_ in c})()
	ex_style_normal = "Normal"

	def __init__(self, json_content_):
		self.json_content =  json_content_

		self.operators : list[Operator] = []
		self.DamageRule = None
		self.DmgPerShotColorScaleRule = None

		if type(self.json_content) != dict:
			raise Exception(f"{error()}: Weapon '{self.name}' doesn't deserialize to a dict.")
		
		# get weapon name
		if "name" not in self.json_content:
			raise Exception(f"{error()}: Weapon is missing its name.")
		if type(self.json_content["name"]) != str:
			raise Exception(f"{error()}: Weapon has a name that doesn't deserialize to a string.")
		self.name = self.json_content["name"]
		
		# get weapon class
		if "class" not in self.json_content:
			raise Exception(f"{error()}: Weapon '{self.name}' is missing a type.")
		if type(self.json_content["class"]) != str:
			raise Exception(f"{error()}: Weapon '{self.name}' has a type that doesn't deserialize to a string.")
		if self.json_content["class"] not in self.classes:
			raise Exception(f"{error()}: Weapon '{self.name}' has an invalid type.")
		self.class_ = self.json_content["class"]

		# get weapon fire rate
		if "rpm" in self.json_content:
			if type(self.json_content["rpm"]) != int:
				raise Exception(f"{error()}: Weapon '{self.name}' has a fire rate that doesn't deserialize to an int.")
			self.rpm = self.json_content["rpm"]
		else:
			print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing a {warning('fire rate')}. Using default value ({self.default_rpm}) instead.")
			self.rpm = self.default_rpm

		# get weapon ads time
		if "ads" in self.json_content:
			if type(self.json_content["ads"]) != float:
				raise Exception(f"{error()}: Weapon '{self.name}' has an ads time that doesn't deserialize to a float.")
			self.ads_time = self.json_content["ads"]
		else:
			print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing an {warning('ads time')}. Using default value ({self.default_ads}) instead.")
			self.ads_time = self.default_ads

		# get weapon pellet count
		if "pellets" in self.json_content:
			if type(self.json_content["pellets"]) != int:
				raise Exception(f"{error()}: Weapon '{self.name}' has a pellet count that doesn't deserialize to an integer.")
			self.pellets = self.json_content["pellets"]
		else:
			print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing a {warning('pellet count')}. Using default value ({self.default_pellets}) instead.")
			self.pellets = self.default_pellets

		# get weapon magazine capacity (magazine, chamber)
		if "capacity" in self.json_content:
			if type(self.json_content["capacity"]) != list:
				raise Exception(f"{error()}: Weapon '{self.name}' has a magazine capacity that doesn't deserialize to a list.")
			if len(self.json_content["capacity"]) != 2:
				raise Exception(f"{error()}: Weapon '{self.name}' doesn't have exactly 2 magazine capacity values.")
			if type(self.json_content["capacity"][0]) != int or type(self.json_content["capacity"][1]) != int:
				raise Exception(f"{error()}: Weapon '{self.name}' has magazine capacities that don't deserialize to integers.")
			self._capacity = (self.json_content["capacity"][0], self.json_content["capacity"][1])
		else:
			print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing the {warning('magazine capacity')}. Using default value ({self.default_capacity}) instead.")
			self._capacity = self.default_capacity

		# get extra ammo
		if "extra_ammo" in self.json_content:
			if type(self.json_content["extra_ammo"]) != int:
				raise Exception(f"{error()}: Weapon '{self.name}' has an extra ammo value that doesn't deserialize to an integer.")
			self.extra_ammo = self.json_content["extra_ammo"]
		else:
			print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing an {warning('extra ammo value')}. Using default value ({self.default_extra_ammo}) instead.")
			self.extra_ammo = self.default_extra_ammo

		# get weapon damages
		if "damages" not in self.json_content:
			raise Exception(f"{error()}: Weapon '{self.name}' is missing damage values.")
		if type(self.json_content["damages"]) != dict:
			raise Exception(f"{error()}: Weapon '{self.name}' has damage values that don't deserialize to a dict.")
		if not all(isinstance(damage, int) for damage in self.json_content["damages"].values()):
			raise Exception(f"{error()}: Weapon '{self.name}' has damage values that don't deserialize to integers.")
		distance_damage_dict = {int(distance) : int(damage) for distance, damage in self.json_content["damages"].items()}
		self.damages = self.validate_damages(distance_damage_dict)

		# get laser
		if "laser" in self.json_content:
			if type(self.json_content["laser"]) != bool:
				raise Exception(f"{error()}: Weapon '{self.name}' has a laser value that doesn't deserialize to a bool.")
			self.has_laser = self.json_content["laser"]
		else:
			print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing a {warning('laser')} value. Using default value ({self.default_has_laser}) instead.")
			self.has_laser = self.default_has_laser

		# get weapon grip
		if "grip" in self.json_content:
			if type(self.json_content["grip"]) != bool:
				raise Exception(f"{error()}: Weapon '{self.name}' has a grip value that doesn't deserialize to a bool.")
			self.has_grip = self.json_content["grip"]
		else:
			print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing a {warning('grip')} value. Using default value ({self.default_has_grip}) instead.")
			self.has_grip = self.default_has_grip

		# get weapon reload times in seconds
		# if "reload_times" in self.json_content:
		# 	if type(self.json_content["reload_times"]) != list:
		# 		raise Exception(f"{error()}: Weapon '{self.name}' has reload times that don't deserialize to a list.")
		# 	if len(self.json_content["reload_times"]) != 2:
		# 		raise Exception(f"{error()}: Weapon '{self.name}' doesn't have exactly 2 reload times.")
		# 	if type(self.json_content["reload_times"][0]) != float or type(self.json_content["reload_times"][1]) != float:
		# 		raise Exception(f"{error()}: Weapon '{self.name}' has reload times that don't deserialize to floats.")
		# 	self.reload_times = (self.json_content["reload_times"][0], self.json_content["reload_times"][1])
		# else:
		# 	print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing the {warning('reload times')}. Using default value ({self.default_reload_times}) instead.")
		# 	self.reload_times = self.default_reload_times
				

		# adjust Weapon.lowest_highest_base_dps
		DPS = tuple(int(self.dps(index=i) + 0.5) for i in range(len(Weapon.distances)))
		if self.class_ not in Weapon.lowest_highest_base_dps:
			Weapon.lowest_highest_base_dps[self.class_] = min(DPS), max(DPS)
		else:
			Weapon.lowest_highest_base_dps[self.class_] = (
				min(Weapon.lowest_highest_base_dps[self.class_][0], min(DPS)),
				max(Weapon.lowest_highest_base_dps[self.class_][1], max(DPS))
				)

		# adjust Weapon.lowest_highest_base_ttdok
		for hp in self.hp_levels:
			if hp not in Weapon.lowest_highest_base_ttdok:
				Weapon.lowest_highest_base_ttdok[hp] = {}
				
			TTDOK = tuple([int(self.ttdok(index=i, hp=hp) + 0.5) for i in range(len(Weapon.distances))])
			if self.class_ not in Weapon.lowest_highest_base_ttdok[hp]:
				Weapon.lowest_highest_base_ttdok[hp][self.class_] = min(TTDOK), max(TTDOK)
			else:
				Weapon.lowest_highest_base_ttdok[hp][self.class_] = (
					min(Weapon.lowest_highest_base_ttdok[hp][self.class_][0], min(TTDOK)),
					max(Weapon.lowest_highest_base_ttdok[hp][self.class_][1], max(TTDOK))
					)

		# get extended barrel weapon, needs to be last bc of copy()
		potential_eb = copy.copy(self)
		self.is_extended_barrel = False	# whether this is an extended barrel version
		if "extended_barrel" in self.json_content:
			if type(self.json_content["extended_barrel"]) == bool:	# use default damage multiplier for extended barrel
				self.has_extended_barrel = self.json_content["extended_barrel"]
				if self.has_extended_barrel == True:
					print(f"{warning('Warning:')} Using {warning('approximated')} extended barrel stats for weapon '{warning(self.name)}'.")
					potential_eb.damages = tuple(math.ceil(dmg * self.extended_barrel_damage_multiplier) for dmg in self.damages)

			elif type(self.json_content["extended_barrel"]) == dict:	# use custom damages for extended barrel
				if not all(isinstance(damage, int) for damage in self.json_content["extended_barrel"].values()):
					raise Exception(f"{error()}: Weapon '{self.name}' has damage values that don't deserialize to integers.")
				#print(f"{message('Message:')} Using {message('exact')} extended barrel stats for weapon '{message(self.name)}'.")
				self.has_extended_barrel = True
				potential_eb.damages = self.validate_damages({int(distance) : int(damage) for distance, damage in self.json_content["extended_barrel"].items()})
			else:
				raise Exception(f"{error()}: Weapon '{self.name}' has an extended barrel value that doesn't deserialize to a bool or a dict.")
		else:
			print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing an {warning('extended barrel')} value. Using default value ({self.default_has_extended_barrel}) instead.")
			self.has_extended_barrel = Weapon.default_has_extended_barrel
				
		if self.has_extended_barrel == True:
			potential_eb.json_content = None
			potential_eb.name = Weapon.extended_barrel_weapon_name

			potential_eb.has_extended_barrel = False
			potential_eb.extended_barrel_weapon = None
			potential_eb.is_extended_barrel = True
			potential_eb.extended_barrel_parent = self
		
			potential_eb.DmgPerShotColorScaleRule = None
				
			self.extended_barrel_weapon = potential_eb

		return

	def __repr__(self):
		return f"<Weapon: {self.name}>"

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
		if Weapon.distances != distances:
			raise Exception(f"{error()}: Weapon '{self.name}' has incorrect distance values.")

		# make sure the last damage value is given. otherwise the extrapolation will be wrong
		if damages[-1] == 0:
			raise Exception(f"{error()}: Weapon '{self.name}' is missing a damage value at {distances[-1]}m.")

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
					raise Exception(f"{error()}: Weapon '{self.name}' has a damage increase from '{previous_real_damage}' to '{damages[i]}' at {Weapon.distances[i]}m.")
				if previous_real_damage != 0 and previous_was_interpolated == True and damages[i] != previous_real_damage:
					raise Exception(f"{error()}: Tried to interpolate between two unequal damage values '{previous_real_damage}' and '{damages[i]}' at {Weapon.distances[i]}m for weapon '{self.name}'.")
				
				previous_real_damage = damages[i]
				previous_was_interpolated = False

		# get index to first non-zero damage
		first_nonzero_index = next((i for i, damage in enumerate(damages) if damage != 0), -1)
		
		if first_nonzero_index > 5:
			raise Exception(f"{error()}: First non-zero damage value for weapon '{self.name}' is at {Weapon.distances[first_nonzero_index]}m. Should be at 5m or less.")

		# extrapolate first 5 meters. damages will be continuous in [0;40]
		if first_nonzero_index == 0:
			pass	# no extrapolation needed
		elif first_nonzero_index == -1:
			raise Exception(f"{error()}: Weapon '{self.name}' has no damage values at all.")
		else:
			if self.class_ == "SG" or self.name == "Glaive-12":	# special treatment for shotgunsand glaive-12
				if first_nonzero_index <= 5:
					for i in range(first_nonzero_index):
						damages[i] = damages[first_nonzero_index]
				else:
					raise Exception(f"{error()}: Can't extrapolate first {first_nonzero_index} meters for shotgun '{self.name}'.")
			else:
				if damages[first_nonzero_index] == damages[first_nonzero_index+1] == damages[first_nonzero_index+2]:
					for i in range(first_nonzero_index):
						damages[i] = damages[first_nonzero_index]
				else:
					raise Exception(f"{error()}: Can't extrapolate first {first_nonzero_index} meters for weapon '{self.name}'.")

		# return the damage stats
		return tuple(damages)

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
			return None
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
			return None
	def how_useful_is_extended_barrel(self, hp : int):
		return self.extended_barrel_parent.stdok(0, hp) - self.stdok(0, hp), round(self.extended_barrel_parent.ttdok(0, hp) - self.ttdok(0, hp))
	@property
	def capacity(self):
		return str(self._capacity[0]) + "+" + str(self._capacity[1])
	@property
	def background_color(self):
		return background_colors[self.class_]
	@property
	def damage_drop_off_intervals(self):
		intervals = get_non_stagnant_intervals(self.damages)
		if self.class_ == "SG":
			if len(intervals) != 2:
				raise Exception(f"{error()}: A {self.class_} should have exactly 2 damage dropoff intervals but weapon '{self.name}' has {len(intervals)}.")
			return intervals
		return (intervals[0][0], intervals[-1][-1]),

	# primary stats
	def damage(self, index : int, *_):
		return self.damages[index]
	def damage_per_shot(self, index : int, *_):
		return self.damages[index] * self.pellets
	def dps(self, index : int, *_):
		return int(self.damages[index] * self.pellets * self.rps)
	def stdok(self, index : int, hp : int):
		return math.ceil(hp / self.damages[index] / self.pellets)
	def ttdok(self, index : int, hp : int):
		return (self.stdok(index, hp) - 1) / self.rpms

	# style helper methods
	def is_in_damage_drop_off(self, index : int):
		return is_index_in_intervals(index, self.damage_drop_off_intervals) is None
	def damage_to_base_damage_ratio(self, index : int):
		return self.damages[index] / self.damages[0]
		#return normalize(min(self.damages), max(self.damages), self.damages[index])
	def base_dps_class_interval(self):
		return self.lowest_highest_base_dps[self.class_]
	def dps_to_base_dps_class_ratio(self, index : int):
		min_dps, max_dps = self.base_dps_class_interval()
		return normalize(min_dps, max_dps, self.dps(index))
	def base_ttdok_hp_class_interval(self, hp : int):
		return self.lowest_highest_base_ttdok[hp][self.class_]
	def ttdok_to_base_ttdok_hp_class_ratio(self, index : int, hp : int):
		min_ttdok, max_ttdok = self.base_ttdok_hp_class_interval(hp)
		return 1 - normalize(min_ttdok, max_ttdok, self.ttdok(index, hp))

	# excel properties
	@property
	def ex_fill(self):
		return self.ex_fills[self.class_]
	@property
	def ex_style_bf(self):
		return self.ex_styles_bf[self.class_]
	@property
	def ex_border(self):
		return self.ex_borders[self.class_]

	# primary stats excel fills
	def ex_damage_drop_off_fill(self, index : int, *_):
		if self.is_in_damage_drop_off(index):
			return self.ex_fill
		return self.ex_empty_fill
	def ex_damage_to_base_damage_gradient_fill(self, index : int, *_):
		ratio = self.damage_to_base_damage_ratio(index)
		rgb = lighten_color(self.background_color, ratio)
		return PatternFill(start_color=rgb, end_color=rgb, fill_type="solid")
	def ex_dps_to_base_dps_class_gradient_fill(self, index : int, *_):
		ratio = self.dps_to_base_dps_class_ratio(index)
		rgb = lighten_color(self.background_color, ratio)
		return PatternFill(start_color=rgb, end_color=rgb, fill_type="solid")
	def ex_eb_stdok_improvement_fill(self, index : int, hp : int, *_):
		# if this is the extended barrel version of a weapon
		if self.is_extended_barrel:
			# if the stdok of the extended barrel version is different from the stdok of the normal version
			if self.stdok(index, hp) != self.extended_barrel_parent.stdok(index, hp):
				return self.ex_fill

		return self.ex_empty_fill
	def ex_ttdok_to_base_ttdok_hp_class_gradient_fill(self, index : int, hp : int, *_):
		ratio = self.ttdok_to_base_ttdok_hp_class_ratio(index, hp)
		rgb = lighten_color(self.background_color, ratio)
		return PatternFill(start_color=rgb, end_color=rgb, fill_type="solid")

	# other excel styles
	def ex_name_style(self):
		if self.name == self.extended_barrel_weapon_name:
			return self.ex_style_normal
		else:
			return self.ex_style_bf
	def ex_operators_rich_text(self):
		elements = [op.rich_text_name for op in self.operators]
		n = 1
		i = n
		while i < len(elements):
			elements.insert(i, ', ')
			i += (n+1)
		return CellRichText(*elements)
	def ex_how_useful_is_extended_barrel_fill(self, hp : int):
		if self.how_useful_is_extended_barrel(hp)[0] == 0:
			return self.ex_empty_fill
		return self.ex_fill
	
	# pandas styling
	def pd_name_color(self):
		if not self.is_extended_barrel:
			return ""
		else:
			return self.background_color


class Operator:
	attacker_color = color_to_openpyxl_color("198FEB")
	defender_color = color_to_openpyxl_color("FB3636")

	def __init__(self, json_content_, name : str, weapons : list[Weapon]):
		self.json_content =  json_content_
		self.name = name

		if "side" not in self.json_content:
			raise Exception(f"{error()}: Operator '{self.name}' is missing a side.")
		if type(self.json_content["side"]) != str:
			raise Exception(f"{error()}: Operator '{self.name}' has a side value that doesn't deserialize to a string.")
		if self.json_content["side"] not in ("A", "D"):
			raise Exception(f"{error()}: Operator '{self.name}' has an invalid side value.")
		self.side = bool(self.json_content["side"] == "D")	# False: attack, True: defense
		
		if "weapons" not in self.json_content:
			raise Exception(f"{error()}: Operator '{self.name}' is missing weapons.")
		if type(self.json_content["weapons"]) != list:
			raise Exception(f"{error()}: Operator '{self.name}' has weapons that don't deserialize to a list.")
		if not all(isinstance(weapon, str) for weapon in self.json_content["weapons"]):
			raise Exception(f"{error()}: Operator '{self.name}' has weapons that don't deserialize to strings.")
		weapons_strings = list(self.json_content["weapons"])	# tuple of weapon names
		
		weapons_strings_copy = copy.copy(weapons_strings)
		self.weapons = []
		for weapons_string in weapons_strings:
			for weapon in weapons:
				if weapon.name == weapons_string:
					self.weapons.append(weapon)
					weapon.operators.append(self)
					weapons_strings_copy.remove(weapons_string)
					break

		for fake_weapons_string in weapons_strings_copy:
			print(f"{warning('Warning:')} Weapon '{warning(fake_weapons_string)}' found on operator '{self.name}' is {warning('not an actual weapon')}.")
	
		self.rich_text_name = TextBlock(InlineFont(color=Operator.defender_color if self.side else Operator.attacker_color), self.name)

		return


def deserialize_json(file_name : str):
	with open(file_name, "r", encoding='utf-8') as file:
		try:
			content = json.load(file)
		except json.JSONDecodeError:
			raise Exception(f"{error()}: The json deserialization of file '{file_name}' failed.")
	return content

def get_operators_list(weapons : list[Weapon], file_name : str) -> None:
	json_content = deserialize_json(file_name)

	if type(json_content) != dict:
		raise Exception(f"{error()}: File '{file_name}' doesn't deserialize to a dict of operators and weapons lists.")

	if not all(isinstance(operator_name, str) for operator_name in json_content):
		raise Exception(f"{error()}: The operator names in file '{file_name}' don't deserialize to strings.")
	if not all(isinstance(op, dict) for op in json_content.values()):
		raise Exception(f"{error()}: The operators in file '{file_name}' don't deserialize to dicts.")

	operators = [Operator(js, op_name, weapons) for (op_name, js) in json_content.items()]
		
	return

def get_weapons_dict() -> dict[str, Weapon]:
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
			print(f"{message('Message: Excluding')} weapon '{message(file_name)}' because of _.")
			continue
		
		weapons.append(Weapon(deserialize_json(file_path)))
		
	# get all operator weapons
	get_operators_list(weapons, operators_file_name)
	
	weapons = sorted(weapons, key=lambda weapon: (weapon_classes.index(weapon.class_), weapon.name), reverse=False)

	return {w.name : w for w in weapons}


def add_worksheet_header(worksheet : typing.Any, stat_name : str, stat_link : str | None, description : str | tuple[str,...], row : int, cols_inbetween : int):
	
	worksheet.column_dimensions[get_column_letter(1)].width = 22
	
	def add_header_entry(row, start_column, end_column, value, font = None):
		worksheet.merge_cells(start_row=row, end_row=row, start_column=start_column, end_column=end_column)
		c = worksheet.cell(row=row, column=start_column)
		c.value = value
		if font is not None:
			c.font = font

	add_header_entry(row, 2, 1 + cols_inbetween,
	 f"created by hanslhansl, updated for {patch_version}", Font(bold=True))

	row += 1
	add_header_entry(row, 2, 6,
	 '=HYPERLINK("https://github.com/hanslhansl/Rainbow-Six-Siege-Weapon-Statistics/", "detailed explanation")', Font(color = "FF0000FF"))

	add_header_entry(row, 8, 14,
	 '=HYPERLINK("https://docs.google.com/spreadsheets/d/1QgbGALNZGLlvf6YyPLtywZnvgIHkstCwGl1tvCt875Q", "spreadsheet on google sheets")', Font(color = "FF0000FF"))

	add_header_entry(row, 16, 1 + cols_inbetween,
	 '=HYPERLINK("https://docs.google.com/spreadsheets/d/e/2PACX-1vQ1KitQsZksdVP9YPDInxK3xE2gtu1mpUxV5_PNyE8sSm-vFINdbiL8vo9RA2CRSIbIUePLVA1GCTWZ/pubhtml", "spreadsheet on google drive")', Font(color = "FF0000FF"))

	row += 2
	if type(stat_link) == str:
		add_header_entry(row, 2, 1 + cols_inbetween,
			f'=HYPERLINK("https://github.com/hanslhansl/Rainbow-Six-Siege-Weapon-Statistics/#{stat_link}", "{stat_name}")', Font(color = "FF0000FF", bold=True))
	else:
		add_header_entry(row, 2, 1 + cols_inbetween, stat_name, Font(bold=True))
	
	if type(description) == str:
		description = (description, )
	for desc in description:
		row += 1
		add_header_entry(row, 2, 1 + cols_inbetween, desc)
	
	return row

def add_weapon_to_worksheet(worksheet : typing.Any, weapon : Weapon, stat_method : typing.Any, fill_method : typing.Any,
							sub_name : None | str, row : int):
	if sub_name != None:
		c = worksheet.cell(row=row, column=1)
		c.value = sub_name

	for col in range(2, len(Weapon.distances) + 2):
		c = worksheet.cell(row=row, column=col)
		c.value = stat_method(weapon, col - 2)
		c.fill = fill_method(weapon, col - 2)
		c.alignment = center_alignment
		c.border = weapon.ex_border

	return

def add_secondary_weapon_stats_header(worksheet : typing.Any, row : int, col : int):
	
	empty = (None, 3)
	values_widths = (
		empty,
		("class", 10),
		empty,
		("rpm", 6),
		("capacity", 10),
		("ammo", 8),
		("pellets", 8),
		empty,
		("ads", 6),
		("+ laser", 9),
		empty,
		("full reload", 11),
		("tactical", 8),
		("+ angled grip", 7),
		(None, 7),
		empty,
		("operators", 50)
	)

	for value, width in values_widths:
		if value != None:
			c = worksheet.cell(row=row, column=col)
			c.value = value
			c.alignment = center_alignment
		worksheet.column_dimensions[get_column_letter(col)].width = width

		col += 1

	worksheet.cell(row=row, column=col-1).alignment = left_alignment

	worksheet.merge_cells(start_row=row, end_row=row, start_column=col-4, end_column=col-4+1)

	return row

def add_secondary_weapon_stats(worksheet : typing.Any, weapon : Weapon, row : int, col : int):
	values = [weapon.class_, weapon.rpm, weapon.capacity, weapon.extra_ammo, weapon.pellets, weapon.ads_time, weapon.ads_time_with_laser,
		  #Weapon.getReload, Weapon.getReloadTimesWithAngledGrip
		  ]
	skips = [2, 1, 1, 1, 2, 1, 7]

	for value, skip in zip(values, skips):
		c = worksheet.cell(row=row, column=col)
		
		if value != None:
			if type(value) == float:
				value = str(round(value, 3))
			c.value = value
			c.style = weapon.ex_style_bf
			c.alignment = center_alignment
			weapon.ex_border
		col += skip

	c1 = worksheet.cell(row=row, column=col)
	c1.value = weapon.ex_operators_rich_text()

	return

def add_stats_worksheet(workbook : typing.Any, weapons : list[Weapon], worksheet_name : str, stat_name : str, stat_link : str,
						description : str, stat_method : typing.Any, fill_method : typing.Any,
						additional_params : tuple[tuple[str, typing.Any],...] = ((None, None), )):

	worksheet = workbook.create_sheet(worksheet_name)
	row = 1
	col = 1

	c = worksheet.cell(row=row, column=col)
	c.value = "weapon"
	worksheet.column_dimensions[get_column_letter(col)].width = 22
	col += 1

	for d in Weapon.distances:
		c = worksheet.cell(row=row, column=col)
		c.value = d
		c.alignment = center_alignment
		worksheet.column_dimensions[get_column_letter(col)].width = 4.8
		col += 1

	row = add_secondary_weapon_stats_header(worksheet, row, 2+len(Weapon.distances))
	worksheet.freeze_panes = worksheet.cell(row=row+1, column=2)
	row += 2

	row = add_worksheet_header(worksheet, stat_name, stat_link, description, row, len(Weapon.distances))
	row += 2

	for i, weapon in enumerate(weapons):

		c = worksheet.cell(row=row, column=1)
		c.value = weapon.name
		c.style = weapon.ex_name_style()

		add_secondary_weapon_stats(worksheet, weapon, row, len(Weapon.distances) + 3)

		if len(additional_params) != 1:
			worksheet.merge_cells(start_row=row, end_row=row, start_column=2, end_column=1 + len(Weapon.distances))
			row += 1

		if False: # eb in between: 100hp, eb, 110hp, eb, 125hp, eb, 120hp, eb, 130hp, eb, 145hp
			skip = 2
			inline_skip = 1
			final_skip = 0
		else: # eb afterwards: 100hp, 110hp, 125hp, 120hp, 130hp, 145hp, eb, eb, eb, eb, eb, eb
			skip = 1
			inline_skip = len(additional_params)
			final_skip = len(additional_params)

		for sub_name, additional_param in additional_params:
			bound_stat_method = lambda w, i, arg=additional_param: stat_method(w, i, arg)
			bound_fill_method = lambda w, i, arg=additional_param: fill_method(w, i, arg)
				
			add_weapon_to_worksheet(worksheet, weapon, bound_stat_method, bound_fill_method, sub_name, row)
				
			if (weapon.has_extended_barrel):
				add_weapon_to_worksheet(
					worksheet, weapon.extended_barrel_weapon, bound_stat_method, bound_fill_method,
					Weapon.extended_barrel_weapon_name, row+inline_skip
					)
				
			row += skip

		if weapon.has_extended_barrel: row += final_skip
		
	return

def add_extended_barrel_overview(worksheet : typing.Any, weapons : list[Weapon], row : int, col : int, with_secondary_weapon_stats : bool):
	col_names = ("1 (100)", "2 (110)", "3 (125)", "1R (120)", "2R (130)", "3R (145)")
	original_col = col

	worksheet.merge_cells(start_row=row, end_row=row, start_column=col+1, end_column=+len(col_names))
	c = worksheet.cell(row=row, column=col+1)
	c.value = "stdok"
	c.alignment = center_alignment

	worksheet.column_dimensions[get_column_letter(col+len(col_names)+1)].width = 5
	
	worksheet.merge_cells(start_row=row, end_row=row, start_column=col+len(col_names)+2, end_column=col+len(col_names)*2+1)
	c = worksheet.cell(row=row, column=col+8)
	c.value = "ttdok"
	c.alignment = center_alignment
	
	row += 1
	
	c = worksheet.cell(row=row, column=col)
	c.value = "weapon"
	worksheet.column_dimensions[get_column_letter(col)].width = 22

	for col_name in col_names:
		col += 1
		c = worksheet.cell(row=row, column=col)
		c.value = col_name
		c.alignment = center_alignment
		worksheet.column_dimensions[get_column_letter(col)].width = 9
		
		c = worksheet.cell(row=row, column=col + len(col_names) + 1)
		c.value = col_name
		c.alignment = center_alignment
		worksheet.column_dimensions[get_column_letter(col)].width = 9
	col += len(col_names) + 2
		
	if with_secondary_weapon_stats:
		add_secondary_weapon_stats_header(worksheet, row, col)

	row += 1

	for weapon in weapons:
		if weapon.has_extended_barrel == False:
			continue

		col = original_col
		
		c = worksheet.cell(row=row, column=col)
		c.value = weapon.name
		c.style = weapon.ex_name_style()

		if with_secondary_weapon_stats:
			add_secondary_weapon_stats(worksheet, weapon, row, col+15)

		for hp in Weapon.hp_levels:
			col += 1
			c1 = worksheet.cell(row=row, column=col)
			c2 = worksheet.cell(row=row, column=col + len(col_names) + 1)
			c1.value, c2.value = weapon.extended_barrel_weapon.how_useful_is_extended_barrel(hp)
			fill = weapon.extended_barrel_weapon.ex_how_useful_is_extended_barrel_fill(hp)
			c1.fill = c2.fill = fill
			c1.alignment = c2.alignment = center_alignment
			c1.border = c2.border = weapon.ex_border
			
		row += 1

	return row

def add_attachment_overview(workbook : typing.Any, weapons : list[Weapon]):
	json_content = deserialize_json(attachment_overview_file_name)
	
	if not isinstance(json_content, dict):
		raise Exception(f"{error()}: File '{attachment_overview_file_name}' doesn't deserialize to a dictionary.")
	attachment_categories : dict[str, typing.Any] = json_content

	worksheet = workbook.create_sheet("Attachments")
	row = add_worksheet_header(worksheet, "Attachment overview", None, "A short overview over all available attachments.", 1, 19)
	#worksheet.freeze_panes = worksheet.cell(row=row, column=2)

	for attachment_category, attachment_dict in attachment_categories.items():
		if not isinstance(attachment_dict, dict):
			raise Exception(f"{error()}: An attachment category in file '{attachment_overview_file_name}' doesn't deserialize to a dictionary but to '{type(attachment_dict)}'.")
		attachment_dict : dict[str, typing.Any]

		c = worksheet.cell(row=row, column=1)
		c.value = attachment_category
		c.font = Font(bold=True)
		
		for attachment_name, attachment in attachment_dict.items():
			if not isinstance(attachment, dict):
				raise Exception(f"{error()}: An attachment in file '{attachment_overview_file_name}' doesn't deserialize to a dictionary but to '{type(attachment)}'.")

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
				row = add_extended_barrel_overview(worksheet, weapons, row, 2, True)
				
			row += 1
			
		row += 1


def save_to_xlsx_file(weapons : list[Weapon], stat_names : tuple[str,...], stat_links : tuple[str,...]):
	""" https://openpyxl.readthedocs.io/en/stable/ """

	sheet_names = ("damage per bullet", "damage per shot", "dps", "stdok", "ttdok")
	explanations = (
		"the colored areas represent steady damage, the white areas represent decreasing damage",
		"the color gradient illustrates the damage compared to the weapon's base damage",
		"the color gradient illustrates the dps compared to the highest dps of the weapon's type (excluding extended barrel stats)",
		"the colored areas show where the extended barrel attachment actually affects the stdok",
		"the color gradient illustrates the ttdok compared to the lowest ttdok of the weapon's type against the same armor rating (excluding extended barrel stats)")

	
	tdok_additional_params = [(f"{int(i%3)+1} armor {'+ Rook ' if with_rook else ''}({hp} hp)", hp) for i, (hp, with_rook) in enumerate(zip(Weapon.hp_levels, Weapon.with_rook))]

	# excel file
	workbook = Workbook()

	workbook.remove(workbook.active)
		
	add_stats_worksheet(workbook, weapons, sheet_names[0], stat_names[0], stat_links[0], explanations[0],
					 Weapon.damage, Weapon.ex_damage_drop_off_fill)

	add_stats_worksheet(workbook, weapons, sheet_names[1], stat_names[1], stat_links[1], explanations[1], 
					 Weapon.damage_per_shot, Weapon.ex_damage_to_base_damage_gradient_fill)

	add_stats_worksheet(workbook, weapons, sheet_names[2], stat_names[2], stat_links[2], explanations[2],
					 Weapon.dps, Weapon.ex_dps_to_base_dps_class_gradient_fill)

	add_stats_worksheet(workbook, weapons, sheet_names[3], stat_names[3], stat_links[3], explanations[3],
					 Weapon.stdok, Weapon.ex_eb_stdok_improvement_fill, tdok_additional_params)

	add_stats_worksheet(workbook, weapons, sheet_names[4], stat_names[4], stat_links[4], explanations[4],
					 Weapon.ttdok, Weapon.ex_ttdok_to_base_ttdok_hp_class_gradient_fill, tdok_additional_params)
	
	add_attachment_overview(workbook, weapons)

	# save to file
	workbook.save(xlsx_output_file_name)
	
	os.system("start " + xlsx_output_file_name)
	return

def save_to_output_files(weapons : list[Weapon]):
	stat_names = ("damage per bullet", "damage per shot", "damage per second", "shots to down or kill", "time to down or kill")
	stat_links = ("damage-per-bullet", "damage-per-shot", "damage-per-second---dps", "shots-to-down-or-kill---stdok", "time-to-down-or-kill---ttdok")
	
	#save_to_html_file(weapons, stat_names)
	save_to_xlsx_file(weapons, stat_names, stat_links)
	return

if __name__ == "__main__":
	# get all weapons from the files
	weapons = list(get_weapons_dict().values())

	# verify
	# group weapons by class and by base damage
	weapons_sorted = sorted(weapons.copy(), key=lambda obj: (obj.class_, obj.damages[0]))
	grouped = [list(group) for key, group in itertools.groupby(weapons_sorted, key=lambda o: (o.class_, o.damages[0]))]
	# find all weapons with the same base damage but different damage drop-off
	failed = False
	for group in grouped:
		if len(group) > 1:
			for i, distance in enumerate(Weapon.distances):
				if len(set(weapon.damages[i] for weapon in group)) > 1:
					print(f"{warning()}: These {group[0].class_}s have the {warning('same base damage')} ({group[0].damages[0]}) but {warning('different damages')} at {distance}m:")
					for weapon in group:
						print(f"{weapon.name}: {weapon.damages[i]}")
					failed = True
	if failed: raise Exception(f"{error()}: See above warnings.")


	# save to excel file
	save_to_output_files(weapons)
	#input("Completed!")

