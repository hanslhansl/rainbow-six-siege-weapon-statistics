

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

# weapon type background colors
weapon_colors = {"AR":"#5083EA",
				 "SMG":"#B6668E",
				 "MP":"#76A5AE",
				 "LMG":"#8771BD",
				 "DMR":"#7CB563",
				 "SR":"#DE2A00",
				 "SG":"#FFBC01",
				 "Slug SG":"#A64E06",
				 "Handgun":"#A3A3A3",
				 "Revolver":"#F48020",
				 "Hand Canon":"#948A54"}

weapon_colors_ = {"AR":"#1f77b4",
				 "SMG":"#e377c2",
				 "MP":"#1abc9c",
				 "LMG":"#9467bd",
				 "DMR":"#2ca02c",
				 "SR":"#d62728",
				 "SG":"#17becf",
				 "Slug SG":"#8c564b",
				 "Handgun":"#7f7f7f",
				 "Revolver":"#ff7f0e",
				 "Hand Canon":"#bcbd22"}

###################################################
# settings end
# don't edit from here on
###################################################

#imports
import os, json, typing, math, ctypes, copy, sys, itertools, colorama, sys, colorsys, pandas as pd, numpy as np, io
import openpyxl
import openpyxl.workbook.workbook
from openpyxl.cell.text import InlineFont
from openpyxl.cell.rich_text import TextBlock, CellRichText
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Border, Alignment, NamedStyle, Side, Font
from openpyxl.utils import get_column_letter
from dataclasses import dataclass

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

github_link = "https://github.com/hanslhansl/Rainbow-Six-Siege-Weapon-Statistics/"
google_sheets_link = "https://docs.google.com/spreadsheets/d/1QgbGALNZGLlvf6YyPLtywZnvgIHkstCwGl1tvCt875Q"
google_drive_link = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ1KitQsZksdVP9YPDInxK3xE2gtu1mpUxV5_PNyE8sSm-vFINdbiL8vo9RA2CRSIbIUePLVA1GCTWZ/pubhtml"

tdok_hp_levels = (100, 110, 125, 120, 130, 145)
tdok_with_rook = (False, False, False, True, True, True)
tdok_levels_descriptions = tuple(f"{int(i%3)+1} armor {'+ Rook ' if with_rook else ''}({hp} hp)"
								 for i, (hp, with_rook) in enumerate(zip(tdok_hp_levels, tdok_with_rook)))
tdok_levels_descriptions_short = tuple(f"{int(i%3)+1}{'R' if with_rook else ''} ({hp})"
								 for i, (hp, with_rook) in enumerate(zip(tdok_hp_levels, tdok_with_rook)))

# check if the settings are correct
if not os.path.isfile(operators_file_name):
	raise Exception(f"{error()}: '{operators_file_name}' is not a valid file path.")
if not os.path.isdir(weapon_data_dir):
	raise Exception(f"{error()}: '{weapon_data_dir}' is not a valid directory.")
if not 0 <= first_distance:
	raise Exception(f"{error()}: 'first_distance' must be >=0 but is {first_distance}.")
if not first_distance <= last_distance:
	raise Exception(f"{error()}: 'last_distance' must be >='first_distance'={first_distance} but is {last_distance}.")

def deserialize_json(file_name : str):
	with open(file_name, "r", encoding='utf-8') as file:
		try:
			content = json.load(file)
		except json.JSONDecodeError:
			raise Exception(f"{error()}: The json deserialization of file '{file_name}' failed.")
	return content

class RGBA:
	def __init__(self, r : int, g : int, b : int, a : float):
		self.r = r
		self.g = g
		self.b = b
		self.a = a

	@classmethod
	def from_rgb_hex(cls, hex_str : str):
		hex_str = hex_str.lstrip('#')
		if len(hex_str) != 6:
			raise ValueError("Hex string must be 6 characters long.")
		r = int(hex_str[0:2], 16)
		g = int(hex_str[2:4], 16)
		b = int(hex_str[4:6], 16)
		a = 1.0

		return cls(r, g, b, a)
		
	def to_rgb_hex(self, with_hastag = True):
		r = int(self.r + (255 - self.r) * (1 - self.a))
		g = int(self.g + (255 - self.g) * (1 - self.a))
		b = int(self.b + (255 - self.b) * (1 - self.a))
		return f"{'#' if with_hastag else ''}{r:02X}{g:02X}{b:02X}"

	def to_css(self):
		return f"#{self.r:02X}{self.g:02X}{self.b:02X}{int(self.a*0xFF):02X}"
		return f"rgba({self.r}, {self.g}, {self.b}, {self.a})"

	def with_alpha(self, a : float):
		return RGBA(self.r, self.g, self.b, a)

	def to_ex_fill(self):
		rgb = self.to_rgb_hex(False)
		return PatternFill(start_color=rgb, end_color=rgb, fill_type="solid")

	def to_border_color(self, factor=0.8):
		# RGB to HLS
		h, l, s = colorsys.rgb_to_hls(self.r/255, self.g/255, self.b/255)
		l = max(0, min(1, l * factor))  # Helligkeit anpassen
		r_new, g_new, b_new = colorsys.hls_to_rgb(h, l, s)
		
		return RGBA(int(r_new*255), int(g_new*255), int(b_new*255), 1.)

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

def safe_division(a : float, b : float):
	if a == b:
		return 1.
	return a / b

border_style = "thin"
center_alignment = Alignment("center", wrapText=True)
left_alignment = Alignment("left", wrapText=True)

attachment_overview_json = deserialize_json(attachment_overview_file_name)

class Weapons:
	def __init__(self):
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
		
			w = Weapon(deserialize_json(file_path))
			weapons.append(w)

			"""# adjust Weapon.lowest_highest_base_dps
			DPS = tuple(int(self.dps(index=i) + 0.5) for i in range(len(Weapon.distances)))
			if self.class_ not in Weapon.lowest_highest_base_dps:
				Weapon.lowest_highest_base_dps[self.class_] = min(DPS), max(DPS)
			else:
				Weapon.lowest_highest_base_dps[self.class_] = (
					min(Weapon.lowest_highest_base_dps[self.class_][0], min(DPS)),
					max(Weapon.lowest_highest_base_dps[self.class_][1], max(DPS))
					)

			# adjust Weapon.lowest_highest_base_ttdok
			for hp in tdok_hp_levels:
				if hp not in Weapon.lowest_highest_base_ttdok:
					Weapon.lowest_highest_base_ttdok[hp] = {}
				
				TTDOK = tuple([int(self.ttdok(index=i, hp=hp) + 0.5) for i in range(len(Weapon.distances))])
				if self.class_ not in Weapon.lowest_highest_base_ttdok[hp]:
					Weapon.lowest_highest_base_ttdok[hp][self.class_] = min(TTDOK), max(TTDOK)
				else:
					Weapon.lowest_highest_base_ttdok[hp][self.class_] = (
						min(Weapon.lowest_highest_base_ttdok[hp][self.class_][0], min(TTDOK)),
						max(Weapon.lowest_highest_base_ttdok[hp][self.class_][1], max(TTDOK))
						)"""

		
		# get all operator weapons
		get_operators_list(weapons, operators_file_name)

		# add eb weapons
		weapons += (w.extended_barrel_weapon for w in weapons if w.extended_barrel_weapon)
	
		weapons_sorted = sorted(weapons, key=lambda w: (Weapon.classes.index(w.class_), w.name))
		self.weapons = {w.name : w for w in weapons_sorted}

		self._damages = pd.DataFrame({name : w.damages for name, w in self.weapons.items()}).transpose()
		self._damages.index.rename("weapons", inplace=True)
		for name, w in self.weapons.items():
			del w.damages

		return

	def filter(self, df : pd.DataFrame, filter_func : typing.Callable[["Weapon"], bool]):
		return df[df.apply(lambda row: filter_func(self.weapons[row.name]), axis=1)]
	def apply(self, df : pd.DataFrame, callback : typing.Callable[["Weapon", int], typing.Any]):
		def cb(x):
			w = self.weapons[x.name]
			return pd.Series(callback(w, i) for i, v in x.items())
		return df.apply(cb, axis=1)
	def apply_style(self, df : pd.DataFrame, callback : typing.Callable[["Weapon", int], str]):
		def cb(x):
			w = self.weapons[x.name]
			return pd.Series(callback(w, i) for i in df.columns)
		return df.style.apply(cb, axis=1)
	def apply_background_color(self, df : pd.DataFrame, callback : typing.Callable[["Weapon", int], RGBA]):
		convert_color = RGBA.to_rgb_hex if __name__ == "__main__" else RGBA.to_css
		return self.apply_style(df, lambda *args: f"background-color: {convert_color(callback(*args))}")

	# stats helper
	def is_in_damage_drop_off(self, w : "Weapon", index : int):
		return is_index_in_intervals(index, w.damage_drop_off_intervals) is None

	# primary stats
	def damage_per_bullet(self, _ = None):
		return self._damages
	def damage_per_shot(self, _ = None):
		pellets = {name : w.pellets for name, w in self.weapons.items()}
		return self._damages.mul(pellets, axis=0)
	def dps(self, _ = None):
		"""damage per second"""
		bullets_per_second = {name : w.pellets * w.rps for name, w in self.weapons.items()}
		return self._damages.mul(bullets_per_second, axis=0).round()
	
	def btdok(self, hp : int):
		"""bullets to down or kill"""
		return np.ceil(hp / self._damages)
	def stdok(self, hp : int):
		"""shots to down or kill"""
		pellets = {name : w.pellets for name, w in self.weapons.items()}
		return np.ceil((hp / self._damages).div(pellets, axis=0))
	def ttdok(self, hp : int):
		"""time to down or kill"""
		rpms = {name : w.rpms for name, w in self.weapons.items()}
		return (self.stdok(hp) - 1).div(rpms, axis=0).round()
	
	def theoretical_btdok(self, hp : int):
		""""""
		return hp / self._damages
	def theoretical_stdok(self, hp : int):
		""""""
		pellets = {name : w.pellets for name, w in self.weapons.items()}
		return (hp / self._damages).div(pellets, axis=0)
	def theoretical_ttdok(self, hp : int):
		""""""
		pellets_rpms = {name : w.pellets * w.rpms for name, w in self.weapons.items()}
		return (hp / self._damages).div(pellets_rpms, axis=0)

	def btk(self, hp : int):
		"""bullets to kill"""
		return self.btdok(hp + 20)
	def stk(self, hp : int):
		"""shots to kill"""
		return self.stdok(hp + 20)
	def ttk(self, hp : int):
		"""time to kill"""
		return self.ttdok(hp + 20)
	
	def how_useful_is_extended_barrel(self, hp : int):
		a = self.extended_barrel_parent.stdok(0, hp) - self.stdok(0, hp)
		b = round(self.extended_barrel_parent.ttdok(0, hp) - self.ttdok(0, hp))
		if a == 0:
			return a, b, self.empty_color
		return a, b, self.color

	# illustrations helper
	def eb_or_parent_weapon(self, w : "Weapon", consider_eb : bool):
		return w if w.is_extended_barrel <= consider_eb else w.extended_barrel_parent

	# illustrations
	def damage_drop_off_coloring(self, target : pd.DataFrame, consider_eb : bool, source : pd.DataFrame = None):
		"""the colored areas represent steady damage, the colorless areas represent decreasing damage"""
		return self.apply_background_color(target, lambda w, i: w.color if
									 self.is_in_damage_drop_off(self.eb_or_parent_weapon(w, consider_eb), i) else w.empty_color)
	def stat_to_base_stat_gradient_coloring(self, target : pd.DataFrame, consider_eb : bool, source : pd.DataFrame = None):
		"""the color gradient illustrates the stat compared to the weapon's base stat (i.e. at 0 m)"""
		return self.apply_background_color(target, lambda w, i: w.color.with_alpha(safe_division(df.loc[w.name][i], df.loc[self.eb_or_parent_weapon(w, consider_eb).name].max())))
	def stat_to_class_stat_gradient_coloring(self, target : pd.DataFrame, consider_eb : bool, source : pd.DataFrame = None):
		"""the color gradient illustrates the stat compared to the weapon class' highest stat at the same distance"""
		df2 = df if consider_eb else self.filter(df, lambda w: not w.is_extended_barrel)
		class_max = {class_ : group_df.max(axis=0) for class_, group_df in df2.groupby(lambda name: self.weapons[name].class_)}
		return self.apply_background_color(target, lambda w, i: w.color.with_alpha(safe_division(df.loc[w.name][i], class_max[w.class_][i])))
	def stat_to_class_base_stat_gradient_coloring(self, target : pd.DataFrame, consider_eb : bool, source : pd.DataFrame = None):
		"""the color gradient illustrates the stat compared to the weapon's class' highest base stat (i.e. at 0 m)"""
		df2 = df if consider_eb else self.filter(df, lambda w: not w.is_extended_barrel)
		class_base_max = {class_ : group_df.to_numpy().max() for class_, group_df in df2.groupby(lambda name: self.weapons[name].class_)}
		return self.apply_background_color(target, lambda w, i: w.color.with_alpha(safe_division(df.loc[w.name][i], class_base_max[w.class_])))
	def extended_barrel_improvement_coloring(self, target : pd.DataFrame, consider_eb : bool, source : pd.DataFrame = None):
		"""the colored areas show where the extended barrel attachment actually affects the stat"""
		return self.apply_background_color(target, lambda w, i:
									 w.color if
									 w.is_extended_barrel and consider_eb and df.loc[w.name][i] != df.loc[w.extended_barrel_parent.name][i]
									 else w.empty_color)

class Weapon:
	colors = {class_: RGBA.from_rgb_hex(color) for class_, color in weapon_colors.items()}
	classes = tuple(colors)
	distances = list(range(first_distance, last_distance+1))

	# excel stuff
	ex_borders = {class_ : Border(
				left=  Side(border_style=border_style, color=color.to_border_color().to_rgb_hex(False)),
				right= Side(border_style=border_style, color=color.to_border_color().to_rgb_hex(False)),
				top=   Side(border_style=border_style, color=color.to_border_color().to_rgb_hex(False)),
				bottom=Side(border_style=border_style, color=color.to_border_color().to_rgb_hex(False))
				) for class_, color in colors.items()}

	default_rpm = 0
	default_ads = 0.
	default_pellets = 1
	default_reload_times = (0., 0.)
	default_capacity = (0, 0)
	default_extra_ammo = 0
	default_has_extended_barrel = False
	default_has_grip = False
	default_has_laser = False

	extended_barrel_damage_multiplier = 0.0
	laser_ads_speed_multiplier = 0.0
	angled_grip_reload_speed_multiplier = 0.0

	# lowest_highest_base_dps : dict[str, tuple[int, int]] = {}				# class : (lowest dps, highest dps)
	# lowest_highest_base_ttdok : dict[int, dict[str, tuple[int, int]]] = {}	# hp : {class : (lowest ttdok, highest ttdok)}
	
	empty_color = RGBA(0,0,0,0)

	def __init__(self, json_content):
		self.operators : list[Operator] = []

		if type(json_content) != dict:
			raise Exception(f"{error()}: Weapon '{self.name}' doesn't deserialize to a dict.")
		
		# get weapon name
		if "name" not in json_content:
			raise Exception(f"{error()}: Weapon is missing its name.")
		if type(json_content["name"]) != str:
			raise Exception(f"{error()}: Weapon has a name that doesn't deserialize to a string.")
		self.name = json_content["name"]
		
		# get weapon class
		if "class" not in json_content:
			raise Exception(f"{error()}: Weapon '{self.name}' is missing a type.")
		if type(json_content["class"]) != str:
			raise Exception(f"{error()}: Weapon '{self.name}' has a type that doesn't deserialize to a string.")
		if json_content["class"] not in self.classes:
			raise Exception(f"{error()}: Weapon '{self.name}' has an invalid type.")
		self.class_ = json_content["class"]

		# get weapon fire rate
		if "rpm" in json_content:
			if type(json_content["rpm"]) != int:
				raise Exception(f"{error()}: Weapon '{self.name}' has a fire rate that doesn't deserialize to an int.")
			self.rpm = json_content["rpm"]
		else:
			print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing a {warning('fire rate')}. Using default value ({self.default_rpm}) instead.")
			self.rpm = self.default_rpm

		# get weapon ads time
		if "ads" in json_content:
			if type(json_content["ads"]) != float:
				raise Exception(f"{error()}: Weapon '{self.name}' has an ads time that doesn't deserialize to a float.")
			self.ads_time = json_content["ads"]
		else:
			print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing an {warning('ads time')}. Using default value ({self.default_ads}) instead.")
			self.ads_time = self.default_ads

		# get weapon pellet count
		if "pellets" in json_content:
			if type(json_content["pellets"]) != int:
				raise Exception(f"{error()}: Weapon '{self.name}' has a pellet count that doesn't deserialize to an integer.")
			self.pellets = json_content["pellets"]
		else:
			print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing a {warning('pellet count')}. Using default value ({self.default_pellets}) instead.")
			self.pellets = self.default_pellets

		# get weapon magazine capacity (magazine, chamber)
		if "capacity" in json_content:
			if type(json_content["capacity"]) != list:
				raise Exception(f"{error()}: Weapon '{self.name}' has a magazine capacity that doesn't deserialize to a list.")
			if len(json_content["capacity"]) != 2:
				raise Exception(f"{error()}: Weapon '{self.name}' doesn't have exactly 2 magazine capacity values.")
			if type(json_content["capacity"][0]) != int or type(json_content["capacity"][1]) != int:
				raise Exception(f"{error()}: Weapon '{self.name}' has magazine capacities that don't deserialize to integers.")
			self._capacity = (json_content["capacity"][0], json_content["capacity"][1])
		else:
			print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing the {warning('magazine capacity')}. Using default value ({self.default_capacity}) instead.")
			self._capacity = self.default_capacity

		# get extra ammo
		if "extra_ammo" in json_content:
			if type(json_content["extra_ammo"]) != int:
				raise Exception(f"{error()}: Weapon '{self.name}' has an extra ammo value that doesn't deserialize to an integer.")
			self.extra_ammo = json_content["extra_ammo"]
		else:
			print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing an {warning('extra ammo value')}. Using default value ({self.default_extra_ammo}) instead.")
			self.extra_ammo = self.default_extra_ammo

		# get weapon damages
		if "damages" not in json_content:
			raise Exception(f"{error()}: Weapon '{self.name}' is missing damage values.")
		if type(json_content["damages"]) != dict:
			raise Exception(f"{error()}: Weapon '{self.name}' has damage values that don't deserialize to a dict.")
		if not all(isinstance(damage, int) for damage in json_content["damages"].values()):
			raise Exception(f"{error()}: Weapon '{self.name}' has damage values that don't deserialize to integers.")
		distance_damage_dict = {int(distance) : int(damage) for distance, damage in json_content["damages"].items()}
		#self.damages = self.validate_damages(distance_damage_dict)
		setattr(self, "damages", self.validate_damages(distance_damage_dict))

		self.damage_drop_off_intervals = self.get_damage_drop_off_intervals()

		# get laser
		if "laser" in json_content:
			if type(json_content["laser"]) != bool:
				raise Exception(f"{error()}: Weapon '{self.name}' has a laser value that doesn't deserialize to a bool.")
			self.has_laser = json_content["laser"]
		else:
			print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing a {warning('laser')} value. Using default value ({self.default_has_laser}) instead.")
			self.has_laser = self.default_has_laser

		# get weapon grip
		if "grip" in json_content:
			if type(json_content["grip"]) != bool:
				raise Exception(f"{error()}: Weapon '{self.name}' has a grip value that doesn't deserialize to a bool.")
			self.has_grip = json_content["grip"]
		else:
			print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing a {warning('grip')} value. Using default value ({self.default_has_grip}) instead.")
			self.has_grip = self.default_has_grip

		# get weapon reload times in seconds
		# if "reload_times" in json_content:
		# 	if type(json_content["reload_times"]) != list:
		# 		raise Exception(f"{error()}: Weapon '{self.name}' has reload times that don't deserialize to a list.")
		# 	if len(json_content["reload_times"]) != 2:
		# 		raise Exception(f"{error()}: Weapon '{self.name}' doesn't have exactly 2 reload times.")
		# 	if type(json_content["reload_times"][0]) != float or type(json_content["reload_times"][1]) != float:
		# 		raise Exception(f"{error()}: Weapon '{self.name}' has reload times that don't deserialize to floats.")
		# 	self.reload_times = (json_content["reload_times"][0], json_content["reload_times"][1])
		# else:
		# 	print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing the {warning('reload times')}. Using default value ({self.default_reload_times}) instead.")
		# 	self.reload_times = self.default_reload_times

		# get extended barrel weapon, needs to be last bc of copy()
		potential_eb = copy.copy(self)
		self.is_extended_barrel = False	# whether this is an extended barrel version
		if "extended_barrel" in json_content:
			if type(json_content["extended_barrel"]) == bool:	# use default damage multiplier for extended barrel
				has_extended_barrel = json_content["extended_barrel"]
				if has_extended_barrel == True:
					print(f"{warning('Warning:')} Using {warning('approximated')} extended barrel stats for weapon '{warning(self.name)}'.")
					potential_eb.damages = tuple(math.ceil(dmg * self.extended_barrel_damage_multiplier) for dmg in self.damages)

			elif type(json_content["extended_barrel"]) == dict:	# use custom damages for extended barrel
				if not all(isinstance(damage, int) for damage in json_content["extended_barrel"].values()):
					raise Exception(f"{error()}: Weapon '{self.name}' has damage values that don't deserialize to integers.")
				#print(f"{message('Message:')} Using {message('exact')} extended barrel stats for weapon '{message(self.name)}'.")
				has_extended_barrel = True
				potential_eb.damages = self.validate_damages({int(distance) : int(damage) for distance, damage in json_content["extended_barrel"].items()})
			else:
				raise Exception(f"{error()}: Weapon '{self.name}' has an extended barrel value that doesn't deserialize to a bool or a dict.")
		else:
			print(f"{warning('Warning:')} Weapon '{warning(self.name)}' is missing an {warning('extended barrel')} value. Using default value ({self.default_has_extended_barrel}) instead.")
			has_extended_barrel = Weapon.default_has_extended_barrel
			
		self.extended_barrel_parent : Weapon | None = None
		if has_extended_barrel == True:
			potential_eb.name = self.name + " + eb"
			potential_eb.damage_drop_off_intervals = potential_eb.get_damage_drop_off_intervals()

			potential_eb.extended_barrel_weapon = None
			potential_eb.is_extended_barrel = True
			potential_eb.extended_barrel_parent = self
		
			self.extended_barrel_weapon = potential_eb
		else:
			self.extended_barrel_weapon = None

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
	def display_name(self):
		if self.is_extended_barrel:
			return "+ extended barrel"
		return self.name
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
		return None
	@property
	def capacity(self):
		return str(self._capacity[0]) + "+" + str(self._capacity[1])
	@property
	def color(self):
		return self.colors[self.class_]
	@property
	def name_color(self):
		if self.is_extended_barrel:
			return self.empty_color
		return self.color

	def get_damage_drop_off_intervals(self):
		intervals = get_non_stagnant_intervals(self.damages)
		if self.class_ == "SG":
			if len(intervals) != 2:
				raise Exception(f"{error()}: A {self.class_} should have exactly 2 damage dropoff intervals but weapon '{self.name}' has {len(intervals)}.")
			return intervals
		return (intervals[0][0], intervals[-1][-1]),
	def how_useful_is_extended_barrel(self, hp : int):
		a = self.extended_barrel_parent.stdok(0, hp) - self.stdok(0, hp)
		b = round(self.extended_barrel_parent.ttdok(0, hp) - self.ttdok(0, hp))
		if a == 0:
			return a, b, self.empty_color
		return a, b, self.color

	# excel properties
	@property
	def ex_border(self):
		return self.ex_borders[self.class_]

	# other excel styles
	def ex_operators_rich_text(self):
		elements = [op.rich_text_name for op in self.operators]
		n = 1
		i = n
		while i < len(elements):
			elements.insert(i, ', ')
			i += (n+1)
		return CellRichText(*elements)
	
class Operator:
	attacker_color = RGBA.from_rgb_hex("#198FEB")
	defender_color = RGBA.from_rgb_hex("#FB3636")

	def __init__(self, json_content, name : str, ws : Weapons):
		self.name = name

		if "side" not in json_content:
			raise Exception(f"{error()}: Operator '{self.name}' is missing a side.")
		if type(json_content["side"]) != str:
			raise Exception(f"{error()}: Operator '{self.name}' has a side value that doesn't deserialize to a string.")
		if json_content["side"] not in ("A", "D"):
			raise Exception(f"{error()}: Operator '{self.name}' has an invalid side value.")
		self.side = bool(json_content["side"] == "D")	# False: attack, True: defense
		
		if "weapons" not in json_content:
			raise Exception(f"{error()}: Operator '{self.name}' is missing weapons.")
		if type(json_content["weapons"]) != list:
			raise Exception(f"{error()}: Operator '{self.name}' has weapons that don't deserialize to a list.")
		if not all(isinstance(weapon, str) for weapon in json_content["weapons"]):
			raise Exception(f"{error()}: Operator '{self.name}' has weapons that don't deserialize to strings.")
		weapons_strings = list(json_content["weapons"])	# tuple of weapon names
		
		weapons_strings_copy = copy.copy(weapons_strings)
		self.weapons = []
		for weapons_string in weapons_strings:
			for weapon in ws:
				if weapon.name == weapons_string:
					self.weapons.append(weapon)
					weapon.operators.append(self)
					weapons_strings_copy.remove(weapons_string)
					break

		for fake_weapons_string in weapons_strings_copy:
			print(f"{warning('Warning:')} Weapon '{warning(fake_weapons_string)}' found on operator '{self.name}' is {warning('not an actual weapon')}.")
	
		self.rich_text_name = TextBlock(InlineFont(color=Operator.defender_color.to_rgb_hex(False) if self.side else Operator.attacker_color.to_rgb_hex(False)), self.name)

		return

def get_operators_list(ws : list[Weapon], file_name : str) -> None:
	json_content = deserialize_json(file_name)

	if type(json_content) != dict:
		raise Exception(f"{error()}: File '{file_name}' doesn't deserialize to a dict of operators and weapons lists.")

	if not all(isinstance(operator_name, str) for operator_name in json_content):
		raise Exception(f"{error()}: The operator names in file '{file_name}' don't deserialize to strings.")
	if not all(isinstance(op, dict) for op in json_content.values()):
		raise Exception(f"{error()}: The operators in file '{file_name}' don't deserialize to dicts.")

	operators = [Operator(js, op_name, ws) for (op_name, js) in json_content.items()]
		
	return

@dataclass
class Stat:
	_link : str
	stat_method : typing.Callable[[Weapons, typing.Any], pd.DataFrame]
	is_tdok : bool

	def __post_init__(self):
		self.link = f"https://github.com/hanslhansl/Rainbow-Six-Siege-Weapon-Statistics/#{self._link}"

		self.short_name = self.stat_method.__name__.replace("_", " ")
		self.name = self.stat_method.__doc__ if self.stat_method.__doc__ else self.short_name

		if self.is_tdok:
			self.additional_parameter_name = "hp"
			self.additional_parameters = tdok_hp_levels
			self.additional_parameters_descriptions = tdok_levels_descriptions
			#self.additional_parameters_descriptions_short = tdok_levels_descriptions_short
		else:
			self.additional_parameter_name = None
			self.additional_parameters = None

	@property
	def display_name(self):
		if self.short_name != self.name:
			self.short_name + " - " + self.name
		return self.name


stats = (
	Stat("damage-per-bullet", Weapons.damage_per_bullet, False),
	Stat("damage-per-shot", Weapons.damage_per_shot, False),
	Stat("damage-per-second---dps", Weapons.dps, False),

	Stat("bullets-to-down-or-kill---btdok", Weapons.btdok, True),
	Stat("shots-to-down-or-kill---stdok", Weapons.stdok, True),
	Stat("time-to-down-or-kill---ttdok", Weapons.ttdok, True),

	Stat("bullets-to-down-or-kill---btdok", Weapons.btk, True),
	Stat("shots-to-down-or-kill---stdok", Weapons.stk, True),
	Stat("time-to-down-or-kill---ttdok", Weapons.ttk, True),

	Stat("bullets-to-down-or-kill---btdok", Weapons.theoretical_btdok, True),
	Stat("shots-to-down-or-kill---stdok", Weapons.theoretical_stdok, True),
	Stat("time-to-down-or-kill---ttdok", Weapons.theoretical_ttdok, True),
)
stat_illustrations = (
	Weapons.damage_drop_off_coloring,
	Weapons.stat_to_base_stat_gradient_coloring,
	Weapons.stat_to_class_stat_gradient_coloring,
	Weapons.stat_to_class_base_stat_gradient_coloring,
	Weapons.extended_barrel_improvement_coloring,
	)

def add_worksheet_header(worksheet : typing.Any, stat : Stat | str, description : str, row : int, cols_inbetween : int):
	
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

	add_header_entry(row, 2, 6, f'=HYPERLINK("{github_link}", "detailed explanation")', Font(color = "FF0000FF"))

	add_header_entry(row, 8, 14, f'=HYPERLINK("{google_sheets_link}", "spreadsheet on google sheets")', Font(color = "FF0000FF"))

	add_header_entry(row, 16, 1 + cols_inbetween,
				  f'=HYPERLINK("{google_drive_link}", "spreadsheet on google drive")', Font(color = "FF0000FF"))
	row += 2

	if type(stat) == str:
		add_header_entry(row, 2, 1 + cols_inbetween, stat, Font(color = "FF0000FF", bold=True))
	else:
		add_header_entry(row, 2, 1 + cols_inbetween, f'=HYPERLINK("{github_link}#{stat.link}", "{stat.name}")', Font(color = "FF0000FF", bold=True))
	row += 1

	add_header_entry(row, 2, 1 + cols_inbetween, description)
	row += 1

	return row

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
			c.fill = weapon.color.to_ex_fill()
			c.border = weapon.ex_border
			c.alignment = center_alignment
			weapon.ex_border
		col += skip

	c1 = worksheet.cell(row=row, column=col)
	c1.value = weapon.ex_operators_rich_text()

	return

def add_extended_barrel_overview(worksheet : typing.Any, ws : Weapons, row : int, col : int, with_secondary_weapon_stats : bool):
	col_names = tdok_levels_descriptions_short
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

	for weapon in ws:
		if weapon.has_extended_barrel == False:
			continue

		col = original_col
		
		c = worksheet.cell(row=row, column=col)
		c.value = weapon.name
		c.fill = weapon.name_color.to_ex_fill()
		c.border = weapon.ex_border

		if with_secondary_weapon_stats:
			add_secondary_weapon_stats(worksheet, weapon, row, col+15)

		for hp in tdok_hp_levels:
			col += 1
			c1 = worksheet.cell(row=row, column=col)
			c2 = worksheet.cell(row=row, column=col + len(col_names) + 1)
			c1.value, c2.value, color = weapon.extended_barrel_weapon.how_useful_is_extended_barrel(hp)
			c1.fill = c2.fill = color.to_ex_fill()
			c1.alignment = c2.alignment = center_alignment
			c1.border = c2.border = weapon.ex_border
			
		row += 1

	return row

def add_attachment_overview(workbook : typing.Any, ws : Weapons):
	json_content = deserialize_json(attachment_overview_file_name)
	
	if not isinstance(json_content, dict):
		raise Exception(f"{error()}: File '{attachment_overview_file_name}' doesn't deserialize to a dictionary.")
	attachment_categories : dict[str, typing.Any] = json_content

	worksheet = workbook.create_sheet("attachments")
	row = add_worksheet_header(worksheet, "attachment overview", "a short overview over all available attachments.", 1, 19)
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
				row = add_extended_barrel_overview(worksheet, ws, row, 2, True)
				
			row += 1
			
		row += 1

def modify_stats_worksheet(workbook : openpyxl.workbook.workbook.Workbook, ws : Weapons, stat : Stat, illustration):

	if stat.additional_parameters:
		if False: # eb in between: 100hp, eb, 110hp, eb, 125hp, eb, 120hp, eb, 130hp, eb, 145hp
			skip = 2
			inline_skip = 1
			final_skip = 0
		else: # eb afterwards: 100hp, 110hp, 125hp, 120hp, 130hp, 145hp, eb, eb, eb, eb, eb, eb
			skip = 1
			inline_skip = len(stat.additional_parameters)
			final_skip = len(stat.additional_parameters)

		worksheet = workbook.create_sheet(stat.short_name)
	else:
		worksheet = workbook[stat.short_name]

	row = 1

	c = worksheet.cell(row=row, column=1)
	c.value = "weapon"
	worksheet.column_dimensions[get_column_letter(1)].width = 22

	col = 2
	for d in Weapon.distances:
		c = worksheet.cell(row=row, column=col)
		c.value = d
		c.alignment = center_alignment
		worksheet.column_dimensions[get_column_letter(col)].width = 4.8
		col += 1

	row = add_secondary_weapon_stats_header(worksheet, row, 2+len(Weapon.distances))
	worksheet.freeze_panes = worksheet.cell(row=row+1, column=2)
	row += 2

	row = add_worksheet_header(worksheet, stat, illustration.__doc__, row, len(Weapon.distances))
	row += 1

	table_first_row = row
	for wi, weapon in enumerate(ws.weapons.values()):
		c = worksheet.cell(row=row, column=1)
		c.value = weapon.display_name
		c.style = "Normal"
		c.alignment = left_alignment
		c.fill = weapon.name_color.to_ex_fill()
		c.border = weapon.ex_border

		if not weapon.is_extended_barrel:
			add_secondary_weapon_stats(worksheet, weapon, row, len(Weapon.distances) + 3)

		if stat.additional_parameters:
			row += 1
			for j, additional_parameter in enumerate(stat.additional_parameters):
				c = worksheet.cell(row=row, column=1)
				c.value = stat.additional_parameters_descriptions[j]

				source_sheet = workbook[stat.short_name+str(additional_parameter)]
				for i in range(len(Weapon.distances)):
					source_cell = source_sheet.cell(row=table_first_row+wi, column=i+2)
					c = worksheet.cell(row=row, column=i+2)

					c.value = source_cell.value
					c.fill = copy.copy(source_cell.fill)
					c.border = weapon.ex_border
					c.alignment = center_alignment

				row += 1
		else:
			for i in range(len(Weapon.distances)):
				c = worksheet.cell(row=row, column=i+2)
				c.border = weapon.ex_border
				c.alignment = center_alignment
			row += 1


	if stat.additional_parameters:
		for additional_parameter in stat.additional_parameters:
			del workbook[stat.short_name+str(additional_parameter)]

	return


def save_to_xlsx_file(ws : Weapons):
	""" https://openpyxl.readthedocs.io/en/stable/ """

	stat_indices = (0, 1, 2, 4, 5)
	illustration_indices = (0, 1, 2, 4, 3)

	excel_buffer = io.BytesIO()
	with pd.ExcelWriter(excel_buffer) as writer:
		for i_stat, i_illustration in zip(stat_indices, illustration_indices):
			stat = stats[i_stat]
			illustration = stat_illustrations[i_illustration]

			for param in stat.additional_parameters if stat.additional_parameters else ("", ):
				df = stat.stat_method(ws, param)
				styler = illustration(ws, df, True)
				styler.to_excel(writer, sheet_name=stat.short_name + str(param), header=False, startrow=8)

	workbook = openpyxl.load_workbook(excel_buffer)

	for i_stat, i_illustration in zip(stat_indices, illustration_indices):
		stat = stats[i_stat]
		illustration = stat_illustrations[i_illustration]

		modify_stats_worksheet(workbook, ws, stat, illustration)


	#add_attachment_overview(workbook, ws)

	# save to file
	workbook.save(xlsx_output_file_name)
	
	os.system("start " + xlsx_output_file_name)
	return

def save_to_output_files(ws : Weapons):
	#save_to_html_file(weapons, stat_names)
	save_to_xlsx_file(ws)
	return

if __name__ == "__main__":
	# get all weapons from the files
	ws = Weapons()

	# verify
	# group weapons by class and by base damage
	weapons_sorted = sorted((w for w in ws.weapons.values() if not w.is_extended_barrel), key=lambda w: (w.class_, ws.damages()[0][w.name]))
	grouped = [list(group) for key, group in itertools.groupby(weapons_sorted, key=lambda w: (w.class_, ws.damages()[0][w.name]))]
	# find all weapons with the same base damage but different damage drop-off
	failed = False
	for group in grouped:
		if len(group) > 1:
			for i, distance in enumerate(Weapon.distances):
				if len(set(ws.damages()[i][weapon.name] for weapon in group)) > 1:
					print(f"{warning()}: These {group[0].class_}s have the {warning('same base damage')} ({ws.damages()[0][group[0].name]}) but {warning('different damages')} at {distance}m:")
					for weapon in group:
						print(f"{weapon.name}: {ws.damages()[i][weapon.name]}")
					failed = True
	if failed: raise Exception(f"{error()}: See above warnings.")


	# save to excel file
	save_to_output_files(ws)
	#input("Completed!")

