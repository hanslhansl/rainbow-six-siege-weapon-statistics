
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
import os, json, typing, copy, sys, itertools, colorama, sys, colorsys, pandas as pd, numpy as np, io, marshmallow.exceptions
import openpyxl, dataclasses_json, warnings, googleapiclient.http, openpyxl.workbook.workbook, googleapiclient.discovery
import google.oauth2.service_account
from openpyxl.cell.text import InlineFont
from openpyxl.cell.rich_text import TextBlock, CellRichText
from openpyxl.styles import PatternFill, Border, Alignment, NamedStyle, Side, Font
from openpyxl.utils import get_column_letter
from dataclasses import dataclass, field

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

        # get operators
        json_content = deserialize_json(operators_file_name)
        if type(json_content) != dict:
            raise Exception(f"{error()}: File '{operators_file_name}' doesn't deserialize to a dict of operators and weapons lists.")
        if not all(isinstance(operator_name, str) for operator_name in json_content):
            raise Exception(f"{error()}: The operator names in file '{operators_file_name}' don't deserialize to strings.")
        if not all(isinstance(op, dict) for op in json_content.values()):
            raise Exception(f"{error()}: The operators in file '{operators_file_name}' don't deserialize to dicts.")
        self.operators = [Operator(js, op_name) for (op_name, js) in json_content.items()]

        # get weapons
        weapons : list[Weapon] = []
        for file_name in os.listdir(weapon_data_dir):
            file_path = os.path.join(weapon_data_dir, file_name)

            name, extension = os.path.splitext(file_name);		
            if not extension == ".json":
                continue
            if name.startswith("_"):
                print(f"{message('Message: Excluding')} weapon '{message(file_name)}' because of _.")
                continue
        

            w = Weapon(file_path, self.operators)
            weapons.append(w)
        
        # verify operator weapons
        for op in self.operators:
            for weapon_name in op._weapons:
                if weapon_name not in op.weapons:
                    print(f"{warning('Warning:')} Weapon '{warning(weapon_name)}' found on operator '{op.name}' is {warning('not an actual weapon')}.")
            del op._weapons

        # add eb weapons
        weapons += (w.extended_barrel_weapon for w in weapons if w.extended_barrel_weapon)
    
        weapons_sorted = sorted(weapons, key=lambda w: (Weapon.classes.index(w.class_), w.name))
        self.weapons = {w.name : w for w in weapons_sorted}
        self.base_weapons = {n : w for n, w in self.weapons.items() if not w.is_extended_barrel}

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

    # illustrations
    def damage_drop_off_coloring(self, target : pd.DataFrame, source : pd.DataFrame = None):
        """the colored areas represent steady damage, the colorless areas represent decreasing damage"""
        if source is None: source = target
        return self.apply_background_color(target, lambda w, i: w.color if
                                     self.is_in_damage_drop_off(w, i) else w.empty_color)
    def stat_to_base_stat_gradient_coloring(self, target : pd.DataFrame, source : pd.DataFrame = None):
        """the color gradient illustrates the {stat} compared to the weapon's base {stat} (i.e. at 0 m)"""
        if source is None: source = target
        return self.apply_background_color(target, lambda w, i: w.color.with_alpha(safe_division(source.loc[w.name][i], source.loc[w.name].max())))
    def stat_to_class_stat_gradient_coloring(self, target : pd.DataFrame, source : pd.DataFrame = None):
        """the color gradient illustrates the {stat} compared to the weapon class' highest {stat} at the same distance"""
        if source is None: source = target
        class_max = {class_ : group_df.max(axis=0) for class_, group_df in source.groupby(lambda name: self.weapons[name].class_)}
        return self.apply_background_color(target, lambda w, i: w.color.with_alpha(safe_division(source.loc[w.name][i], class_max[w.class_][i])))
    def stat_to_class_base_stat_gradient_coloring(self, target : pd.DataFrame, source : pd.DataFrame = None):
        """the color gradient illustrates the {stat} compared to the weapon's class' highest base {stat} (i.e. at 0 m)"""
        if source is None: source = target
        class_base_max = {class_ : group_df.to_numpy().max() for class_, group_df in source.groupby(lambda name: self.weapons[name].class_)}
        return self.apply_background_color(target, lambda w, i: w.color.with_alpha(safe_division(source.loc[w.name][i], class_base_max[w.class_])))
    def extended_barrel_improvement_coloring(self, target : pd.DataFrame, source : pd.DataFrame = None):
        """the colored areas show where the extended barrel attachment actually affects the {stat}"""
        if source is None: source = target
        return self.apply_background_color(target, lambda w, i:
                                     w.color
                                     if w.is_extended_barrel and source.loc[w.name][i] != source.loc[w.extended_barrel_parent.name][i]
                                     else w.empty_color)


@dataclasses_json.dataclass_json#(undefined=dataclasses_json.Undefined.EXCLUDE)
@dataclass
class _Weapon:
    name : str
    class_ : str = field(metadata=dataclasses_json.config(field_name="class"))
    rpm : int
    ads_time : float
    pellets : int
    _capacity : tuple[int, int] = field(metadata=dataclasses_json.config(field_name="capacity")) # magazine, chamber
    extra_ammo : int
    _damages : dict[int, int] = field(metadata=dataclasses_json.config(field_name="damages"))
    has_laser : bool
    has_grip : bool
    #reload_times : tuple[float, float]
    _extended_barrel : dict[str, int] | bool = field(metadata=dataclasses_json.config(field_name="extended_barrel"))

class Weapon(_Weapon):
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
    ex_borders = {class_ : Border(*(Side(border_style=border_style, color=color.to_border_color().to_rgb_hex(False)) for i in range(4)))
                  for class_, color in colors.items()}

    extended_barrel_damage_multiplier = 0.0
    laser_ads_speed_multiplier = 0.0
    angled_grip_reload_speed_multiplier = 0.0

    empty_color = RGBA(0,0,0,0)

    def __init__(self, file_path, operators : list["Operator"]):
        json_content = deserialize_json(file_path)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                _w = _Weapon.schema().load(json_content)
        except marshmallow.exceptions.ValidationError as e:
            raise Exception(f"{error()}: File '{file_path}' could not be deserialized: {str(e)}.")
        super().__init__(**vars(_w))

        # get operators
        self.operators : list[Operator] = []
        for op in operators:
            if self.name in op._weapons:
                self.operators.append(op)
                op.weapons.append(self)
                op._weapons.remove(self.name)

        # operators rich text
        self.ex_operators_rich_text = CellRichText([elem for op in self.operators for elem in (op.rich_text_name, ", ")][:-1])

        # get weapon names
        
        
        # verify weapon class
        if self.class_ not in self.classes:
            raise Exception(f"{error()}: Weapon '{self.name}' has an invalid weapon class '{json_content["class"]}'.")
        
        # derived fields
        self.base_name = self.name
        self.display_name = self.name
        self.color = self.colors[self.class_]
        self.name_color = self.color
        self.rps = self.rpm / 60.
        self.rpms = self.rpm / 60000.
        self.ads_time_with_laser = self.ads_time / self.laser_ads_speed_multiplier if self.has_laser else None
        self.capacity = str(self._capacity[0]) + "+" + str(self._capacity[1])
        #self.reload_times_with_angled_grip = tuple(x / self.angled_grip_reload_speed_multiplier for x in self.reload_times) if self.has_grip else None

         # excel fields
        self.ex_border = self.ex_borders[self.class_]

        # verify weapon damages
        setattr(self, "damages", self.validate_damages(self._damages))
        self.damage_drop_off_intervals = self.get_damage_drop_off_intervals()

        # get extended barrel weapon, needs to be last bc of copy()
        self.extended_barrel_parent : Weapon | None = None
        self.is_extended_barrel = False
        self.extended_barrel_weapon = None
        if self._extended_barrel != False:
            self.extended_barrel_weapon = copy.copy(self)
            self.extended_barrel_weapon.name = self.name + " + eb"
            self.extended_barrel_weapon.display_name = "+ extended barrel"
            self.extended_barrel_weapon.name_color = self.empty_color
            setattr(
                self.extended_barrel_weapon,
                "damages",
                self.extended_barrel_weapon.validate_damages({int(k) : v for k, v in self._extended_barrel.items()}))
            self.extended_barrel_weapon.damage_drop_off_intervals = self.extended_barrel_weapon.get_damage_drop_off_intervals()

            self.extended_barrel_weapon.extended_barrel_parent = self
            self.extended_barrel_weapon.is_extended_barrel = True

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

    def get_damage_drop_off_intervals(self):
        intervals = get_non_stagnant_intervals(self.damages)
        if self.class_ == "SG":
            if len(intervals) != 2:
                raise Exception(f"{error()}: A {self.class_} should have exactly 2 damage dropoff intervals but weapon '{self.name}' has {len(intervals)}.")
            return intervals
        return (intervals[0][0], intervals[-1][-1]),
    

@dataclasses_json.dataclass_json
@dataclass
class _Operator:
    _side : str = field(metadata=dataclasses_json.config(field_name="side"))
    _weapons : list[str] = field(metadata=dataclasses_json.config(field_name="weapons"))

class Operator(_Operator):
    attacker_color = RGBA.from_rgb_hex("#198FEB")
    defender_color = RGBA.from_rgb_hex("#FB3636")

    def __init__(self, json_content, name : str):
        self.name = name

        try:
            _w = _Operator.schema().load(json_content)
        except marshmallow.exceptions.ValidationError as e:
            raise Exception(f"{error()}: Operator '{self.name}' could not be deserialized: {str(e)}.")
        super().__init__(**vars(_w))

        if self._side not in ("A", "D"):
            raise Exception(f"{error()}: Operator '{self.name}' has an invalid side value '{self._side}'.")
        self.side = bool(self._side == "D")	# False: attack, True: defense
        
        self.weapons : list[Weapon] = []
        
        self.rich_text_name = TextBlock(InlineFont(color=Operator.defender_color.to_rgb_hex(False) if self.side else Operator.attacker_color.to_rgb_hex(False)), self.name)

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
        else:
            self.additional_parameter_name = None
            self.additional_parameters = None

    @property
    def display_name(self):
        if self.short_name != self.name:
            return self.short_name + " - " + self.name
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

    if isinstance(stat, str):
        add_header_entry(row, 2, 1 + cols_inbetween, stat, Font(color = "FF0000FF", bold=True))
    else:
        add_header_entry(row, 2, 1 + cols_inbetween,
                         f'=HYPERLINK("{github_link}#{stat.link}", "{stat.display_name}")', Font(color = "FF0000FF", bold=True))
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
    values = [weapon.class_, weapon.rpm, weapon.capacity, weapon.extra_ammo, weapon.pellets if weapon.pellets != 1 else None,
           weapon.ads_time, weapon.ads_time_with_laser,
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
    c1.value = weapon.ex_operators_rich_text

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

    row = add_worksheet_header(worksheet, stat, illustration.__doc__.format(stat=stat.short_name), row, len(Weapon.distances))
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
                styler = illustration(ws, df)
                styler.to_excel(writer, sheet_name=stat.short_name + str(param), header=False, startrow=8)

    workbook = openpyxl.load_workbook(excel_buffer)

    for i_stat, i_illustration in zip(stat_indices, illustration_indices):
        stat = stats[i_stat]
        illustration = stat_illustrations[i_illustration]

        modify_stats_worksheet(workbook, ws, stat, illustration)


    #add_attachment_overview(workbook, ws)

    # save to file
    workbook.save(xlsx_output_file_name)
    
    return xlsx_output_file_name

if __name__ == "__main__":

    # get all weapons from the files
    ws = Weapons()

    # verify
    # group weapons by class and by base damage
    weapons_sorted = sorted((w for w in ws.weapons.values() if not w.is_extended_barrel), key=lambda w: (w.class_, ws.damage_per_bullet()[0][w.name]))
    grouped = [list(group) for key, group in itertools.groupby(weapons_sorted, key=lambda w: (w.class_, ws.damage_per_bullet()[0][w.name]))]
    # find all weapons with the same base damage but different damage drop-off
    failed = False
    for group in grouped:
        if len(group) > 1:
            for i, distance in enumerate(Weapon.distances):
                if len(set(ws.damage_per_bullet()[i][weapon.name] for weapon in group)) > 1:
                    print(f"{warning()}: These {group[0].class_}s have the {warning('same base damage')} ({ws.damage_per_bullet()[0][group[0].name]}) but {warning('different damages')} at {distance}m:")
                    for weapon in group:
                        print(f"{weapon.name}: {ws.damage_per_bullet()[i][weapon.name]}")
                    failed = True
    if failed: raise Exception(f"{error()}: See above warnings.")


    # save to excel file
    excel_file_name = save_to_xlsx_file(ws)

    if "GOOGLE_SERVICE_ACCOUNT_CREDENTIALS" in os.environ:
        # we are in the github action
        credentials_json = os.environ["GOOGLE_SERVICE_ACCOUNT_CREDENTIALS"]
        info = json.loads(credentials_json)

        # create service account credentials
        creds = google.oauth2.service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/drive"]
        )

        drive_service = googleapiclient.discovery.build("drive", "v3", credentials=creds)

        # Google Sheet ID
        sheet_id = "1QgbGALNZGLlvf6YyPLtywZnvgIHkstCwGl1tvCt875Q"

        # upload excel file replace sheet
        file_metadata = {"mimeType": "application/vnd.google-apps.spreadsheet"}
        media = googleapiclient.http.MediaFileUpload(
            excel_file_name,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            resumable=True
        )

        updated_file = drive_service.files().update(
            fileId=sheet_id,
            media_body=media,
            body=file_metadata,
            fields="id"
        ).execute()

    else:
        # we are local
        os.system("start " + excel_file_name)

