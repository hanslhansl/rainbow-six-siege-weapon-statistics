# --- CONFIG ---
# normalized to dimensions of containing rect

# the position of the rect of interest
NORMALIZED_RECT_X = 0.745
NORMALIZED_RECT_Y = 0.872
NORMALIZED_RECT_W = 0.198
NORMALIZED_RECT_H = 0.043
NORMALIZED_SECONDARY_RECT_Y_OFFSET = 0.035
NORMALIZED_TERTIARY_RECT_Y_OFFSET = -0.042

# ignore blobs smaller than this
MIN_NORMALIZED_HEIGHT_FOR_BLOB = 0.8   # relative to crop height

# the hole in 0 and 4 is centered with this tolerance, otherwise 6 or 9
MAX_NORMALIZED_HOLE_OFFSET_FROM_CENTER_FOR_ZERO_FOUR = 0.1
# the hole in 0 is this tall, otherwise it's 4
MIN_NORMALIZED_HOLE_HEIGHT_FOR_ZERO = 0.6

# the aspect ratio of 1 is smaller than this
MAX_ASPECT_RATIO_FOR_ONE = 19 / 44  # width / height

# Progress bar
PROGRESS_BAR_WIDTH = 30


# --- CODE ---
import cv2, numpy as np, sys, typing, av, atexit, pathlib, scipy.optimize, scipy.special, argparse, collections, json, itertools
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

# --- argument parsing ---
parser = argparse.ArgumentParser(description="measure reload times and fire rates of weapons from video recordings and update JSON files accordingly")

# positional argument: file path
parser.add_argument( "path", type=pathlib.Path, help="path to the video file (or directory containing video files)")

# group 1: primary vs secondary
role_group = parser.add_mutually_exclusive_group(required=True)
role_group.add_argument("--primary", dest="weapon_slot", action="store_const", const="primary")
role_group.add_argument("--secondary", dest="weapon_slot", action="store_const", const="secondary")
role_group.add_argument("--tertiary", dest="weapon_slot", action="store_const", const="tertiary")

# aspect ratio
parser.add_argument("--aspect-ratio", type=str, required=True, help="in-game aspect ratio like 16:9 or 4:3")

# group 2: crop offsets
parser.add_argument("-l", "--left-crop", type=int, default=0, help="left crop offset (integer)")
parser.add_argument("-t", "--top-crop", type=int, default=0, help="top crop offset (integer)")
parser.add_argument("-r", "--right-crop", type=int, default=0, help="right crop offset (integer)")
parser.add_argument("-b", "--bottom-crop", type=int, default=0, help="bottom crop offset (integer)")

# don't update json files, just print results
parser.add_argument("--dry-run", action="store_true", help="process the video and print results without updating JSON files")

# whether the video was recorded with angled grip attached, which changes the reload times and ammo counter behavior
parser.add_argument("--angled-grip", action="store_false", help="if the video was recorded with angled grip attached")

# --- video processing ---
@dataclass
class Number:
    value: int | None
    start: float
    end: float

@dataclass
class Event:
    t0: float
    t1: float
    t2: float
    t3: float

    def __post_init__(self):
        self.minimum_duration = self.t2 - self.t1
        self.maximum_duration = self.t3 - self.t0

        self.statistical_start = (self.t0 + self.t1) / 2
        self.statistical_end = (self.t2 + self.t3) / 2

        self.statistical_duration = self.statistical_end - self.statistical_start
        self.statistical_radius = (-self.t0 + self.t3 - self.t2 + self.t1) / 2

    def __str__(self):
        return (
            f"{self.__doc__}: {self.statistical_start:10.6f}s → {self.statistical_end:10.6f}s | "
            f"Δt: {self.statistical_duration:.6f} ± {self.statistical_radius:.6f} s | "
            f"min/max Δt: {self.minimum_duration:.6f}/{self.maximum_duration:.6f} s"
            )
    
    @staticmethod
    def get_display_value(value):
        return round_half_up(value, ndigits=2)
class TacticalReloadEvent(Event):
    """tactical reload"""

    @staticmethod
    def get_json_value(json_data):
        return json_data["reload_times"][0 if ANGLED_GRIP else 2]
    @staticmethod
    def set_json_value(value, json_data):
        json_data["reload_times"][0 if ANGLED_GRIP else 2] = value
class FullReloadEvent(Event):
    """full reload"""

    @staticmethod
    def get_json_value(json_data):
        return json_data["reload_times"][1 if ANGLED_GRIP else 3]
    @staticmethod
    def set_json_value(value, json_data):
        json_data["reload_times"][1 if ANGLED_GRIP else 3] = value
@dataclass
class FireRateEvent(Event):
    """fire rate"""
    rounds : int

    def __str__(self):
        return super().__str__() + f" | rounds: {self.rounds}"

    def __post_init__(self):
        super().__post_init__()
        # Adjust time values to be per bullet (time between shots)
        # N rounds have N-1 intervals between them
        intervals = self.rounds - 1
        # self.minimum_duration /= intervals
        # self.maximum_duration /= intervals
        # self.statistical_duration /= intervals
        # self.statistical_radius /= intervals

    @staticmethod
    def get_display_value(value):
        return int(round_half_up(60 / value, ndigits=0))

    @staticmethod
    def get_json_value(json_data):
        return json_data["rpm"]
    @staticmethod
    def set_json_value(value, json_data):
        json_data["rpm"] = value

def round_half_up(fl, ndigits=0):
    q = Decimal('1.' + '0' * ndigits)
    return float(Decimal.from_float(fl).quantize(q, rounding=ROUND_HALF_UP))

def pad(arr, relative_to):
    return np.pad(arr, ((0, relative_to.shape[0]-arr.shape[0]), (0, 0)), mode='constant', constant_values=127)

def crop_padding(mask):
    """crop to bounding box"""
    nonzero_points = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(nonzero_points)
    cropped = mask[y:y+h, x:x+w]
    return cropped

def classify_digit(cropped_mask):
    h, w = cropped_mask.shape
    aspect_ratio = w / h

    # check apsect ratio, 1 is narrowest digit
    if aspect_ratio <= MAX_ASPECT_RATIO_FOR_ONE:
        return 1, f"{aspect_ratio=:.4f}"
    
    # find contours
    contours, hierarchy = cv2.findContours(
        cropped_mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )
    number_of_holes = len(contours) - 1

    # if there are 2 holes it's 8
    if number_of_holes == 2:
        return 8, f"{number_of_holes=}"
    
    # if there is 1 hole it's 0, 4, 6, or 9
    if number_of_holes == 1:
        hole, = [cnt for i, cnt in enumerate(contours) if hierarchy[0][i][3] != -1]
        hx, hy, hw, hh = cv2.boundingRect(hole)
        normalized_hole_offset_from_center_x = (hx + hw / 2.) / w - 0.5
        normalized_hole_offset_from_center_y = (hy + hh / 2.) / h - 0.5

        if normalized_hole_offset_from_center_y < -MAX_NORMALIZED_HOLE_OFFSET_FROM_CENTER_FOR_ZERO_FOUR:
            return 9, f"{normalized_hole_offset_from_center_y=:.4f}"
        if normalized_hole_offset_from_center_y > MAX_NORMALIZED_HOLE_OFFSET_FROM_CENTER_FOR_ZERO_FOUR:
            return 6, f"{normalized_hole_offset_from_center_y=:.4f}"

        normalized_hole_height = hh / h
        return 0 if normalized_hole_height >= MIN_NORMALIZED_HOLE_HEIGHT_FOR_ZERO else 4, f"{normalized_hole_height=:.4f}"

    # match 2, 3, 5, 7 with templates
    normalized_area = np.sum(cropped_mask) / (255 * w * h)
    templates = []
    def match_template(area, points, num_samples = 50):
        points = [((x0*w, y0*h), (x1*w, y1*h)) for (x0, y0), (x1, y1) in points]
        lines = [np.linspace(p0, p1, num_samples).astype(int) for p0, p1 in points]

        template = cropped_mask.copy()
        for line in lines:
            template[line[:, 1], line[:, 0]] = 160

        return all(np.all(cropped_mask[line[:, 1], line[:, 0]] == 255) for line in lines), template

    points7 = [
        ((0.15, 0.08), (0.8, 0.08)),
        ((0.85, 0.08), (0.4, 0.92)),
        ]

    points5 = [
        ((0.8, 0.07), (0.21, 0.07)),
        ((0.21, 0.07), (0.21, 0.4)),
        ((0.85, 0.5), (0.85, 0.8)),
        ((0.7, 0.93), (0.3, 0.93)),
        ]

    points3 = [
        ((0.4, 0.07), (0.6, 0.07)),
        ((0.81, 0.15), (0.81, 0.35)),
        ((0.5, 0.47), (0.75, 0.47)),
        ((0.82, 0.6), (0.82, 0.8)),
        ((0.4, 0.93), (0.6, 0.93)),
        ]

    points2 = [
        ((0.4, 0.07), (0.6, 0.07)),
        ((0.8, 0.4), (0.25, 0.7)),
        ((0.15, 0.91), (0.85, 0.91)),
        ]

    digits = (7, 5, 3, 2)
    points = (points7, points5, points3, points2)

    for digit, points in zip(digits, points):
        res, *template = match_template(normalized_area, points)
        templates.extend(template)
        if res:
            return digit, f"matched template", *template
             
    return -1, f"{number_of_holes=}, {normalized_area=:.4f}", *templates

def process_rect(img):
    # Strict but tolerant RGB mask: red/black with GB noise allowed
    B, G, R = cv2.split(img)
    if np.any(np.all(B > 0xE6, axis=0)):
        return None, "wave overlay"

    # create mask
    mask = crop_padding(((G < 0x40) & (B < 0x40)).astype(np.uint8) * 255)

    # either all black or all white means mask is empty
    if np.all(mask == 0) or np.all(mask != 0):
        return None, "mask is empty"

    # morphological closing to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morphed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    morphed_mask = cv2.GaussianBlur(morphed_mask, (7, 7), 0)
    morphed_mask = cv2.threshold(morphed_mask, 127, 255, cv2.THRESH_BINARY)[1]

    # Connected components
    num, labels, stats, _ = cv2.connectedComponentsWithStats(morphed_mask)
    if num <= 1:
        return None, "found no blobs", mask, morphed_mask

    # filter out small blobs
    h = img.shape[0]
    blobs = []
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_HEIGHT] / h >= MIN_NORMALIZED_HEIGHT_FOR_BLOB:
            x, y, w, h, area = stats[i]
            blob = morphed_mask[y:y+h, x:x+w]
            if area != cv2.countNonZero(blob):
                return None, "overlapping blobs", mask, morphed_mask, blob
            blobs.append(blob)

    if len(blobs) == 0:
        return None, "no blobs passed height filter", mask, morphed_mask

    number = 0
    masks = []
    for i, blob in enumerate(blobs):
        digit, reason, *blob_masks = classify_digit(blob)
        if digit == -1:
            return -1, reason, mask, morphed_mask, blob, *blob_masks
        number += digit * int(10 ** (len(blobs) - 1 - i))
        masks.append(blob)
        masks.extend(blob_masks)
    
    if len(blobs) != 1:
        reason = f"found {len(blobs)} blobs"

    return number, reason, mask, morphed_mask, *masks

def get_increasing_suffix(ammo_counter : list[Number], delta : int):
    suf : list[Number] = []
    for number in reversed(ammo_counter):
        if not isinstance(number.value, int):
            break
        if suf and number.value + delta != suf[0].value:
            break
        suf.insert(0, number)
    return suf

def detect_tactical_reload(ammo_counter : list[Number]):
    if len(ammo_counter) >= 3:
        e2, e1, e0 = ammo_counter[-3:]
        if None not in (e2.value, e1.value, e0.value):
            if e2.value > 1 and e1.value <= 1 and e0.value > 1:
                t0, t1 = e2.end, e1.start
                t2, t3 = e1.end, e0.start
                return TacticalReloadEvent(t0, t1, t2, t3), False
    return None
def detect_full_reload(ammo_counter : list[Number]):
    if len(ammo_counter) >= 4:
        e3, e2, e1, e0 = ammo_counter[-4:]
        if e3.value == 2 and e2.value == 1 and e1.value == 0 and e0.value is not None and e0.value > 1:
            t0, t1 = e2.end, e1.start
            t2, t3 = e1.end, e0.start
            return FullReloadEvent(t0, t1, t2, t3), False
    return None

def detect_tube_fed_tactical_reload(ammo_counter : list[Number]):
    reload = get_increasing_suffix(ammo_counter, 1)
    if len(reload) >= 3 and reload[0].value == 0:
        e2, e1, e0 = reload[-3:]
        t0, t1 = e2.end, e1.start
        t2, t3 = e1.end, e0.start
        return TacticalReloadEvent(t0, t1, t2, t3), len(reload) > 3
    return None
def detect_tube_fed_full_reload(ammo_counter : list[Number]):
    reload = get_increasing_suffix(ammo_counter, 1)
    if len(reload) >= 3 and len(ammo_counter) > len(reload) + 1:
        e3 = ammo_counter[-len(reload)-1]
        e2 = reload[0]
        e1, e0 = reload[-2:]
        if e3.value == 1 and e2.value == 0:
            t0, t1 = e3.end, e2.start
            t2, t3 = e1.end, e0.start
            return FullReloadEvent(t0, t1, t2, t3), len(reload) > 3
    return None

def detect_burst(ammo_counter : list[Number]):
    burst = get_increasing_suffix(ammo_counter, -1)
    if len(burst) >= 3:
        e3 = burst[0]
        e2 = burst[1]
        e1 = burst[-2]
        e0 = burst[-1]
        return FireRateEvent(e3.end, e2.start, e1.end, e0.start, rounds=len(burst)-1), len(burst) > 3
    return None

def process_video(video_path : pathlib.Path):
    print("--- Video File ---")

    def handle_error(rect, masks, reason):
        separator = np.full((rect.shape[0], 2, rect.shape[2]), (0,255,0), dtype=rect.dtype)
        chain = itertools.chain.from_iterable((cv2.cvtColor(pad(m, rect), cv2.COLOR_GRAY2BGR), separator) for m in masks)
        cv2.imshow("Video", np.hstack((rect, separator, *chain)))
        cv2.waitKey(1)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        print("\n")
        raise ValueError(f"Unable to classify digit: {reason}")
        # print(f"\nUnable to classify digit: {reason}")

    with av.open(video_path) as container:
        print(f"path: {container.name}")
        stream = container.streams.video[0]
        print(f"fps: {float(stream.average_rate)}")
        total_duration = float(container.duration) / av.time_base
        print(f"duration: {total_duration} s")

        if WEAPON_SLOT == "primary":
            normalized_rect_y_offset = 0
        elif WEAPON_SLOT == "secondary":
            normalized_rect_y_offset = NORMALIZED_SECONDARY_RECT_Y_OFFSET
        elif WEAPON_SLOT == "tertiary":
            normalized_rect_y_offset = NORMALIZED_TERTIARY_RECT_Y_OFFSET
        width = stream.width
        height = stream.height
        print(f"resolution: {width}x{height}")
        absolute_width =  LEFT_CROP + width + RIGHT_CROP
        absolute_height = TOP_CROP + height + BOTTOM_CROP
        print(f"absolute resolution: {absolute_width}x{absolute_height}")
        absolute_rect_x = int(absolute_width * ((NORMALIZED_RECT_X - 1) / ASPECT_RATIO + 1))
        absolute_rect_y = int((NORMALIZED_RECT_Y + normalized_rect_y_offset) * absolute_height)
        print(f"absolute rect position: x={absolute_rect_x}, y={absolute_rect_y}")
        rect_x = absolute_rect_x - LEFT_CROP
        rect_y = absolute_rect_y - TOP_CROP
        print(f"rect position: x={rect_x}, y={rect_y}")
        rect_w = int(absolute_width * NORMALIZED_RECT_W / ASPECT_RATIO)
        rect_h = int(absolute_height * NORMALIZED_RECT_H)
        print(f"rect resolution: {rect_w}x{rect_h}")


        print("--- Video Processing ---")
        ammo_counter : list[Number] = []
        events = collections.defaultdict[type, list[Event]](list)
        
        for frame in container.decode(video=0):
            frame_time = frame.time

            # crop frame
            rect = frame.to_rgb().to_ndarray(format="bgr24")[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w]

            # get status from rect
            number, reason, *masks = process_rect(rect)

            # update progress bar
            filled_length = min(round(PROGRESS_BAR_WIDTH * frame_time / total_duration), PROGRESS_BAR_WIDTH)
            bar = '█' * filled_length + '-' * (PROGRESS_BAR_WIDTH - filled_length)
            sys.stdout.write(f"\r\033[K|{bar}| {frame_time:.4f}/{total_duration:.4f} s | {frame_time / total_duration:.2%} | {number!r} ({reason})")
            sys.stdout.flush()
            
            # display masks for debugging, maybe raise
            if number == -1:
                handle_error(rect, masks, reason)

            # update ammo_counter/reload_events
            if len(ammo_counter) == 0 or number != ammo_counter[-1].value:

                # verify valid state transition
                if len(ammo_counter) >= 1:
                    old_number = ammo_counter[-1].value
                    if old_number not in (None, 0, 1) and number is not None:
                        if not (old_number > number or old_number + 1 == number):
                            handle_error(rect, masks, f"Invalid state transition: {old_number} → {number}")

                ammo_counter.append(Number(value=number, start=frame_time, end=frame_time))

                new_events : list[None | tuple[Event, bool]] = [
                    detect_tactical_reload(ammo_counter),
                    detect_full_reload(ammo_counter),
                    detect_tube_fed_tactical_reload(ammo_counter),
                    detect_tube_fed_full_reload(ammo_counter),
                    detect_burst(ammo_counter),
                ]
                for tup in new_events:
                    if tup is not None:
                        e, remove_previous = tup
                        t0, t1, t2, t3 = e.t0, e.t1, e.t2, e.t3
                        if not (t0 < t1 < t2 < t3):
                            handle_error(rect, masks, f"Timestamps must be in order: {t0}, {t1}, {t2}, {t3}")
                        if not remove_previous:
                            old_e = events[type(e)][-1] if len(events[type(e)]) > 0 else None
                            if old_e is not None:
                                sys.stdout.write(f"\r\033[K{old_e}\n")
                                sys.stdout.flush()
                        if remove_previous:
                            events[type(e)].pop()
                        events[type(e)].append(e)

            else:
                ammo_counter[-1].end = frame_time

            separator = np.full((rect.shape[0], 2, rect.shape[2]), (0,255,0), dtype=rect.dtype)
            chain = itertools.chain.from_iterable((cv2.cvtColor(pad(m, rect), cv2.COLOR_GRAY2BGR), separator) for m in masks)
            cv2.imshow("Video", np.hstack((rect, separator, *chain)))
            cv2.waitKey(1)

    sys.stdout.write("\r\033[2K")
    sys.stdout.flush()
    for event_list in events.values():
        print((f"{event_list[-1]}"))
    print(f"found {sum(len(v) for v in events.values())} events")
    return events

# --- Analysis ---
def interval_cost(D, reload_events : list[Event], sigma=0.0005, delta=1.5):
    """
    D: current duration estimate (scalar)
    reload_events: list of ReloadEvent
    sigma: small softening for minimal violations
    delta: Huber parameter
    """
    total = 0.0
    for re in reload_events:
        # compute interval violation
        v = max(0.0, abs(D - re.statistical_duration) - re.statistical_radius)
        # scale by sigma and apply Huber
        total += scipy.special.huber(delta, v / sigma)
    return total

def analyze_reload_events(event_type : type[Event], events : list[Event]):
    if len(events) < 7:
        print(f"Not enough measurements for {event_type.__doc__}: {len(events)} (need at least 7 for robust estimation)")
        return None

    # Run optimization, initial guess: weighted average
    x0 = np.mean([m.statistical_duration for m in events])

    res = scipy.optimize.minimize(
        interval_cost,
        x0=[x0],
        args=(events,),
        method='Nelder-Mead',  # robust 1D optimizer
        options={'xatol':1e-9, 'disp': False}
    )
    D_star = float(res.x[0])

    # Estimate effective radius (uncertainty), max violation after robust estimate
    R_star = max(max(0.0, abs(D_star - m.statistical_duration) - m.statistical_radius) for m in events)
    assert R_star < 0.0084, f"Unreasonably high uncertainty: {R_star:.5f} s for {event_type.__doc__}"

    print(f"{len(events)} {event_type.__doc__} events, statistical duration: {D_star} ± {R_star} s -> {event_type.get_display_value(D_star)}")

    return D_star, R_star, event_type

def analyze(events : dict[type[Event], list[Event]]):
    print("--- Analysis ---")
    results = [result for event_type, events in events.items() if (result := analyze_reload_events(event_type, events)) is not None]
    print(f"completed analysis of {len(results)} event types")
    return results


if __name__ == "__main__":
    cb = lambda: input("\npress enter to exit...")
    atexit.register(cb)
    args = parser.parse_args()
    PATH : pathlib.Path = args.path
    WEAPON_SLOT = args.weapon_slot
    w, h = map(int, args.aspect_ratio.split(":"))
    ASPECT_RATIO = w / h
    LEFT_CROP = args.left_crop
    TOP_CROP = args.top_crop
    RIGHT_CROP = args.right_crop
    BOTTOM_CROP = args.bottom_crop
    DRY_RUN = args.dry_run
    ANGLED_GRIP = args.angled_grip

    if PATH.is_file():
        video_paths = [PATH]
    else:
        video_paths = list(PATH.iterdir())
    
    for video_path in video_paths:
        assert video_path.is_file(), f"Video file not found: {video_path}"
        events = process_video(video_path)
        results = analyze(events)

        if not DRY_RUN and len(results) != 0:
            parent_path = pathlib.Path(__file__).parent
            weapons_dict = {path.stem: path for path in (parent_path / "weapons").glob("*.json")}
            
            # Get weapon name from video file and update corresponding JSON
            weapon_name = video_path.stem
            weapon_json_path = weapons_dict[weapon_name]
            with open(weapon_json_path, 'r') as f:
                weapon_data = json.load(f)
            
            # Update reload_times based on results
            change_made = False
            for D_star, R_star, event_type in results:
                current_value = event_type.get_json_value(weapon_data)
                new_value = event_type.get_display_value(D_star)
                if current_value is None:
                    change_made = True
                    event_type.set_json_value(new_value, weapon_data)
                    print(f"setting {event_type.__doc__} for {weapon_name} to {new_value} (was {current_value})")
                elif current_value == new_value:
                    print(f"no change in {event_type.__doc__} for {weapon_name} (value is {current_value})")
                else:
                    raise ValueError(f"new value for {event_type.__doc__} for {weapon_name} '{new_value}' unequal old value '{current_value}'")
            
            # Save the updated JSON file
            if change_made:
                with open(weapon_json_path, 'w') as f:
                    json.dump(weapon_data, f, indent=4)
            
                print(f"Updated {weapon_json_path}")

        print()

    cv2.destroyAllWindows()
    # atexit.unregister(cb)
    