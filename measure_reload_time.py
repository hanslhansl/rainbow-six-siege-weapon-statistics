# --- CONFIG ---
# normalized to dimensions of containing rect

# the position the rect of interest
NORMALIZED_RECT_X = 0.830
NORMALIZED_RECT_Y = 0.872
NORMALIZED_RECT_W = 0.132
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
import cv2, cv2.ximgproc, numpy as np, sys, typing, av, atexit, pathlib, scipy.optimize, scipy.special, argparse, collections, json, itertools
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

# --- Argument parsing ---
parser = argparse.ArgumentParser(
    description="scan a video file for reload animations and measure the duration"
)

# Positional Argument: file path
parser.add_argument(
    "filepath",
    type=pathlib.Path,
    help="path to the video file (or directory containing video files)"
)

# Group 1: primary vs secondary
role_group = parser.add_mutually_exclusive_group(required=True)
role_group.add_argument("--primary", dest="mode", action="store_const", const="primary")
role_group.add_argument("--secondary", dest="mode", action="store_const", const="secondary")
role_group.add_argument("--tertiary", dest="mode", action="store_const", const="tertiary")

# Group 2: crop offsets
parser.add_argument(
    "-l",
    "--left-crop",
    type=int,
    default=0,
    help="Left crop offset (integer)"
)
parser.add_argument(
    "-t",
    "--top-crop",
    type=int,
    default=0,
    help="Top crop offset (integer)"
)
parser.add_argument(
    "-r",
    "--right-crop",
    type=int,
    default=0,
    help="Right crop offset (integer)"
)
parser.add_argument(
    "-b",
    "--bottom-crop",
    type=int,
    default=0,
    help="Bottom crop offset (integer)"
)

parser.add_argument("--dry-run", action="store_true", help="process the video and print results without updating JSON files")
parser.add_argument("--angled-grip", action="store_false", help="if the video was recorded with angled grip attached")

# --- Video processing ---
@dataclass
class Number:
    value: int | None
    start: float
    end: float

@dataclass
class ReloadEvent:
    statistical_duration: float
    radius: float

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

def process_video(video_path, mode, l = 0, t = 0, r = 0, b = 0):
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

        width = stream.width
        height = stream.height
        print(f"resolution: {width}x{height}")
        total_width =  l + width + r
        total_height = t + height + b
        print(f"total resolution: {total_width}x{total_height}")
        rect_x = int(NORMALIZED_RECT_X * total_width) - l
        if mode == "primary":
            normalized_rect_y_offset = 0
        elif mode == "secondary":
            normalized_rect_y_offset = NORMALIZED_SECONDARY_RECT_Y_OFFSET
        elif mode == "tertiary":
            normalized_rect_y_offset = NORMALIZED_TERTIARY_RECT_Y_OFFSET
        rect_y = int((NORMALIZED_RECT_Y + normalized_rect_y_offset) * total_height) - t
        rect_w = int(NORMALIZED_RECT_W * total_width)
        rect_h = int(NORMALIZED_RECT_H * total_height)
        print(f"crop rect: x={rect_x}, y={rect_y}, w={rect_w}, h={rect_h}")
        print(f"fps: {float(stream.average_rate)}")
        total_duration = float(container.duration) / av.time_base
        print(f"duration: {total_duration} s")


        print("--- Video Processing ---")
        ammo_counter : list[Number] = []
        reload_events = collections.defaultdict[typing.Literal["TACTICAL", "FULL"], list[ReloadEvent]](list)
        
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
                        if not old_number > number:
                            handle_error(rect, masks, f"Invalid state transition: {old_number} → {number}")

                ammo_counter.append(Number(value=number, start=frame_time, end=frame_time))

                type = None

                if len(ammo_counter) >= 3:
                    e2, e1, e0 = ammo_counter[-3:]
                    if e2.value not in (None, 0, 1) and e1.value in (0, 1) and e0.value not in (None, 0, 1): # tactical
                        type = "TACTICAL"

                if len(ammo_counter) >= 4:
                    e3, e2, e1, e0 = ammo_counter[-4:]
                    if e3.value not in (None, 0, 1) and e2.value == 1 and e1.value == 0 and e0.value not in (None, 0): # full
                        type = "FULL"

                if type is not None:
                    t0, t1 = e2.end, e1.start
                    t2, t3 = e1.end, e0.start
                    if not (t0 < t1 < t2 < t3):
                        handle_error(rect, masks, f"Timestamps must be in order: {t0}, {t1}, {t2}, {t3}")
                    S = (t0 + t1) / 2
                    E = (t2 + t3) / 2
                    D = E - S
                    R = (-t0 + t3 - t2 + t1) / 2
                    reload_events[type].append(ReloadEvent(statistical_duration=D, radius=R))
                    sys.stdout.write(f"\r\033[K{type} reload: {S:10.6f}s → {E:10.6f}s | Δt: {D:.6f} ± {R:.6f} s | min/max Δt: {t2 - t1:.6f}/{t3 - t0:.6f} s\n")
                    sys.stdout.flush()
            else:
                ammo_counter[-1].end = frame_time

    print()
    print(f"found {sum(len(v) for v in reload_events.values())} {'primary' if mode == 'primary' else 'secondary' if mode == 'secondary' else 'tertiary'} weapon reloads")
    return reload_events


# --- Analysis ---
def interval_cost(D, reload_events : list[ReloadEvent], sigma=0.0005, delta=1.5):
    """
    D: current duration estimate (scalar)
    reload_events: list of ReloadEvent
    sigma: small softening for minimal violations
    delta: Huber parameter
    """
    total = 0.0
    for re in reload_events:
        # compute interval violation
        v = max(0.0, abs(D - re.statistical_duration) - re.radius)
        # scale by sigma and apply Huber
        total += scipy.special.huber(delta, v / sigma)
    return total

def round_half_up(fl, ndigits=0):
    q = Decimal('1.' + '0' * ndigits)
    return float(Decimal.from_float(fl).quantize(q, rounding=ROUND_HALF_UP))

def analyze_reload_events(type, reload_events : list[ReloadEvent]):
    assert len(reload_events) >= 7, f"Not enough measurements for {type} reload: {len(reload_events)} (need at least 7 for robust estimation)"

    # Run optimization, initial guess: weighted average
    x0 = np.mean([m.statistical_duration for m in reload_events])

    res = scipy.optimize.minimize(
        interval_cost,
        x0=[x0],
        args=(reload_events,),
        method='Nelder-Mead',  # robust 1D optimizer
        options={'xatol':1e-9, 'disp': False}
    )
    D_star = float(res.x[0])

    # Estimate effective radius (uncertainty), max violation after robust estimate
    R_star = max(max(0.0, abs(D_star - m.statistical_duration) - m.radius) for m in reload_events)

    print(f"{len(reload_events)} {type} reloads, statistical duration: {D_star} ± {R_star} s")
    print(f"rounded to 3 decimal places: {round_half_up(D_star, ndigits=2)} ± {round_half_up(R_star, ndigits=2)} s")

    return D_star, R_star, type

def analyze(reload_events : dict[str, list[ReloadEvent]]):
    print("--- Analysis ---")
    result = [analyze_reload_events(type, events) for type, events in reload_events.items()]
    print(f"analyzed {len(result)} reload types")
    return result


if __name__ == "__main__":
    # video_path = pathlib.Path(r"D:\clips\2026-02-22 20-52-12.mp4")
    # mode = "primary"
    # left_crop = 2108
    # top_crop = 1185
    # right_crop = 68
    # bottom_crop = 63

    cb = lambda: input("\npress enter to exit...")
    atexit.register(cb)
    args = parser.parse_args()
    video_path : pathlib.Path = args.filepath
    mode = args.mode
    left_crop = args.left_crop
    top_crop = args.top_crop
    right_crop = args.right_crop
    bottom_crop = args.bottom_crop
    dry_run = args.dry_run

    if video_path.is_file():
        video_paths = [video_path]
    else:
        video_paths = list(video_path.iterdir())
    
    for video_path in video_paths:
        assert video_path.is_file(), f"Video file not found: {video_path}"
        reload_events = process_video(video_path, mode, l=left_crop, t=top_crop, r=right_crop, b=bottom_crop)
        result = analyze(reload_events)

        if len(result) != 0:
            parent_path = pathlib.Path(__file__).parent
            weapons_dict = {path.stem: path for path in (parent_path / "weapons").glob("*.json")}
            
            # Get weapon name from video file and update corresponding JSON
            weapon_name = video_path.stem
            weapon_json_path = weapons_dict[weapon_name]
            with open(weapon_json_path, 'r') as f:
                weapon_data = json.load(f)
            
            # Update reload_times based on results
            for D_star, R_star, reload_type in result:
                assert R_star < 0.0084, f"Unreasonably high uncertainty: {R_star:.5f} s"
                assert reload_type in ("TACTICAL", "FULL"), f"Unexpected reload type: {reload_type}"
                index = 0 if reload_type == "TACTICAL" else 1
                current_value = weapon_data["reload_times"][index]
                new_value = round_half_up(D_star, 2)
                assert current_value in (None, new_value), f"new {reload_type} reload time for {weapon_name} {new_value} unequal old {current_value}"
                weapon_data["reload_times"][index] = new_value
            
            # Save the updated JSON file
            if not dry_run:
                with open(weapon_json_path, 'w') as f:
                    json.dump(weapon_data, f, indent=4)
            
                print(f"Updated {weapon_json_path}")

        print()

    cv2.destroyAllWindows()
    # atexit.unregister(cb)
    