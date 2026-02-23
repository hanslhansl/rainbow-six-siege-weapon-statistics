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
    help="path to the video file"
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

def analyze_skeleton(mask):
    """Analyze skeleton topology: count endpoints
    Assumes skeleton already has 1-pixel border padding"""

    padded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    skeleton = cv2.ximgproc.thinning(padded_mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    h, w = skeleton.shape
    
    endpoints = 0
    
    # Check each white pixel (not on border)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton[y, x] > 0:
                # Count neighbors in 8-connectivity (convert to int to avoid overflow)
                neighbors = (
                    int(skeleton[y-1, x-1]) + int(skeleton[y-1, x]) + int(skeleton[y-1, x+1]) +
                    int(skeleton[y, x-1]) + int(skeleton[y, x+1]) +
                    int(skeleton[y+1, x-1]) + int(skeleton[y+1, x]) + int(skeleton[y+1, x+1])
                ) // 255
                
                if neighbors == 1:
                    endpoints += 1
    
    return endpoints, skeleton

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


    # 3 has 3 endpoints
    endpoints, skeleton = analyze_skeleton(cropped_mask)
    if endpoints == 3:
        return 3, f"{endpoints=}", skeleton

    # 2, 5, 7 have 2 endpoints
    if endpoints != 2:
        raise ValueError(f"Unable to classify digit with aspect ratio: {aspect_ratio:.4f}, holes: {number_of_holes}, endpoints: {endpoints}")


    def match(area, area_range, points, num_samples = 50):
        if not (area_range[0] <= area <= area_range[1]):
            return False
        points = [((x0*w, y0*h), (x1*w, y1*h)) for (x0, y0), (x1, y1) in points]
        lines = [np.linspace(p0, p1, num_samples).astype(int) for p0, p1 in points]
        return all(np.all(cropped_mask[line[:, 1], line[:, 0]] == 255) for line in lines)


    normalized_area = mass / (255 * w * h)

    if match(normalized_area, (0.45, 0.5), [
        ((0.15, 0.08), (0.85, 0.08)),
        ((0.85, 0.08), (0.4, 0.92)),
        ]):
        return 7, "matched 7"
    
    if match(normalized_area, (0.59, 0.64), [
        ((0., 0.), (0., 0.)),
        ]):
        return 5, "matched 5"
    
    if match(normalized_area, (0.57, 0.59), [
        ((0., 0.), (0., 0.)),
        ]):
        return 2, "matched 2"
    


    mass = np.sum(cropped_mask)
    y_coords, x_coords = np.indices((h, w))
    normalized_center_x = np.sum(x_coords * cropped_mask) / mass / w
    normalized_center_y = np.sum(y_coords * cropped_mask) / mass / h

    digit = None
    if 0.45 <= normalized_area <= 0.5 and 0.52 <= normalized_center_x <= 0.55 and 0.39 <= normalized_center_y <= 0.41:
        digit = 7
    if 0.59 <= normalized_area <= 0.64 and 0.485 <= normalized_center_x <= 0.505 and 0.48 <= normalized_center_y <= 0.5:
        digit = 5
    if 0.57 <= normalized_area <= 0.59 and 0.48 <= normalized_center_x <= 0.505 and 0.49 <= normalized_center_y <= 0.51:
        digit = 2
    
    if digit is not None:
        return digit, f"{normalized_area:.4f} {normalized_center_x:.4f} {normalized_center_y:.4f}"

    raise ValueError(f"Unable to classify digit with area: {normalized_area:.4f}, center: ({normalized_center_x:.4f}, {normalized_center_y:.4f})")

def process_rect(img):
    # Strict but tolerant RGB mask: red/black with GB noise allowed
    B, G, R = cv2.split(img)
    mask = crop_padding(((G < 0x40) & (B < 0x40)).astype(np.uint8) * 255)

    # either all black or all white means mask is empty
    if np.all(mask == 0) or np.all(mask != 0):
        return None, "mask is empty"

    # morphological closing to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morphed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Connected components
    num, labels, stats, _ = cv2.connectedComponentsWithStats(morphed_mask)
    if num <= 1:
        return None, "found no blobs", mask, morphed_mask

    # filter out small blobs
    h = img.shape[0]
    cleaned_mask = np.zeros_like(mask)
    valid_blobs = 0
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_HEIGHT] / h >= MIN_NORMALIZED_HEIGHT_FOR_BLOB:
            cleaned_mask[labels == i] = 255
            valid_blobs += 1
    cleaned_mask = crop_padding(cleaned_mask)
        
    if valid_blobs == 0:
        return None, "no blobs passed height filter", mask, morphed_mask

    if valid_blobs == 1:
        digit, reason, *output = classify_digit(cleaned_mask)
        return digit, reason, mask, morphed_mask, cleaned_mask, *output

    return -1, f"{valid_blobs=}", mask, morphed_mask, cleaned_mask

def process_video(video_path, mode, l = 0, t = 0, r = 0, b = 0):
    print("\n--- Video File ---")

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

        fps = float(stream.average_rate)
        print(f"fps: {fps}")
        total_duration = float(container.duration) / av.time_base
        print(f"duration: {total_duration} s")


        print("\n--- Video Processing ---")
        ammo_counter : list[Number] = []
        reload_events = collections.defaultdict[typing.Literal["TACTICAL", "FULL"], list[ReloadEvent]](list)
        
        for frame in container.decode(video=0):
            # crop frame
            rect = frame.to_rgb().to_ndarray(format="bgr24")[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w]

            # get status from rect
            number, reason, *masks = process_rect(rect)

            # update states
            frame_time = frame.time
            if len(ammo_counter) == 0 or number != ammo_counter[-1].value:
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
                    assert t0 < t1 < t2 < t3, f"Timestamps must be in order: {t0}, {t1}, {t2}, {t3}"
                    S = (t0 + t1) / 2
                    E = (t2 + t3) / 2
                    D = E - S
                    R = (-t0 + t3 - t2 + t1) / 2
                    reload_events[type].append(ReloadEvent(statistical_duration=D, radius=R))
                    sys.stdout.write(f"\r\033[K{type} reload: {S:10.6f}s → {E:10.6f}s | Δt: {D:.6f} ± {R:.6f} s | min/max Δt: {t2 - t1:.6f}/{t3 - t0:.6f} s\n")
                    sys.stdout.flush()
            else:
                ammo_counter[-1].end = frame_time

            # update progress bar
            filled_length = min(round(PROGRESS_BAR_WIDTH * frame_time / total_duration), PROGRESS_BAR_WIDTH)
            bar = '█' * filled_length + '-' * (PROGRESS_BAR_WIDTH - filled_length)
            sys.stdout.write(f"\r\033[K|{bar}| {frame_time:.4f}/{total_duration:.4f} s | {frame_time / total_duration:.2%} | {number!r} ({reason})")
            sys.stdout.flush()
            
            # display mask
            # if digit in -2:
            if True:
                separator = np.full((rect.shape[0], 2, rect.shape[2]), (0,255,0), dtype=rect.dtype)
                chain = itertools.chain.from_iterable((cv2.cvtColor(pad(m, rect), cv2.COLOR_GRAY2BGR), separator) for m in masks)
                cv2.imshow("Video", np.hstack((rect, separator, *chain)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()
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
    print(f"rounded to 3 decimal places: {round_half_up(D_star, ndigits=2)} ± {round_half_up(R_star, ndigits=2)} s\n")

    return D_star, R_star, type

def analyze(reload_events : dict[str, list[ReloadEvent]]):
    print("\n--- Analysis ---")
    result = [analyze_reload_events(type, events) for type, events in reload_events.items()]
    print(f"analyzed {len(result)} reload types")
    return result


if __name__ == "__main__":
    cb = lambda: input("\npress enter to exit...")
    atexit.register(cb)
    args = parser.parse_args()
    video_path = args.filepath
    mode = args.mode
    left_crop = args.left_crop
    top_crop = args.top_crop
    right_crop = args.right_crop
    bottom_crop = args.bottom_crop

    # video_path = pathlib.Path(r"D:\clips\2026-02-22 20-52-12.mp4")
    # mode = "primary"
    # left_crop = 2108
    # top_crop = 1185
    # right_crop = 68
    # bottom_crop = 63

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
            assert R_star < 0.0084, f"Unreasonably high uncertainty: {R_star:.3f} s"
            assert reload_type in ("TACTICAL", "FULL"), f"Unexpected reload type: {reload_type}"
            index = 0 if reload_type == "TACTICAL" else 1
            assert weapon_data["reload_times"][index] is None, f"Reload time for {reload_type} reload already set for {weapon_name}: {weapon_data['reload_times'][index]}"
            weapon_data["reload_times"][index] = round_half_up(D_star, 2)
        
        # Save the updated JSON file
        with open(weapon_json_path, 'w') as f:
            json.dump(weapon_data, f, indent=4)
        
        print(f"Updated {weapon_json_path}")

        atexit.unregister(cb)
    