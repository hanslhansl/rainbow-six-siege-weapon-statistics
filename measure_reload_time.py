# --- CONFIG ---
NORMALIZED_RECT_X = 0.830
NORMALIZED_RECT_Y = 0.872
NORMALIZED_RECT_W = 0.132
NORMALIZED_RECT_H = 0.043
NORMALIZED_SECONDARY_RECT_Y_OFFSET = 0.035
NORMALIZED_TERTIARY_RECT_Y_OFFSET = -0.042

# Blob filtering
MIN_NORMALIZED_HEIGHT_FOR_BLOB = 0.8   # relative to crop height

# ZERO detection
MAX_NORMALIZED_HOLE_CENTER_OFFSET_FOR_ZERO_FOUR = 0.1
MIN_NORMALIZED_HOLE_HEIGHT_FOR_ZERO = 0.6

# ONE detection
MAX_ASPECT_RATIO_FOR_ONE = 19 / 44  # width / height

# Progress bar
PROGRESS_BAR_WIDTH = 40


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
class Status:
    value: int | None
    start: float
    end: float

@dataclass
class ReloadEvent:
    statistical_duration: float
    radius: float
    type : typing.Literal["TACTICAL", "FULL"]

def pad(arr, relative_to):
    return np.pad(arr, ((0, relative_to.shape[0]-arr.shape[0]), (0, 0)), mode='constant', constant_values=127)

def crop_padding(mask):
    """crop to bounding box"""
    nonzero_points = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(nonzero_points)
    cropped = mask[y:y+h, x:x+w]
    return cropped

def analyze_skeleton(skeleton):
    """Analyze skeleton topology: count endpoints and junctions
    Assumes skeleton already has 1-pixel border padding"""
    h, w = skeleton.shape
    
    endpoints = 0
    junctions = 0
    
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
                elif neighbors >= 3:
                    junctions += 1
    
    return endpoints, junctions

def classify_digit(cropped_mask):
    h, w = cropped_mask.shape
    aspect_ratio = w / h

    # check apsect ratio, 1 is narrowest digit
    if aspect_ratio <= MAX_ASPECT_RATIO_FOR_ONE:
        return 1, f"aspect ratio: {aspect_ratio:.4f}"
    
    # find contours
    contours, hierarchy = cv2.findContours(
        cropped_mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )
    no_of_holes = len(contours) - 1

    # if there are 2 holes it's 8
    if no_of_holes == 2:
        return 8, f"#holes: {no_of_holes}"
    
    # if there is 1 hole it's 0, 4, 6, or 9
    if no_of_holes == 1:
        hole, = [cnt for i, cnt in enumerate(contours) if hierarchy[0][i][3] != -1]
        hx, hy, hw, hh = cv2.boundingRect(hole)
        hx = hx + hw / 2
        hy = hy + hh / 2

        normalized_hole_center_y = hy/h - 0.5
        if normalized_hole_center_y < -MAX_NORMALIZED_HOLE_CENTER_OFFSET_FOR_ZERO_FOUR:
            return 9, f"hole center normalized y: ({normalized_hole_center_y:.4f})"
        if normalized_hole_center_y > MAX_NORMALIZED_HOLE_CENTER_OFFSET_FOR_ZERO_FOUR:
            return 6, f"hole center normalized y: ({normalized_hole_center_y:.4f})"

        normalized_hole_height = hh / h
        return 0 if normalized_hole_height >= MIN_NORMALIZED_HOLE_HEIGHT_FOR_ZERO else 4, f"#normalized hole height: {normalized_hole_height}"

    # 3 has 3 endpoints
    padded_mask = cv2.copyMakeBorder(cropped_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    skeleton = cv2.ximgproc.thinning(padded_mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    endpoints, junctions = analyze_skeleton(skeleton)
    if endpoints == 3:
        return 3, f"endpoints: {endpoints}, junctions: {junctions}", skeleton

    num_samples = 50

    # match 7
    p0 = (0.15 * w, 0.08 * h)
    p1 = (0.85 * w, 0.08 * h)
    p2 = (0.4 * w, 0.92 * h)
    points = (
        (p0, p1),
        (p1, p2),
    )
    lines = [np.linspace(p0, p1, num_samples).astype(int) for p0, p1 in points]
    # seven = cropped_mask.copy()
    # for line in lines:
    #     seven[line[:, 1], line[:, 0]] = 0
    if all(np.all(cropped_mask[line[:, 1], line[:, 0]] == 255) for line in lines):
        return 7, f"matched 7"
    
    # match 5
    p0 = (0.85 * w, 0.07 * h)
    p1 = (0.21 * w, 0.07 * h)
    p2 = (0.21 * w, 0.4 * h)
    p3 = (0.85 * w, 0.5 * h)
    p4 = (0.85 * w, 0.8 * h)
    p5 = (0.7 * w, 0.93 * h)
    p6 = (0.3 * w, 0.93 * h)
    points = (
        (p0, p1),
        (p1, p2),
        (p3, p4),
        (p5, p6),
    )
    lines = [np.linspace(p0, p1, num_samples).astype(int) for p0, p1 in points]
    # five = cropped_mask.copy()
    # for line in lines:
    #     five[line[:, 1], line[:, 0]] = 0
    if all(np.all(cropped_mask[line[:, 1], line[:, 0]] == 255) for line in lines):
        return 5, f"matched 5"


    return -2, f"no digit found"

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
        h_i = stats[i, cv2.CC_STAT_HEIGHT]
        if h_i >= MIN_NORMALIZED_HEIGHT_FOR_BLOB * h:
            cleaned_mask[labels == i] = 255
            valid_blobs += 1
    cleaned_mask = crop_padding(cleaned_mask)
        
    if valid_blobs == 0:
        return None, "no blobs passed height filter", mask, morphed_mask

    output = []
    digit = -1
    reason = f"valid blobs: {valid_blobs}"
    if valid_blobs == 1:
        digit, reason, *output = classify_digit(cleaned_mask)

    return digit, reason, mask, morphed_mask, cleaned_mask, *output

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
        states : list[Status] = []
        reload_events : list[ReloadEvent] = []
        
        for frame in container.decode(video=0):
            # crop frame
            rect = frame.to_rgb().to_ndarray(format="bgr24")[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w]

            # get status from rect
            digit, reason, *masks = process_rect(rect)

            # update states
            frame_time = frame.time
            if len(states) == 0 or digit != states[-1].value:
                states.append(Status(value=digit, start=frame_time, end=frame_time))

                type = None

                if len(states) >= 3:
                    e2, e1, e0 = states[-3:]
                    if e2.value not in (None, 0, 1) and e1.value in (0, 1) and e0.value not in (None, 0, 1): # tactical
                        t0, t1 = e2.end, e1.start
                        t2, t3 = e1.end, e0.start
                        type = "TACTICAL"

                if len(states) >= 4:
                    e3, e2, e1, e0 = states[-4:]
                    if e3.value not in (None, 0, 1) and e2.value == 1 and e1.value == 0 and e0.value not in (None, 0): # full
                        t0, t1 = e2.end, e1.start
                        t2, t3 = e1.end, e0.start
                        type = "FULL"

                if type is not None:
                    t0, t1 = e2.end, e1.start
                    t2, t3 = e1.end, e0.start
                    assert t0 < t1 < t2 < t3, f"Timestamps must be in order: {t0}, {t1}, {t2}, {t3}"
                    S = (t0 + t1) / 2
                    E = (t2 + t3) / 2
                    D = E - S
                    R = (-t0 + t3 - t2 + t1) / 2
                    reload_events.append(ReloadEvent(statistical_duration=D, radius=R, type=type))
                    sys.stdout.write(f"\r\033[K{type} reload: {S:10.6f}s → {E:10.6f}s | Δt: {D:.6f} ± {R:.6f} s | min/max Δt: {t2 - t1:.6f}/{t3 - t0:.6f} s\n")
                    sys.stdout.flush()
            else:
                states[-1].end = frame_time

            # update progress bar
            filled_length = min(round(PROGRESS_BAR_WIDTH * frame_time / total_duration), PROGRESS_BAR_WIDTH)
            bar = '█' * filled_length + '-' * (PROGRESS_BAR_WIDTH - filled_length)
            sys.stdout.write(f"\r\033[K|{bar}| {frame_time:.4f}/{total_duration:.4f} s | {frame_time / total_duration:.2%} | {digit!r} ({reason})")
            sys.stdout.flush()
            
            # display mask
            if digit == -2:
            # if True:
                separator = np.full((rect.shape[0], 2, rect.shape[2]), (0,255,0), dtype=rect.dtype)
                chain = itertools.chain.from_iterable((cv2.cvtColor(pad(m, rect), cv2.COLOR_GRAY2BGR), separator) for m in masks)
                cv2.imshow("Video", np.hstack((rect, separator, *chain)))
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()
    print()
    print(f"found {len(reload_events)} {'primary' if mode == 'primary' else 'secondary' if mode == 'secondary' else 'tertiary'} weapon reloads")
    return reload_events


# --- Analysis ---
def interval_cost(D, measurements, sigma=0.0005, delta=1.5):
    """
    D: current duration estimate (scalar)
    measurements: list of MeasuredInterval
    sigma: small softening for minimal violations
    delta: Huber parameter
    """
    total = 0.0
    for m in measurements:
        # compute interval violation
        v = max(0.0, abs(D - m.statistical_duration) - m.radius)
        # scale by sigma and apply Huber
        total += scipy.special.huber(delta, v / sigma)
    return total

def round_half_up(fl, ndigits=0):
    q = Decimal('1.' + '0' * ndigits)
    return float(Decimal.from_float(fl).quantize(q, rounding=ROUND_HALF_UP))

def analyze_reload_events(reload_events : list[ReloadEvent]):
    print("\n--- Analysis ---")

    groups : collections.defaultdict[str, list[ReloadEvent]] = collections.defaultdict(list)
    for event in reload_events:
        groups[event.type].append(event)
    event_groups = list(groups.values())

    result : list[tuple[float, float, str]] = []

    for event_group in event_groups:
        type = event_group[0].type
        assert len(event_group) >= 7, f"Not enough measurements for {type} reload: {len(event_group)} (need at least 7 for robust estimation)"

        # Run optimization, initial guess: weighted average
        x0 = np.mean([m.statistical_duration for m in event_group])

        res = scipy.optimize.minimize(
            interval_cost,
            x0=[x0],
            args=(event_group,),
            method='Nelder-Mead',  # robust 1D optimizer
            options={'xatol':1e-9, 'disp': False}
        )
        D_star = res.x[0]

        # Estimate effective radius (uncertainty), max violation after robust estimate
        R_star = max(max(0.0, abs(D_star - m.statistical_duration) - m.radius) for m in event_group)

        print(f"{len(event_group)} {type} reloads, statistical duration: {D_star} ± {R_star} s")
        print(f"rounded to 3 decimal places: {round_half_up(D_star, ndigits=2)} ± {round_half_up(R_star, ndigits=2)} s\n")

        result.append((D_star, R_star, type))

    print(f"analyzed {len(event_groups)} reload types")
    return result


if __name__ == "__main__":
    cb = lambda: input("\npress enter to exit...")
    # atexit.register(cb)
    # args = parser.parse_args()
    video_path = pathlib.Path(r"D:\clips\2026-02-22 20-52-12.mp4") # args.filepath
    mode = "primary" # args.mode
    left_crop = 2108 # args.left_crop
    top_crop = 1185 # args.top_crop
    right_crop = 68 # args.right_crop
    bottom_crop = 63 # args.bottom_crop

    assert video_path.is_file(), f"Video file not found: {video_path}"
    reload_events = process_video(video_path, mode, l=left_crop, t=top_crop, r=right_crop, b=bottom_crop)
    result = analyze_reload_events(reload_events)

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
    

"""

    "reload_times" : [null, null]


,
    "reload_times" : [null, null, null, null]

"""