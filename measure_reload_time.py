# --- CONFIG ---
rect_rel_x = 0.865
rect_rel_y = 0.872
rect_rel_w = 0.042
rect_rel_h = 0.043
secondary_rect_y_offset = 0.035

# Blob filtering
MIN_HEIGHT_RATIO = 0.8   # relative to crop height

# ONE detection
MAX_ASPECT_RATIO_FOR_ONE = 19 / 44  # width / height

# ZERO detection
MIN_RELATIVE_HOLE_CENTER_OFFSET_FOR_ZERO = 0.1
MIN_RELATIVE_HOLE_HEIGHT_FOR_ZERO = 0.6

# Progress bar
BAR_WIDTH = 40

# relative to rect area
MIN_NUMBER_PIXELS_RATIO = 0.08


# --- CODE ---
import cv2, numpy as np, sys, typing, time, av, atexit, pathlib, scipy.optimize, scipy.special, argparse
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
atexit.register(lambda: input("\nPress Enter to exit..."))


# --- Argument parsing ---
parser = argparse.ArgumentParser(
    description="scan a video file for reload animations and measure the duration"
)

# Positional argument
parser.add_argument(
    "filepath",
    help="path to the video file"
)

# Group 1: primary vs secondary
role_group = parser.add_mutually_exclusive_group(required=True)
role_group.add_argument(
    "--primary",
    action="store_true",
    help="primary weapon"
)
role_group.add_argument(
    "--secondary",
    action="store_false",
    help="secondary weapon"
)

# Group 2: tactical vs full
mode_group = parser.add_mutually_exclusive_group(required=True)
mode_group.add_argument(
    "--tactical",
    action="store_true",
    help="tactical reload"
)
mode_group.add_argument(
    "--full",
    action="store_false",
    help="full reload"
)


# --- Video processing ---
@dataclass
class Status:
    type: typing.Literal["TARGET", "ANY", "NONE"]
    start: float
    end: float

@dataclass
class ReloadEvent:
    statistical_duration: float
    radius: float

def process_rect(img, tactical):
    # Strict but tolerant RGB mask: red/black with GB noise allowed
    B, G, R = cv2.split(img)
    mask = ((G < 0x40) & (B < 0x40)).astype(np.uint8) * 255

    # morphological closing to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morphed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Connected components
    num, labels, stats, _ = cv2.connectedComponentsWithStats(morphed_mask)
    if num <= 1:
        return "NONE", "found no blobs", mask, morphed_mask

    # filter out small blobs
    crop_h = img.shape[0]
    cleaned_mask = np.zeros_like(mask)
    valid_blobs = 0
    for i in range(1, num):
        h_i = stats[i, cv2.CC_STAT_HEIGHT]
        if h_i >= MIN_HEIGHT_RATIO * crop_h:
            cleaned_mask[labels == i] = 255
            valid_blobs += 1
        
    if valid_blobs == 0:
        return "NONE", "no blobs passed height filter", mask, morphed_mask

    # crop to bounding box of all blobs
    nonzero_points = cv2.findNonZero(cleaned_mask)
    x, y, w, h = cv2.boundingRect(nonzero_points)
    cropped_mask = cleaned_mask[y:y+h, x:x+w]

    h, w = cropped_mask.shape
    status = "ANY"
    reason = f"valid blobs: {valid_blobs}"
    if valid_blobs == 1:
        if tactical:
            aspect = w / h
            if aspect <= MAX_ASPECT_RATIO_FOR_ONE:
                status = "TARGET"
            reason = f"aspect ratio: {aspect:.4f}"
        else:
            # find contours
            contours, hierarchy = cv2.findContours(
                cropped_mask,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE
            )
            reason = f"contours: {len(contours)}"
            
            # check if there's exactly one hole
            if len(contours) == 2:
                hole, = [cnt for i, cnt in enumerate(contours) if hierarchy[0][i][3] != -1]

                hx, hy, hw, hh = cv2.boundingRect(hole)
                hx = hx + hw / 2
                hy = hy + hh / 2
                hole_center_relative_x_offset = abs(hx/w - 0.5)
                hole_center_relative_y_offset = abs(hy/h - 0.5)
                reason = f"hole center rel offset: ({hole_center_relative_x_offset:.4f}, {hole_center_relative_y_offset:.4f})"

                # check if the hole is centered
                if hole_center_relative_x_offset <= MIN_RELATIVE_HOLE_CENTER_OFFSET_FOR_ZERO and hole_center_relative_y_offset <= MIN_RELATIVE_HOLE_CENTER_OFFSET_FOR_ZERO:

                    hole_height_ratio = hh / h
                    reason = f"rel hole height: ({hole_height_ratio:.4f})"

                    # check if the hole is high enough
                    if hole_height_ratio >= MIN_RELATIVE_HOLE_HEIGHT_FOR_ZERO:
                        status = "TARGET"

    return status, reason, mask, morphed_mask, cleaned_mask

def process_video(video_path, primary, tactical):
    print("--- Video File ---")

    with av.open(video_path) as container:
        print(f"path: {container.name}")
        stream = container.streams.video[0]

        width = stream.width
        height = stream.height
        print(f"resolution: {width}x{height}")
        rect_x = int(rect_rel_x * width)
        rect_y = int((rect_rel_y + (0 if primary else secondary_rect_y_offset)) * height)
        rect_w = int(rect_rel_w * width)
        rect_h = int(rect_rel_h * height)

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
            status, reason, *masks = process_rect(rect, tactical)

            # update states
            frame_time = frame.time
            if len(states) == 0 or status != states[-1].type:
                states.append(Status(type=status, start=frame_time, end=frame_time))

                if len(states) > 2:
                    e0, e1, e2 = states[-3:]
                    if e0.type == "ANY" and e1.type == "TARGET" and e2.type == "ANY":
                        t0, t1 = e0.end, e1.start
                        t2, t3 = e1.end, e2.start
                        assert t0 < t1 < t2 < t3, f"Timestamps must be in order: {t0}, {t1}, {t2}, {t3}"
                        S = (t0 + t1) / 2
                        E = (t2 + t3) / 2
                        D = E - S
                        R = (-t0 + t3 - t2 + t1) / 2
                        reload_events.append(ReloadEvent(statistical_duration=D, radius=R))
                        sys.stdout.write(f"\r\033[Kreload: {S:10.6f}s → {E:10.6f}s | Δt: {D:.6f} ± {R:.6f} s | min/max Δt: {t2 - t1:.6f}/{t3 - t0:.6f} s\n")
                        sys.stdout.flush()
            else:
                states[-1].end = frame_time

            # update progress bar
            filled_length = min(round(BAR_WIDTH * frame_time / total_duration), BAR_WIDTH)
            bar = '█' * filled_length + '-' * (BAR_WIDTH - filled_length)
            sys.stdout.write(f"\r\033[K|{bar}| {frame_time:.2f}/{total_duration:.2f} s | {frame_time / total_duration:.2%} | {status!r} ({reason})")
            sys.stdout.flush()
            
            # display mask
            # cv2.imshow("Video", np.hstack((rect, *(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) for m in masks))))
            # if cv2.waitKey(0 if status == "TARGET" else 0) & 0xFF == ord('q'):
            #     break

    cv2.destroyAllWindows()
    print()
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

def analyze_reload_events(reload_events : list[ReloadEvent]):
    print("\n--- Analysis ---")

    # Run optimization, initial guess: weighted average
    x0 = np.mean([m.statistical_duration for m in reload_events])

    res = scipy.optimize.minimize(
        interval_cost,
        x0=[x0],
        args=(reload_events,),
        method='Nelder-Mead',  # robust 1D optimizer
        options={'xatol':1e-9, 'disp': True}
    )
    D_star = res.x[0]

    # Estimate effective radius (uncertainty), max violation after robust estimate
    R_star = max(max(0.0, abs(D_star - m.statistical_duration) - m.radius) for m in reload_events)

    print(f"Statistical reload duration: {D_star} ± {R_star} s")
    def round_half_up(fl, ndigits=0):
        q = Decimal('1.' + '0' * ndigits)
        return float(Decimal.from_float(fl).quantize(q, rounding=ROUND_HALF_UP))
    print(f"Rounded to 3 decimal places: {round_half_up(D_star, ndigits=3)} ± {round_half_up(R_star, ndigits=3)} s")


if __name__ == "__main__":
    args = parser.parse_args()
    VIDEO_PATH = pathlib.Path(args.filepath)
    assert VIDEO_PATH.is_file(), f"Video file not found: {VIDEO_PATH}"
    reload_events = process_video(VIDEO_PATH, args.primary, args.tactical)
    analyze_reload_events(reload_events)

    print(f"\nfinished scanning for {'primary' if args.primary else 'secondary'} weapon {'tactical' if args.tactical else 'full'} reloads in video: {VIDEO_PATH}")


"""

    "reload_times" : [null, null]


,
    "reload_times" : [null, null, null, null]

"""