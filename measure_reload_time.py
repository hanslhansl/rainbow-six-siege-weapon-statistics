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

# Progress bar
BAR_WIDTH = 40

# relative to rect area
MIN_NUMBER_PIXELS_RATIO = 0.08

# --- Video Processing ---
import cv2, numpy as np, sys, typing, time, av, atexit, pathlib, scipy.optimize, scipy.special
from dataclasses import dataclass

def pause():
    print()
    input("Press Enter to exit...")
atexit.register(pause)

primary = False
if len(sys.argv) == 3:
    if "--primary" in sys.argv:
        primary = True
        sys.argv.remove("--primary")
    elif "--secondary" in sys.argv:
        primary = False
        sys.argv.remove("--secondary")
    else:
        raise ValueError("Usage: python measure_reload_time.py <video_path> --primary|secondary")
assert len(sys.argv) == 2, "Usage: python measure_reload_time.py <video_path> --primary|secondary"
VIDEO_PATH = sys.argv[1]
assert pathlib.Path(VIDEO_PATH).is_file(), f"Video file not found: {VIDEO_PATH}"

@dataclass
class Status:
    type: typing.Literal["ONE", "NON_ONE", "NONE"]
    start: float
    end: float

@dataclass
class ReloadEvent:
    statistical_duration: float
    radius: float

def extract_target_mask(img, min_number_pixels):
    # Strict but tolerant RGB mask: red/black with GB noise allowed
    B, G, R = cv2.split(img)
    mask = ((G < 50) & (B < 50)).astype(np.uint8) * 255

    # Close gaps between digits
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Connected components
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num <= 1:
        return None

    crop_h = img.shape[0]
    out = np.zeros_like(mask)

    for i in range(1, num):
        h_i = stats[i, cv2.CC_STAT_HEIGHT]
        if h_i >= MIN_HEIGHT_RATIO * crop_h:
            out[labels == i] = 255

    return out if cv2.countNonZero(out) > min_number_pixels else None

def normalize(mask):
    ys, xs = np.where(mask > 0)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    crop = mask[y0:y1+1, x0:x1+1]
    return crop if crop.size > 0 else None

def is_one(norm):
    h, w = norm.shape
    aspect = w / h
    return aspect < MAX_ASPECT_RATIO_FOR_ONE

def process_video(video_path, primary):
    print("--- Video File ---")

    with av.open(video_path, options={"hwaccel": "d3d11va"}) as container:
        print(f"path: {container.name}")
        stream = container.streams.video[0]

        width = stream.width
        height = stream.height
        print(f"resolution: {width}x{height}")
        rect_x = int(rect_rel_x * width)
        rect_y = int((rect_rel_y + (0 if primary else secondary_rect_y_offset)) * height)
        rect_w = int(rect_rel_w * width)
        rect_h = int(rect_rel_h * height)
        min_number_pixels = int(rect_w * rect_h * MIN_NUMBER_PIXELS_RATIO)

        fps = float(stream.average_rate)
        print(f"fps: {fps}")
        total_duration = float(container.duration) / av.time_base
        print(f"duration: {total_duration} s")

        
        print("\n--- Video Processing ---")
        states : list[Status] = []
        reload_events : list[ReloadEvent] = []
        last_frame_time = 0
        for frame in container.decode(video=0):
            start_counter = time.perf_counter()

            # current time in seconds
            frame_time = frame.time

            # extract mask from frame
            frame = frame.to_rgb().to_ndarray(format="bgr24")  # OpenCV-friendly
            crop = frame[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w]
            mask = extract_target_mask(crop, min_number_pixels)

            # classify frame
            status = "NONE"
            if mask is None:
                display_frame = np.zeros_like(crop)
                display_frame[:] = (0, 255, 0)  # Green
            else:
                display_frame = mask
                norm = normalize(mask)
                if norm is not None:
                    status = "ONE" if is_one(norm) else "NON_ONE"

            # display mask
            # cv2.imshow("Video", crop)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # update states
            if len(states) == 0 or status != states[-1].type:
                states.append(Status(type=status, start=frame_time, end=frame_time))

                if len(states) > 2:
                    e0, e1, e2 = states[-3:]
                    if e0.type == "NON_ONE" and e1.type == "ONE" and e2.type == "NON_ONE":
                        t0, t1 = e0.end, e1.start
                        t2, t3 = e1.end, e2.start
                        assert t0 < t1 < t2 < t3, f"Timestamps must be in order: {t0}, {t1}, {t2}, {t3}"
                        S = (t0 + t1) / 2
                        E = (t2 + t3) / 2
                        D = E - S
                        R = (-t0 + t3 - t2 + t1) / 2
                        reload_events.append(ReloadEvent(statistical_duration=D, radius=R))
                        sys.stdout.write(f"\r\033[Kreload at: {S:10.6f}s → {E:10.6f}s | statistical duration: {D:.6f} ± {R:.6f} s | min/max duration: {t2 - t1:.6f}/{t3 - t0:.6f} s\n")
                        sys.stdout.flush()
            else:
                states[-1].end = frame_time

            # update progress bar
            processing_speed = (frame_time - last_frame_time) / (time.perf_counter() - start_counter)
            last_frame_time = frame_time
            filled_length = min(round(BAR_WIDTH * frame_time / total_duration), BAR_WIDTH)
            bar = '█' * filled_length + '-' * (BAR_WIDTH - filled_length)
            sys.stdout.write(f"\r\033[K|{bar}| {frame_time:.2f}/{total_duration:.2f} s | {frame_time / total_duration:.2%} | {processing_speed:.2f} s/s | classified as {status!r}")
            sys.stdout.flush()

    print()
    return reload_events

reload_events = process_video(VIDEO_PATH, primary)



# --- Analysis ---
print("\n--- Analysis ---")

# cost function for M-estimator
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