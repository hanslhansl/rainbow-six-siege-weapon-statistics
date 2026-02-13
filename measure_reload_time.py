import cv2, numpy as np, sys, typing, time, av
from dataclasses import dataclass

print(sys.argv)

VIDEO_PATH = r"D:\clips\Tom Clancy's Rainbow Six Siege 2026\vigil crazy flick - border.mp4"


# --- CONFIG ---
rect_rel_x = 0.865
rect_rel_y = 0.872
rect_rel_w = 0.042
rect_rel_h = 0.043

# Blob filtering
MIN_HEIGHT_RATIO = 0.8   # relative to crop height

# ONE detection
MAX_ASPECT_RATIO_FOR_ONE = 19 / 44  # width / height

# Progress bar
BAR_WIDTH = 40

# --- FUNCTIONS ---

def extract_target_mask(img):
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

# --- MAIN PROCESS ---
print("--- Video File ---")
container = av.open(
    VIDEO_PATH,
    #options={"hwaccel": "d3d11va"}  # AMD / Windows hardware decode
)
print(f"path: {container.name}")
stream = container.streams.video[0]

width = stream.width
height = stream.height
print(f"resolution: {width}x{height}")
rect_x = int(rect_rel_x * width)
rect_y = int(rect_rel_y * height)
rect_w = int(rect_rel_w * width)
rect_h = int(rect_rel_h * height)
min_number_pixels = int(rect_w * rect_h * 0.08)

fps = float(stream.average_rate)
print(f"fps: {fps}")
total_duration = float(container.duration) / av.time_base
print(f"duration: {total_duration} s")

@dataclass
class Status:
    type: typing.Literal["ONE", "NON_ONE", "NONE"]
    start: float
    end: float

@dataclass
class ReloadEvent:
    statistical_duration: float
    radius: float

print("\n--- Video Processing ---")
states : list[Status] = []
reload_events : list[ReloadEvent] = []
last_frame_time = 0
for frame in container.decode(video=0):
    start_counter = time.perf_counter()

    frame_time = frame.time # current time in seconds

    # extract mask from frame
    frame = frame.to_rgb().to_ndarray(format="bgr24")  # OpenCV-friendly
    crop = frame[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w]
    mask = extract_target_mask(crop)

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
    # cv2.imshow("Video", display_frame)
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
                sys.stdout.write(f"\r\033[Kreload at: {S}s → {E}s | statistical duration: {D} ± {R} s | min/max duration: {t2 - t1}/{t3 - t0} s\n")
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

print("\n")


print("\n--- Analysis ---")
lower = max(re.statistical_duration - re.radius for re in reload_events)
upper = min(re.statistical_duration + re.radius for re in reload_events)
D_star = 0.5 * (lower + upper)
R_star = 0.5 * abs(upper - lower)
if lower <= upper:
    print(f"Statistical reload duration: {D_star} ± {R_star} s")
else:
    print(f"No consistent duration — intervals do not overlap exactly.")
    print(f"Estimated reload duration: {D_star} ± {R_star} s")