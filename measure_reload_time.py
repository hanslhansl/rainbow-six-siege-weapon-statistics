
import numpy as np, sys, pathlib, cv2
import cv2 as cv
from matplotlib import pyplot as plt
from dataclasses import dataclass


img_path = "test1.png"
DIGIT_TEMPLATES_DIRECTORY = [pth for pth in (pathlib.Path(__file__).parent / "digit_templates").iterdir()]
MATCH_THRESHOLD = 0.8  # Treffer-Schwelle
scales = [1.0]  # Multi-Scale



# --- Non-Maximum Suppression ---
def non_max_suppression(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    pick = []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = idxs[-1]
        i = last
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:-1]])
        yy1 = np.maximum(y1[i], y1[idxs[:-1]])
        xx2 = np.minimum(x2[i], x2[idxs[:-1]])
        yy2 = np.minimum(y2[i], y2[idxs[:-1]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:-1]]
        idxs = np.delete(
            idxs,
            np.concatenate(([len(idxs)-1], np.where(overlap > overlapThresh)[0]))
        )

    return boxes[pick].astype("int")

def non_max_suppression_per_digit(all_boxes : list[MatchResult], digit_labels):
    final_boxes : list[MatchResult] = []
    for digit in digit_labels:
        digit_boxes = [b for b in all_boxes if b.label == digit]
        if digit_boxes:
            boxes_array = np.array([b.box for b in digit_boxes])
            pick = non_max_suppression(boxes_array)
            for i in range(len(pick)):
                final_boxes.append(MatchResult(
                    box=pick[i],
                    label=digit,
                    score=digit_boxes[i].score
                ))
    return final_boxes

# --- Template Matching ---

@dataclass
class MatchResult:
    box : list[int]
    label : str
    score : typing.Any

def match_template(gray_img, template, scale, label):
    w = int(template.shape[1] * scale)
    h = int(template.shape[0] * scale)
    resized_template = cv2.resize(template, (w, h))

    res = cv2.matchTemplate(gray_img, resized_template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= MATCH_THRESHOLD)

    return [MatchResult(
            box=[pt[0], pt[1], pt[0]+w, pt[1]+h],
            label=label,
            score=res[pt[1], pt[0]]
        ) for pt in zip(*loc[::-1])]

def match_template_multi_scale(gray_img, template, label, scales):
    return [r for scale in scales for r in match_template(gray_img, template, scale, label)]

def match_templates_multi_scale(gray_img, templates, labels, scales):
    return [x for template, label in zip(templates, labels) for x in match_template_multi_scale(gray_img, template, label, scales)]

def detect_number(final_boxes : list[MatchResult]):
    return "".join(b.label for b in final_boxes)

def highlight_boxes(final_boxes : list[MatchResult], img):
    output = img.copy()
    for b in final_boxes:
        print(*b.box)
        x1, y1, x2, y2 = b.box
        cv2.rectangle(output, (x1, y1), (x2, y2), (0,0,255), 2)
    return output


def test():
    # --- Bild laden ---
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    digit_labels = [pth.stem for pth in DIGIT_TEMPLATES_DIRECTORY]

    all_boxes = match_templates_multi_scale(gray, [cv2.imread(pth, cv2.IMREAD_GRAYSCALE) for pth in DIGIT_TEMPLATES_DIRECTORY], digit_labels, scales)

    # --- Nach X-Position sortieren ---
    final_boxes = sorted(non_max_suppression_per_digit(all_boxes, digit_labels), key=lambda x: x.box[0])

    # --- Ergebnis ausgeben ---
    detected_number = detect_number(final_boxes)
    print("Erkannte Zahl:", detected_number)

    # --- Treffer markieren ---
    cv2.imshow("Detected Digits", highlight_boxes(final_boxes, img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


test()









sys.exit()







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

def process_rect_old(img, min_number_pixels):
    empty = np.full(img.shape[:2], 255, dtype=np.uint8)

    # Convert to LAB and split channels
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Blackhat to highlight dark details on bright background
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,50))
    blackhat = cv2.morphologyEx(l, cv2.MORPH_BLACKHAT, kernel)
    mask = cv2.adaptiveThreshold(
        blackhat,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        51,
        -10
    )

    # Morphological operations to reduce noise
    kernel = np.ones((3,3), np.uint8)
    morphed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Strict but tolerant RGB mask: red/black with GB noise allowed
    # B, G, R = cv2.split(img)
    # mask = ((G < 50) & (B < 50)).astype(np.uint8) * 255

    # Close gaps between digits
    # kernel = np.ones((3, 3), np.uint8)
    # morphed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # morphed_mask = mask

    # Connected components
    num, labels, stats, _ = cv2.connectedComponentsWithStats(morphed_mask)
    if num <= 1:
        return "NONE", "found no blobs", mask, morphed_mask, empty

    crop_h = img.shape[0]
    cleaned_mask = np.zeros_like(morphed_mask)
    valid_blobs = 0
    for i in range(1, num):
        h_i = stats[i, cv2.CC_STAT_HEIGHT]
        if h_i >= MIN_HEIGHT_RATIO * crop_h:
            cleaned_mask[labels == i] = 255
            valid_blobs += 1
        
    if valid_blobs == 0:
        return "NONE", "no blobs passed height filter", mask, morphed_mask, empty

    # if cv2.countNonZero(cleaned_mask) < min_number_pixels:
    #     return "NONE", mask, morphed_mask, empty
    
    nonzero_points = cv2.findNonZero(cleaned_mask)  # finds all non-zero pixels
    x, y, w, h = cv2.boundingRect(nonzero_points)
    cropped_mask = cleaned_mask[y:y+h, x:x+w]

    # if cropped_mask.size == 0:
    #     return "NONE", "cropped mask is empty", mask, morphed_mask, cleaned_mask

    h, w = cropped_mask.shape
    aspect = w / h
    return "ONE" if aspect < MAX_ASPECT_RATIO_FOR_ONE else "NON_ONE", f"aspect ratio: {aspect:.2f}", mask, morphed_mask, cleaned_mask

templates = [cv2.imread(pth, cv2.IMREAD_GRAYSCALE) for pth in DIGIT_TEMPLATES_DIRECTORY]
digit_labels = [pth.stem for pth in DIGIT_TEMPLATES_DIRECTORY]
def process_rect(img, min_number_pixels):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    all_boxes = match_templates_multi_scale(gray, templates, digit_labels, scales)

    # --- Nach X-Position sortieren ---
    final_boxes = sorted(non_max_suppression_per_digit(all_boxes, digit_labels), key=lambda x: x.box[0])

    # --- Ergebnis ausgeben ---
    detected_number = detect_number(final_boxes)

def process_video(video_path, primary):
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

            # crop frame
            rect = frame.to_rgb().to_ndarray(format="bgr24")[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w]

            # get status from rect
            status, reason, *masks = process_rect(rect, min_number_pixels)

            # update states
            frame_time = frame.time
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
                        sys.stdout.write(f"\r\033[Kreload: {S:10.6f}s → {E:10.6f}s | Δt: {D:.6f} ± {R:.6f} s | min/max Δt: {t2 - t1:.6f}/{t3 - t0:.6f} s\n")
                        sys.stdout.flush()
            else:
                states[-1].end = frame_time

            # update progress bar
            processing_speed = (frame_time - last_frame_time) / (time.perf_counter() - start_counter)
            last_frame_time = frame_time
            filled_length = min(round(BAR_WIDTH * frame_time / total_duration), BAR_WIDTH)
            bar = '█' * filled_length + '-' * (BAR_WIDTH - filled_length)
            sys.stdout.write(f"\r\033[K|{bar}| {frame_time:.2f}/{total_duration:.2f} s | {frame_time / total_duration:.2%} | {processing_speed:.2f} s/s | {status!r} ({reason})")
            sys.stdout.flush()

            # display mask
            combined = np.hstack((rect, *(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) for m in masks)))
            cv2.imshow("Video", combined)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            if status == "NONE":
                cv2.waitKey(0)

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