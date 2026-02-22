import subprocess
from pathlib import Path

ROOT_DIR = Path("D:/clips/reloads")  # 👈 change this
CROP_W = 460
CROP_H = 260

def crop_video(input_path: Path):
    # Mirror directory structure under ROOT_DIR/cropped
    relative_path = input_path.relative_to(ROOT_DIR)
    output_path = ROOT_DIR / "cropped" / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",                                    # overwrite output
        "-hwaccel", "d3d11va",                   # AMD/Windows hardware acceleration
        "-i", str(input_path),
        "-vf", "crop=384:192:2108:1185",         # crop
        "-c:v", "av1_amf",                       # AMD AV1 encoder
        "-quality", "balanced",                  # balanced preset
        "-rc", "cqp",                            # constant quality mode
        "-qp_i", "30",                           # I-frame quality
        "-qp_p", "30",                           # P-frame quality
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",                          # copy audio stream
        str(output_path),
    ]

    print(f"Cropping: {input_path}")
    subprocess.run(cmd, check=True)

def main():
    for mp4 in ROOT_DIR.rglob("*.mp4"):
        # Skip already-cropped videos
        if mp4.parent.name == "cropped":
            continue

        try:
            crop_video(mp4)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed on {mp4}: {e}")

if __name__ == "__main__":
    main()