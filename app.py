"""
Entry point for the Traffic Violation Detection System.

Usage:
    python app.py --video data/samples/test_video.mp4
    python app.py --video data/samples/test_video.mp4 --camera cam_001
    python app.py --video 0                              # webcam
    python app.py --video rtsp://192.168.1.10/stream    # RTSP
    python app.py --video data/samples/test_video.mp4 --dry-run
    python app.py --dashboard                           # launch analytics dashboard
    python app.py --demo                                # launch interactive image demo
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Traffic Violation Detection System")
    parser.add_argument("--video", type=str, help="Video source: file path, RTSP URL, or webcam index")
    parser.add_argument("--camera", type=str, default="cam_001", help="Camera ID from cameras.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Run without violation detection (frame loop only)")
    parser.add_argument("--dashboard", action="store_true", help="Launch the Streamlit analytics dashboard")
    parser.add_argument("--demo", action="store_true", help="Launch the interactive image demo (cloud-friendly)")
    parser.add_argument("--show", action="store_true", help="Display annotated frames in a window during processing")
    args = parser.parse_args()

    if args.dashboard:
        dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)], check=True)
        return

    if args.demo:
        demo_path = Path(__file__).parent / "streamlit_app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(demo_path)], check=True)
        return

    if not args.video:
        parser.error("--video is required unless --dashboard is set")

    from pipelines.video_pipeline import run
    run(
        source=args.video,
        camera_id=args.camera,
        dry_run=args.dry_run,
        show=args.show,
    )


if __name__ == "__main__":
    main()
