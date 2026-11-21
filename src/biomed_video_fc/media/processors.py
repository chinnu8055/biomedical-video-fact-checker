"""FFmpeg audio extraction and OpenCV time-based frame sampling."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from biomed_video_fc.exceptions import MediaProcessingError, ModelUnavailableError

logger = logging.getLogger(__name__)


class FFmpegAudioExtractor:
    def __init__(self, binary: str = "ffmpeg", sample_rate: int = 16000) -> None:
        self.binary = binary
        self.sample_rate = sample_rate

    def _resolve_binary(self) -> str:
        installed_binary = shutil.which(self.binary)
        if installed_binary is not None:
            return installed_binary
        if self.binary == "ffmpeg":
            try:
                import imageio_ffmpeg
            except ImportError:
                imageio_ffmpeg = None
            if imageio_ffmpeg is not None:
                bundled_binary = imageio_ffmpeg.get_ffmpeg_exe()
                if Path(bundled_binary).is_file():
                    logger.info("Using bundled FFmpeg binary at %s", bundled_binary)
                    return bundled_binary
        raise ModelUnavailableError(
            f"FFmpeg binary '{self.binary}' was not found. Install FFmpeg, install "
            "imageio-ffmpeg, or set media.ffmpeg_binary in config/default.toml."
        )

    def extract(self, video_path: Path, output_path: Path) -> Path:
        if not video_path.is_file():
            raise MediaProcessingError(f"Video does not exist: {video_path}")
        binary = self._resolve_binary()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            "1",
            str(output_path),
        ]
        logger.info("Extracting mono audio from %s", video_path.name)
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            message = exc.stderr.strip() or "unknown FFmpeg failure"
            raise MediaProcessingError(f"Audio extraction failed: {message}") from exc
        return output_path


class OpenCVFrameExtractor:
    def __init__(self, interval_seconds: float = 1.0, max_video_seconds: int = 1800) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.interval_seconds = interval_seconds
        self.max_video_seconds = max_video_seconds

    def extract(self, video_path: Path, output_dir: Path) -> list[tuple[Path, float]]:
        try:
            import cv2
        except ImportError as exc:
            raise ModelUnavailableError("Install opencv-python-headless to extract frames") from exc

        output_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise MediaProcessingError(f"OpenCV could not open {video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            capture.release()
            raise MediaProcessingError("Video reports an invalid frame rate")
        duration = frame_count / fps if frame_count > 0 else 0
        if duration > self.max_video_seconds:
            capture.release()
            raise MediaProcessingError(
                f"Video is {duration:.0f}s; configured maximum is {self.max_video_seconds}s"
            )

        frames: list[tuple[Path, float]] = []
        timestamp = 0.0
        try:
            while timestamp <= duration + 1e-6:
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ok, frame = capture.read()
                if not ok:
                    break
                frame_path = output_dir / f"frame_{timestamp:010.3f}.jpg"
                if not cv2.imwrite(str(frame_path), frame):
                    raise MediaProcessingError(f"Could not save extracted frame {frame_path.name}")
                frames.append((frame_path, timestamp))
                timestamp += self.interval_seconds
        finally:
            capture.release()
        logger.info("Extracted %d frames at %.2fs intervals", len(frames), self.interval_seconds)
        return frames
