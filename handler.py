# handler.py
import os
import io
import time
import base64
import shutil
import tempfile
from typing import Dict, Any, Optional

import requests
import runpod
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# Import your transcription logic
from transcription_system import ProfessionalTranscriber

# =========================
# Environment Configuration
# =========================
MODEL_SIZE      = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
COMPUTE_TYPE    = os.getenv("WHISPER_COMPUTE_TYPE", "float16")  # e.g. "int8_float16"
LANGUAGE_DFLT   = os.getenv("WHISPER_LANGUAGE", "en")
VAD_FILTER_DFLT = os.getenv("WHISPER_VAD_FILTER", "false").lower() == "true"

# RunPod S3 (Network Volume) — optional; if not provided, bucket+key mode is unavailable
RUNPOD_S3_ACCESS_KEY = os.getenv("RUNPOD_S3_ACCESS_KEY", "")
RUNPOD_S3_SECRET_KEY = os.getenv("RUNPOD_S3_SECRET_KEY", "")
RUNPOD_S3_ENDPOINT   = os.getenv("RUNPOD_S3_ENDPOINT", "https://s3api-eu-ro-1.runpod.io/")
RUNPOD_S3_REGION     = os.getenv("RUNPOD_S3_REGION", "eu-ro-1")  # lowercase for signing

# Mounted Network Volume (optional)
# Attach your volume to the endpoint; it will appear at /runpod-volume
MOUNT_ROOT = os.getenv("RUNPOD_MOUNT_ROOT", "/runpod-volume")

# Globals reused across warm jobs
_transcriber = None
_s3_client = None


# ========================
# Lazy loaders / utilities
# ========================
def _load_model_once():
    """Load the transcription model once per warm container."""
    global _transcriber
    if _transcriber is None:
        _transcriber = ProfessionalTranscriber(
            model_size=MODEL_SIZE,
            compute_type=COMPUTE_TYPE
        )


def _get_s3() -> Optional[boto3.client]:
    """Create (once) and return an S3 client for RunPod's S3-compatible API."""
    global _s3_client
    if _s3_client is not None:
        return _s3_client
    if not (RUNPOD_S3_ACCESS_KEY and RUNPOD_S3_SECRET_KEY and RUNPOD_S3_ENDPOINT):
        return None  # credentials not configured
    _s3_client = boto3.client(
        "s3",
        aws_access_key_id=RUNPOD_S3_ACCESS_KEY,
        aws_secret_access_key=RUNPOD_S3_SECRET_KEY,
        region_name=RUNPOD_S3_REGION,  # keep lowercase
        endpoint_url=RUNPOD_S3_ENDPOINT,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},  # path-style => https://endpoint/<bucket>/<key>
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )
    return _s3_client


def _save_temp_with_suffix(suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


def _save_from_b64(b64_str: str, suffix: str) -> str:
    data = base64.b64decode(b64_str)
    out = _save_temp_with_suffix(suffix)
    with open(out, "wb") as f:
        f.write(data)
    return out


def _save_from_url(url: str, suffix: str, timeout_sec: int = 180) -> str:
    """Robust downloader with streaming + retries. Works with presigned URLs (if valid)."""
    attempts = 3
    backoff = 2
    last_err = None
    for i in range(attempts):
        try:
            with requests.get(url, timeout=timeout_sec, stream=True, allow_redirects=True) as r:
                r.raise_for_status()
                out = _save_temp_with_suffix(suffix)
                with open(out, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1 MB
                        if chunk:
                            f.write(chunk)
            return out
        except Exception as e:
            last_err = e
            if i < attempts - 1:
                time.sleep(backoff)
                backoff *= 2
    raise RuntimeError(f"HTTP download failed: {last_err}")


def _save_from_bucket(bucket: str, key: str, suffix: str) -> str:
    """Directly pull from RunPod S3 via boto3 (recommended, avoids presigned URL issues)."""
    s3 = _get_s3()
    if s3 is None:
        raise RuntimeError("S3 credentials not configured in environment (RUNPOD_S3_*).")
    out = _save_temp_with_suffix(suffix)
    try:
        s3.download_file(bucket, key, out)
    except ClientError as e:
        # Clean up the empty temp file on failure
        try:
            os.remove(out)
        except Exception:
            pass
        raise RuntimeError(f"S3 download failed: {e}")
    return out


def _save_from_volume(volume_path: str) -> str:
    """
    Copy a file from the mounted Network Volume into /tmp.
    Expects a path like /runpod-volume/uploads/file.mp3
    """
    if not os.path.isabs(volume_path):
        raise ValueError("volume_path must be an absolute path (e.g., /runpod-volume/...).")
    if not volume_path.startswith(MOUNT_ROOT):
        raise ValueError(f"volume_path must live under {MOUNT_ROOT}")
    if not os.path.exists(volume_path):
        raise FileNotFoundError(f"volume_path not found: {volume_path}")
    suffix = os.path.splitext(volume_path)[1] or ""
    out = _save_temp_with_suffix(suffix)
    with open(volume_path, "rb") as src, open(out, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    return out


# =========
# Main run
# =========
def run(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod sends {"input": {...}}. Supported inputs (provide exactly one of the following sources):

    A) Direct S3 (recommended):
       {
         "bucket": "ul8t514xdg",
         "key": "uploads/file.mp3",
         "extension": "mp3"
       }

    B) Mounted volume:
       {
         "volume_path": "/runpod-volume/uploads/file.mp3"
       }

    C) URL (presigned or public):
       {
         "file_url": "https://.../file.mp3",
         "extension": "mp3"
       }

    D) Base64:
       {
         "file_b64": "<base64>",
         "extension": "mp3"
       }

    Optional common fields:
       "language": "en",
       "vad_filter": false,
       "max_words_per_line": 7,
       "generate_srt": true,
       "generate_txt": true,
       "return_files": "inline"   # or "none"
    """
    payload = event.get("input") or event or {}

    # Source selectors
    bucket     = payload.get("bucket")
    key        = payload.get("key")
    volume_path= payload.get("volume_path")
    file_url   = payload.get("file_url")
    file_b64   = payload.get("file_b64")

    # Basic options
    extension  = (payload.get("extension") or "mp3").lower()
    if extension not in ("mp3", "wav"):
        return {"error": "Unsupported extension. Use 'mp3' or 'wav'."}

    language   = payload.get("language", LANGUAGE_DFLT)
    vad_filter = bool(payload.get("vad_filter", VAD_FILTER_DFLT))
    max_words_per_line = int(payload.get("max_words_per_line", 7))
    generate_srt = bool(payload.get("generate_srt", True))
    generate_txt = bool(payload.get("generate_txt", True))
    return_files = payload.get("return_files", "inline")  # "inline" | "none"

    # Make sure we got exactly one source
    source_count = sum(bool(x) for x in [bucket and key, volume_path, file_url, file_b64])
    if source_count != 1:
        return {"error": "Provide exactly ONE of: (bucket+key) OR volume_path OR file_url OR file_b64."}

    # Ensure model is loaded
    _load_model_once()

    # Materialize the audio into a local temp file
    try:
        if bucket and key:
            audio_path = _save_from_bucket(bucket, key, f".{extension}")
        elif volume_path:
            audio_path = _save_from_volume(volume_path)
        elif file_url:
            audio_path = _save_from_url(file_url, f".{extension}")
        else:
            audio_path = _save_from_b64(file_b64, f".{extension}")
    except Exception as e:
        return {"error": f"Failed to fetch audio: {e}"}

    # Transcribe
    try:
        results = _transcriber.transcribe_audio(
            audio_path=audio_path,
            language=language,
            vad_filter=vad_filter
        )
    except Exception as e:
        return {"error": f"Transcription failed: {e}"}

    # Base response
    out: Dict[str, Any] = {
        "language": results.get("language"),
        "language_probability": results.get("language_probability"),
        "duration": results.get("duration"),
        "transcription_time": results.get("transcription_time"),
        "text_preview": (results.get("full_text") or "")[:300],
        "segments_count": len(results.get("segments") or []),
        "source": (
            "bucket+key" if (bucket and key) else
            "volume_path" if volume_path else
            "file_url" if file_url else
            "file_b64"
        ),
    }

    # Optional SRT/TXT generation
    if generate_srt or generate_txt:
        tmpdir = tempfile.mkdtemp()
        base = os.path.join(tmpdir, "transcription")
        if generate_srt:
            try:
                srt_path = _transcriber.generate_srt(results, base + ".srt", max_words_per_line=max_words_per_line)
                if return_files == "inline":
                    with open(srt_path, "r", encoding="utf-8") as f:
                        out["srt"] = f.read()
            except Exception as e:
                out["srt_error"] = f"SRT generation failed: {e}"
        if generate_txt:
            try:
                txt_path = _transcriber.generate_txt(results, base + ".txt")
                if return_files == "inline":
                    with open(txt_path, "r", encoding="utf-8") as f:
                        out["txt"] = f.read()
            except Exception as e:
                out["txt_error"] = f"TXT generation failed: {e}"

    return out


# ================
# RunPod bootstrap
# ================
if __name__ == "__main__":
    print(">>> RunPod serverless worker starting (direct start mode)")
    runpod.serverless.start({"handler": run})
