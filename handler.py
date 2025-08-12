# import os
# import base64
# import tempfile
# import requests
# import runpod
# from typing import Dict, Any
# from transcription_system import ProfessionalTranscriber

# # Load model once per warm container
# MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
# COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")  # "int8_float16" for low VRAM
# LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")
# VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "false").lower() == "true"

# transcriber = None

# def load_model_once():
#     global transcriber
#     if transcriber is None:
#         transcriber = ProfessionalTranscriber(model_size=MODEL_SIZE, compute_type=COMPUTE_TYPE)

# def _save_from_b64(b64_str: str, suffix: str) -> str:
#     audio_bytes = base64.b64decode(b64_str)
#     fd, path = tempfile.mkstemp(suffix=suffix)
#     with os.fdopen(fd, "wb") as f:
#         f.write(audio_bytes)
#     return path

# def _save_from_url(url: str, suffix: str) -> str:
#     r = requests.get(url, timeout=60)
#     r.raise_for_status()
#     fd, path = tempfile.mkstemp(suffix=suffix)
#     with os.fdopen(fd, "wb") as f:
#         f.write(r.content)
#     return path

# def run(event: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Expected inputs (one of):
#       - {"file_url": "https://.../audio.mp3", "generate_srt": true, "generate_txt": true}
#       - {"file_b64": "<base64 of .mp3 or .wav>", "extension": "mp3"}  # extension is required for b64
#     Optional:
#       - "language": "en"
#       - "vad_filter": true|false
#       - "max_words_per_line": 7
#       - "return_files": "inline" | "none"
#     """
#     load_model_once()

#     file_url = event.get("file_url")
#     file_b64 = event.get("file_b64")
#     extension = event.get("extension", "mp3").lower()
#     if extension not in ["mp3", "wav"]:
#         return {"error": "Unsupported extension. Use 'mp3' or 'wav'."}

#     # Save input audio to a temp path
#     try:
#         if file_url:
#             audio_path = _save_from_url(file_url, f".{extension}")
#         elif file_b64:
#             audio_path = _save_from_b64(file_b64, f".{extension}")
#         else:
#             return {"error": "Provide either 'file_url' or 'file_b64'."}
#     except Exception as e:
#         return {"error": f"Failed to fetch audio: {e}"}

#     language = event.get("language", LANGUAGE)
#     vad_filter = bool(event.get("vad_filter", VAD_FILTER))
#     max_words_per_line = int(event.get("max_words_per_line", 7))
#     generate_srt = bool(event.get("generate_srt", True))
#     generate_txt = bool(event.get("generate_txt", True))
#     return_files = event.get("return_files", "inline")

#     # Run transcription
#     try:
#         results = transcriber.transcribe_audio(audio_path=audio_path, language=language, vad_filter=vad_filter)
#     except Exception as e:
#         return {"error": f"Transcription failed: {e}"}

#     out = {
#         "language": results.get("language"),
#         "language_probability": results.get("language_probability"),
#         "duration": results.get("duration"),
#         "transcription_time": results.get("transcription_time"),
#         "text_preview": results.get("full_text", "")[:300],
#         "segments_count": len(results.get("segments", [])),
#     }

#     # Optionally generate files and return inline
#     if generate_srt or generate_txt:
#         tmpdir = tempfile.mkdtemp()
#         base = os.path.join(tmpdir, "transcription")
#         if generate_srt:
#             srt_path = transcriber.generate_srt(results, base + ".srt", max_words_per_line=max_words_per_line)
#             if return_files == "inline":
#                 with open(srt_path, "r", encoding="utf-8") as f:
#                     out["srt"] = f.read()
#         if generate_txt:
#             txt_path = transcriber.generate_txt(results, base + ".txt")
#             if return_files == "inline":
#                 with open(txt_path, "r", encoding="utf-8") as f:
#                     out["txt"] = f.read()

#     return out

# # Start the RunPod handler
# runpod.serverless.start({"handler": run})




# handler.py
import os
import base64
import tempfile
import requests
import runpod
from typing import Dict, Any

# Import your transcription logic
from transcription_system import ProfessionalTranscriber

# -----------------------
# Config via environment
# -----------------------
MODEL_SIZE   = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")  # e.g. "int8_float16" for low VRAM
LANGUAGE     = os.getenv("WHISPER_LANGUAGE", "en")           # fixed default; can be overridden per-request
VAD_FILTER_D = os.getenv("WHISPER_VAD_FILTER", "false").lower() == "true"

# Warm, global model (reused across jobs in warm container)
_transcriber = None

def _load_model_once():
    global _transcriber
    if _transcriber is None:
        _transcriber = ProfessionalTranscriber(
            model_size=MODEL_SIZE,
            compute_type=COMPUTE_TYPE
        )

def _save_from_b64(b64_str: str, suffix: str) -> str:
    audio_bytes = base64.b64decode(b64_str)
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(audio_bytes)
    return path

def _save_from_url(url: str, suffix: str) -> str:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(r.content)
    return path

def run(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod sends {"input": {...}}. We unwrap that first.
    Accepted inputs (provide one of file_url or file_b64):

    {
      "file_url": "https://.../audio.mp3",
      "language": "en",
      "vad_filter": false,
      "max_words_per_line": 7,
      "generate_srt": true,
      "generate_txt": true,
      "return_files": "inline"  # "inline" or "none"
    }

    OR

    {
      "file_b64": "<base64>",
      "extension": "mp3"  # required when using file_b64
    }
    """
    payload = event.get("input") or event or {}

    file_url = payload.get("file_url")
    file_b64 = payload.get("file_b64")
    extension = (payload.get("extension") or "mp3").lower()
    if extension not in ("mp3", "wav"):
        return {"error": "Unsupported extension. Use 'mp3' or 'wav'."}

    language = payload.get("language", LANGUAGE)
    vad_filter = bool(payload.get("vad_filter", VAD_FILTER_D))
    max_words_per_line = int(payload.get("max_words_per_line", 7))
    generate_srt = bool(payload.get("generate_srt", True))
    generate_txt = bool(payload.get("generate_txt", True))
    return_files = payload.get("return_files", "inline")  # "inline" or "none"

    if not file_url and not file_b64:
        return {"error": "Provide either 'file_url' or 'file_b64'."}

    # Ensure model is loaded (warm containers reuse it)
    _load_model_once()

    # Persist input to a temp file
    try:
        if file_url:
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

    out: Dict[str, Any] = {
        "language": results.get("language"),
        "language_probability": results.get("language_probability"),
        "duration": results.get("duration"),
        "transcription_time": results.get("transcription_time"),
        "text_preview": (results.get("full_text") or "")[:300],
        "segments_count": len(results.get("segments") or []),
    }

    # Optionally generate SRT/TXT and return inline
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

# -----------------------
# Direct start (Option B)
# -----------------------
if __name__ == "__main__":
    print(">>> RunPod serverless worker starting (direct start mode)")
    runpod.serverless.start({"handler": run})