
# # streamlit_app.py
# import os, io, time, json, uuid, requests, mimetypes
# import streamlit as st
# import boto3
# from botocore.client import Config

# # --- ENV ---
# RUNPOD_API_KEY   = os.getenv("RUNPOD_API_KEY") or "YOUR_RUNPOD_API_KEY"
# RUNPOD_ENDPOINT  = os.getenv("RUNPOD_ENDPOINT_ID") or "YOUR_ENDPOINT_ID"
# S3_BUCKET        = os.getenv("S3_BUCKET") or "your-bucket-name"
# S3_REGION        = os.getenv("S3_REGION") or "ap-south-1"
# # Optional for Cloudflare R2 (S3-compatible):
# S3_ENDPOINT_URL  = os.getenv("S3_ENDPOINT_URL")  # e.g. "https://<accountid>.r2.cloudflarestorage.com"

# RUN_URL    = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT}/run"
# STATUS_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT}/status/"

# HEADERS = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}

# # --- S3 client (works for AWS S3 or R2 if ENDPOINT_URL set) ---
# s3_client = boto3.client(
#     "s3",
#     region_name=S3_REGION,
#     endpoint_url=S3_ENDPOINT_URL,  # None for AWS S3, set for R2
#     config=Config(s3={"addressing_style": "virtual"})  # plays nice with R2
# )

# def upload_and_presign(file_bytes: bytes, filename: str, content_type: str, ttl_seconds: int = 3600) -> str:
#     key = f"uploads/{uuid.uuid4()}_{filename}"
#     s3_client.put_object(
#         Bucket=S3_BUCKET,
#         Key=key,
#         Body=file_bytes,
#         ContentType=content_type,
#         ACL="private"
#     )
#     # Presign GET URL (signed, temporary, private)
#     url = s3_client.generate_presigned_url(
#         "get_object",
#         Params={"Bucket": S3_BUCKET, "Key": key},
#         ExpiresIn=ttl_seconds
#     )
#     return url

# def runpod_submit(input_payload: dict, sync: bool = False):
#     url = RUN_URL + ("sync" if sync else "")
#     r = requests.post(url, headers=HEADERS, data=json.dumps({"input": input_payload}), timeout=60)
#     r.raise_for_status()
#     return r.json()

# def runpod_poll(job_id: str, delay=2):
#     while True:
#         r = requests.get(STATUS_URL + job_id, headers=HEADERS, timeout=30)
#         r.raise_for_status()
#         j = r.json()
#         if j.get("status") in ("COMPLETED", "FAILED", "CANCELLED"):
#             return j
#         time.sleep(delay)

# # --- UI ---
# st.set_page_config(page_title="Transcribe (RunPod + S3)", layout="centered")
# st.title("🎧 Whisper Transcription (RunPod + S3/R2/GCS)")

# file = st.file_uploader("Upload .mp3 or .wav", type=["mp3", "wav"])
# cols = st.columns(3)
# language   = cols[0].selectbox("Language", ["en"], index=0)
# vad_filter = cols[1].selectbox("VAD filter", [False, True], index=0)
# max_words  = cols[2].number_input("Max words/line (SRT)", min_value=3, max_value=12, value=7)
# mode = st.radio("Run mode", ["Async (recommended)", "Sync"], index=0)

# if st.button("Transcribe", type="primary"):
#     if not file:
#         st.error("Please upload an audio file.")
#         st.stop()

#     ext = file.name.split(".")[-1].lower()
#     content_type = mimetypes.guess_type(file.name)[0] or ("audio/mpeg" if ext == "mp3" else "audio/wav")
#     b = file.read()
#     if len(b) == 0:
#         st.error("Empty file.")
#         st.stop()

#     with st.spinner("Uploading to object storage…"):
#         presigned_url = upload_and_presign(b, file.name, content_type, ttl_seconds=3600)

#     input_payload = {
#         "file_url": presigned_url,     # <-- RunPod downloads from here
#         "language": language,
#         "vad_filter": vad_filter,
#         "generate_srt": True,
#         "generate_txt": True,
#         "return_files": "inline",
#         "max_words_per_line": max_words
#     }

#     try:
#         if mode.startswith("Sync"):
#             with st.spinner("Transcribing (sync)…"):
#                 resp = runpod_submit(input_payload, sync=True)
#                 out = resp.get("output", {})
#         else:
#             with st.spinner("Starting job (async)…"):
#                 job = runpod_submit(input_payload, sync=False)
#                 job_id = job.get("id")
#                 if not job_id:
#                     raise RuntimeError(f"No job id in response: {job}")
#             with st.spinner(f"Running on GPU… (job: {job_id})"):
#                 result = runpod_poll(job_id)
#             if result.get("status") != "COMPLETED":
#                 st.error(f"Job failed: {result}")
#                 st.stop()
#             out = result.get("output") or {}

#         st.success("Transcription complete ✅")
#         mc = st.columns(4)
#         mc[0].metric("Detected lang", str(out.get("language")))
#         mc[1].metric("Confidence", f"{out.get('language_probability', 0):.2f}")
#         mc[2].metric("Audio duration (s)", f"{out.get('duration', 0):.2f}")
#         mc[3].metric("GPU time (s)", f"{out.get('transcription_time', 0):.2f}")

#         st.subheader("Preview")
#         st.write(out.get("text_preview", "") or "(empty)")

#         if out.get("txt"):
#             st.subheader("Transcript (.txt)")
#             st.text_area("Text", out["txt"], height=240)
#             st.download_button("Download transcription.txt", out["txt"], file_name="transcription.txt")

#         if out.get("srt"):
#             st.subheader("Subtitles (.srt)")
#             st.code(out["srt"], language="text")
#             st.download_button("Download transcription.srt", out["srt"], file_name="transcription.srt", mime="application/x-subrip")

#     except requests.HTTPError as e:
#         st.error(f"HTTP error: {e.response.text if e.response is not None else e}")
#     except Exception as e:
#         st.error(str(e))




# streamlit_app.py
import os, io, time, json, uuid, requests, mimetypes
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()  # loads .env locally; safe on server too

# --- ENV ---
RUNPOD_API_KEY   = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT  = os.getenv("RUNPOD_ENDPOINT_ID")
RUN_URL          = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT}/run"
STATUS_URL       = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT}/status/"

HEADERS = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}

# --- Supabase client ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE")  # server-side only
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "transcripts")
sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_and_presign_supabase(file_bytes: bytes, filename: str, content_type: str, ttl_seconds: int = 3600) -> str:
    """Upload to private bucket and return a signed GET URL."""
    key = f"uploads/{uuid.uuid4()}_{filename}"
    # Upload
    resp = sb.storage.from_(SUPABASE_BUCKET).upload(
        path=key,
        file=file_bytes,
        file_options={"contentType": content_type, "upsert": False}
    )
    if isinstance(resp, dict) and resp.get("error"):
        raise RuntimeError(f"Supabase upload error: {resp['error']['message']}")
    # Sign
    signed = sb.storage.from_(SUPABASE_BUCKET).create_signed_url(path=key, expires_in=ttl_seconds)
    signed_url = signed.get("signedURL") or signed.get("signed_url")
    if not signed_url:
        raise RuntimeError(f"Could not create signed URL: {signed}")
    return signed_url

def runpod_submit(input_payload: dict, sync: bool = False):
    url = RUN_URL + ("sync" if sync else "")
    r = requests.post(url, headers=HEADERS, data=json.dumps({"input": input_payload}), timeout=60)
    r.raise_for_status()
    return r.json()

def runpod_poll(job_id: str, delay=2):
    while True:
        r = requests.get(STATUS_URL + job_id, headers=HEADERS, timeout=30)
        r.raise_for_status()
        j = r.json()
        if j.get("status") in ("COMPLETED", "FAILED", "CANCELLED"):
            return j
        time.sleep(delay)


# # --- UI ---
# st.set_page_config(page_title="Transcribe (RunPod + Supabase)", layout="centered")
# st.title("🎧 Whisper Transcription (RunPod + Supabase)")

# file = st.file_uploader("Upload .mp3 or .wav", type=["mp3", "wav"])
# cols = st.columns(3)
# language   = cols[0].selectbox("Language", ["en"], index=0)
# vad_filter = cols[1].selectbox("VAD filter", [False, True], index=0)
# max_words  = cols[2].number_input("Max words/line (SRT)", min_value=3, max_value=12, value=7)
# mode = st.radio("Run mode", ["Async (recommended)", "Sync"], index=0)

# if st.button("Transcribe", type="primary"):
#     if not file:
#         st.error("Please upload an audio file.")
#         st.stop()

#     ext = file.name.split(".")[-1].lower()
#     content_type = mimetypes.guess_type(file.name)[0] or ("audio/mpeg" if ext == "mp3" else "audio/wav")
#     data = file.read()
#     if not data:
#         st.error("Empty file.")
#         st.stop()

#     with st.spinner("Uploading to Supabase Storage…"):
#         presigned_url = upload_and_presign_supabase(data, file.name, content_type, ttl_seconds=3600)

#     input_payload = {
#         "file_url": presigned_url,
#         "language": language,
#         "vad_filter": vad_filter,
#         "generate_srt": True,
#         "generate_txt": True,
#         "return_files": "inline",
#         "max_words_per_line": max_words
#     }

#     try:
#         if mode.startswith("Sync"):
#             with st.spinner("Transcribing (sync)…"):
#                 resp = runpod_submit(input_payload, sync=True)
#                 out = resp.get("output", {})
#         else:
#             with st.spinner("Starting job (async)…"):
#                 job = runpod_submit(input_payload, sync=False)
#                 job_id = job.get("id")
#                 if not job_id:
#                     raise RuntimeError(f"No job id in response: {job}")
#             with st.spinner(f"Running on GPU… (job: {job_id})"):
#                 result = runpod_poll(job_id)
#             if result.get("status") != "COMPLETED":
#                 st.error(f"Job failed: {result}")
#                 st.stop()
#             out = result.get("output") or {}

#         st.success("Transcription complete ✅")
#         mc = st.columns(4)
#         mc[0].metric("Detected lang", str(out.get("language")))
#         mc[1].metric("Confidence", f"{out.get('language_probability', 0):.2f}")
#         mc[2].metric("Audio duration (s)", f"{out.get('duration', 0):.2f}")
#         mc[3].metric("GPU time (s)", f"{out.get('transcription_time', 0):.2f}")

#         st.subheader("Preview")
#         st.write(out.get("text_preview", "") or "(empty)")

#         if out.get("txt"):
#             st.subheader("Transcript (.txt)")
#             st.text_area("Text", out["txt"], height=240)
#             st.download_button("Download transcription.txt", out["txt"], file_name="transcription.txt")

#         if out.get("srt"):
#             st.subheader("Subtitles (.srt)")
#             st.code(out["srt"], language="text")
#             st.download_button("Download transcription.srt", out["srt"], file_name="transcription.srt", mime="application/x-subrip")

#     except requests.HTTPError as e:
#         st.error(f"HTTP error: {e.response.text if e.response is not None else e}")
#     except Exception as e:
#         st.error(str(e))






# --- keep your imports/env/setup as-is ---


# --- UI ---
st.set_page_config(page_title="Transcribe (RunPod + Supabase)", layout="centered")
st.title("🎧 Whisper Transcription (RunPod + Supabase)")

file = st.file_uploader("Upload .mp3 or .wav", type=["mp3", "wav"])

# REMOVE the language select; hardcode it:
LANGUAGE_FIXED = "en"

# Keep VAD filter control if you want it; or hardcode to False/True.
vad_filter = st.selectbox("VAD filter", [False, True], index=0)
max_words  = st.number_input("Max words/line (SRT)", min_value=3, max_value=12, value=7)

# REMOVE run mode radio entirely — we will always run async.

if st.button("Transcribe", type="primary"):
    if not file:
        st.error("Please upload an audio file.")
        st.stop()

    import mimetypes, uuid
    ext = file.name.split(".")[-1].lower()
    content_type = mimetypes.guess_type(file.name)[0] or ("audio/mpeg" if ext == "mp3" else "audio/wav")
    data = file.read()
    if not data:
        st.error("Empty file.")
        st.stop()

    with st.spinner("Uploading to Supabase Storage…"):
        presigned_url = upload_and_presign_supabase(data, file.name, content_type, ttl_seconds=3600)

    # Always async + fixed language
    input_payload = {
        "file_url": presigned_url,
        "language": LANGUAGE_FIXED,         # <- fixed language
        "vad_filter": vad_filter,
        "generate_srt": True,
        "generate_txt": True,
        "return_files": "inline",
        "max_words_per_line": max_words
    }

    try:
        # ALWAYS ASYNC:
        with st.spinner("Starting job (async)…"):
            job = runpod_submit(input_payload, sync=False)
            job_id = job.get("id")
            if not job_id:
                raise RuntimeError(f"No job id in response: {job}")
        with st.spinner(f"Running on GPU… (job: {job_id})"):
            result = runpod_poll(job_id)
        if result.get("status") != "COMPLETED":
            st.error(f"Job failed: {result}")
            st.stop()
        out = result.get("output") or {}

        st.success("Transcription complete ✅")
        mc = st.columns(4)
        mc[0].metric("Detected lang", str(out.get("language")))
        mc[1].metric("Confidence", f"{out.get('language_probability', 0):.2f}")
        mc[2].metric("Audio duration (s)", f"{out.get('duration', 0):.2f}")
        mc[3].metric("GPU time (s)", f"{out.get('transcription_time', 0):.2f}")

        st.subheader("Preview")
        st.write(out.get("text_preview", "") or "(empty)")

        if out.get("txt"):
            st.subheader("Transcript (.txt)")
            st.text_area("Text", out["txt"], height=240)
            st.download_button("Download transcription.txt", out["txt"], file_name="transcription.txt")

        if out.get("srt"):
            st.subheader("Subtitles (.srt)")
            st.code(out["srt"], language="text")
            st.download_button("Download transcription.srt", out["srt"], file_name="transcription.srt", mime="application/x-subrip")

    except requests.HTTPError as e:
        st.error(f"HTTP error: {e.response.text if e.response is not None else e}")
    except Exception as e:
        st.error(str(e))
