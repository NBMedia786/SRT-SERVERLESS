import collections
import webrtcvad
from scipy.io import wavfile
from scipy import signal
import soundfile as sf
import librosa
from faster_whisper import WhisperModel
import numpy as np
import torch
import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioPreprocessor:
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
        self.vad = webrtcvad.Vad(2)

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            audio = audio / rms * 0.1
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9
        return audio.astype(np.float32)

    def reduce_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        D = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(D)
        phase = np.angle(D)
        noise_frames = int(0.5 * sr / 512)
        noise_magnitude = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        alpha = 2.0
        magnitude_clean = magnitude - alpha * noise_magnitude
        magnitude_clean = np.maximum(magnitude_clean, 0.1 * magnitude)
        D_clean = magnitude_clean * np.exp(1j * phase)
        audio_clean = librosa.istft(D_clean, hop_length=512)
        return audio_clean

    def apply_vad(self, audio: np.ndarray, sr: int):
        audio_16bit = (audio * 32767).astype(np.int16)
        frame_duration = 30
        frame_length = int(sr * frame_duration / 1000)
        voiced_frames = []
        speech_segments = []
        for i in range(0, len(audio_16bit) - frame_length, frame_length):
            frame = audio_16bit[i:i + frame_length]
            if sr != 16000:
                frame_resampled = librosa.resample(frame.astype(np.float32), orig_sr=sr, target_sr=16000).astype(np.int16)
            else:
                frame_resampled = frame
            is_speech = self.vad.is_speech(frame_resampled.tobytes(), 16000)
            voiced_frames.append(is_speech)
            if is_speech:
                start_time = i / sr
                end_time = (i + frame_length) / sr
                speech_segments.append((start_time, end_time))
        if speech_segments:
            merged_segments = [speech_segments[0]]
            for start, end in speech_segments[1:]:
                if start - merged_segments[-1][1] < 0.5:
                    merged_segments[-1] = (merged_segments[-1][0], end)
                else:
                    merged_segments.append((start, end))
            speech_segments = merged_segments
        return audio, speech_segments

    def preprocess_audio(self, audio_path: str):
        logger.info(f"Loading audio: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=None)
        logger.info(f"Original: {len(audio)/sr:.2f}s, {sr}Hz")
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr
        audio = self.normalize_audio(audio)
        audio = self.reduce_noise(audio, sr)
        audio, speech_segments = self.apply_vad(audio, sr)
        logger.info(f"Preprocessing complete. Found {len(speech_segments)} speech segments")
        return audio.astype(np.float32), sr, speech_segments

class ProfessionalTranscriber:
    def __init__(self, model_size: str = "large-v3", device: str = "cuda", compute_type: str = "float16"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.preprocessor = AudioPreprocessor()
        logger.info(f"Loading Faster-Whisper model: {model_size}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"selected device:{device}")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type, cpu_threads=8, num_workers=4)
        logger.info("Model loaded successfully")

    def transcribe_audio(self, audio_path: str, language: str = "en", vad_filter: bool = True, vad_parameters: dict = None):
        if vad_parameters is None:
            vad_parameters = {
                "threshold": 0.6,
                "min_speech_duration_ms": 500,
                "max_speech_duration_s": 20,
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 200
            }
        audio, sr, speech_segments = self.preprocessor.preprocess_audio(audio_path)
        logger.info("Starting transcription...")
        start_time = time.time()
        segments, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
            word_timestamps=True,
            initial_prompt="This is a professional transcription. Please be accurate with technical terms, proper nouns, and punctuation."
        )
        transcription_segments = []
        full_text = []
        for segment in segments:
            segment_dict = {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "confidence": getattr(segment, 'avg_logprob', 0.0),
                "words": []
            }
            if hasattr(segment, 'words') and segment.words:
                for word in segment.words:
                    segment_dict["words"].append({
                        "start": word.start,
                        "end": word.end,
                        "word": word.word.strip(),
                        "confidence": word.probability
                    })
            transcription_segments.append(segment_dict)
            full_text.append(segment.text.strip())
        end_time = time.time()
        results = {
            "segments": transcription_segments,
            "full_text": " ".join(full_text),
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "transcription_time": end_time - start_time,
            "speech_segments": speech_segments
        }
        logger.info(f"Transcription completed in {end_time - start_time:.2f}s")
        logger.info(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")
        return results

    def generate_srt(self, results: Dict[str, Any], output_path: str = None, max_words_per_line: int = 7) -> str:
        import re
        if output_path is None:
            output_path = "transcription.srt"
        srt_content = []
        counter = 1
        for segment in results["segments"]:
            words = segment.get("words", [])
            if words and len(words) > 1:
                line = []
                for idx, word in enumerate(words):
                    w = word.copy()
                    w["word"] = w["word"].strip()
                    line.append(w)
                    is_last_word = idx == len(words) - 1
                    is_sentence_end = re.search(r'[.?!]$', w["word"]) is not None
                    is_max_words = len(line) >= max_words_per_line
                    if is_sentence_end or is_max_words or is_last_word:
                        start_time = line[0]["start"]
                        end_time = line[-1]["end"]
                        text = " ".join(w["word"] for w in line)
                        text = re.sub(r'\s+', ' ', text).strip()
                        srt_content.append(f"{counter}")
                        srt_content.append(f"{self.seconds_to_srt_time(start_time)} --> {self.seconds_to_srt_time(end_time)}")
                        srt_content.append(text)
                        srt_content.append("")
                        counter += 1
                        line = []
            else:
                text_words = [w.strip() for w in segment["text"].split()]
                start_time = segment["start"]
                end_time = segment["end"]
                duration = end_time - start_time
                total_words = len(text_words)
                for i in range(0, total_words, max_words_per_line):
                    line_words = text_words[i:i + max_words_per_line]
                    line_start = start_time + (i / total_words) * duration
                    line_end = start_time + (min(i + max_words_per_line, total_words) / total_words) * duration
                    text = " ".join(line_words).strip()
                    srt_content.append(f"{counter}")
                    srt_content.append(f"{self.seconds_to_srt_time(line_start)} --> {self.seconds_to_srt_time(line_end)}")
                    srt_content.append(text)
                    srt_content.append("")
                    counter += 1
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(srt_content))
        logger.info(f"SRT file saved: {output_path}")
        return output_path

    def seconds_to_srt_time(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        seconds = int(seconds)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def generate_txt(self, results: Dict[str, Any], output_path: str = None) -> str:
        if output_path is None:
            output_path = "transcription.txt"
        text = results.get("full_text", "").strip()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"TXT file saved: {output_path}")
        return output_path
