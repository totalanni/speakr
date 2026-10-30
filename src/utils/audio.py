"""Minimal audio decoding shim for ONNX speaker diarization."""
import os
import subprocess


def decode_audio(path, target_sr=16000):
    """Decode audio file to raw PCM at target sample rate using ffmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-i", path, "-ar", str(target_sr), "-ac", "1", "-f", "s16le", "-"],
        capture_output=True, check=True
    )
    return result.stdout
