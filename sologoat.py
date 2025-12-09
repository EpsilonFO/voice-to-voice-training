import pyaudio
import numpy as np

import whisper
import gradium
from mistralai import Mistral

from config import *

model = whisper.load_model("small")
gradium_client = gradium.client.GradiumClient(api_key=GRADIUM_API_KEY)
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

conversation_history = []
audio = pyaudio.PyAudio()

def record_audio(self, silence_threshold: float = 300,
                    silence_duration: float = 1.2, min_recording_time: float = 1.5) -> bytes:
    """Enregistre avec detection de silence."""
    stream = self.audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=STT_SAMPLE_RATE,
        input=True,
        frames_per_buffer=STT_CHUNK_SIZE
    )

    print("\n[MIC] Parlez...")

    frames = []
    silent_chunks = 0
    speech_detected = False
    chunks_for_silence = int(silence_duration * STT_SAMPLE_RATE / STT_CHUNK_SIZE)
    min_chunks = int(min_recording_time * STT_SAMPLE_RATE / STT_CHUNK_SIZE)
    chunk_count = 0

    while True:
        data = stream.read(STT_CHUNK_SIZE, exception_on_overflow=False)
        frames.append(data)
        chunk_count += 1

        audio_array = np.frombuffer(data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))

        level = int(min(rms / 100, 20))
        bar = "#" * level + "-" * (20 - level)
        print(f"\r[{bar}]", end="", flush=True)

        if rms > silence_threshold:
            speech_detected = True
            silent_chunks = 0
        else:
            silent_chunks += 1

        if speech_detected and silent_chunks >= chunks_for_silence and chunk_count >= min_chunks:
            break

    stream.stop_stream()
    stream.close()
    print()

    return b''.join(frames)

user_speech = record_audio()
user_transcript = model.transcribe(user_speech)
chat_response = mistral_client.chat.complete(
    model= model,
    messages = [
        {
            "role": "user",
            "content": user_transcript,
        },
    ]
)