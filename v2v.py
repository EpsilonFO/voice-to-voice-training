"""
Voice-to-Voice Interface avec Gradium API - Version Streaming
Latence minimale grace au streaming bout-en-bout
"""

import asyncio
import gradium
import pyaudio
import numpy as np
import anthropic
from config import *


class VoiceToVoiceStreaming:
    def __init__(self):
        self.gradium_client = gradium.client.GradiumClient(api_key=GRADIUM_API_KEY)
        self.anthropic_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        self.audio = pyaudio.PyAudio()
        self.conversation_history = []

    async def transcribe_audio(self, audio_data: bytes) -> str:
        """STT avec Gradium."""
        async def audio_generator():
            for i in range(0, len(audio_data), STT_CHUNK_SIZE * 2):
                yield audio_data[i:i + STT_CHUNK_SIZE * 2]

        stream = await gradium.speech.stt_stream(
            self.gradium_client,
            setup={"model_name": "default", "input_format": "pcm"},
            audio=audio_generator()
        )

        text_parts = []
        async for message in stream.iter_text():
            if hasattr(message, 'text'):
                text_parts.append(message.text)

        return " ".join(text_parts)

    async def stream_llm_response(self, user_text: str):
        """Stream la reponse du LLM token par token."""
        self.conversation_history.append({"role": "user", "content": user_text})

        async with self.anthropic_client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            system="Tu es un assistant vocal amical. Reponds en francais, de maniere concise (2-3 phrases max). Ne mets pas de formatting markdown.",
            messages=self.conversation_history
        ) as stream:
            full_response = ""
            async for text in stream.text_stream:
                full_response += text
                yield text

            self.conversation_history.append({"role": "assistant", "content": full_response})

    async def stream_tts_and_play(self, text_stream):
        """
        Recoit un stream de texte du LLM, accumule par phrases,
        et joue l'audio en streaming des qu'une phrase est complete.
        """
        output_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=TTS_SAMPLE_RATE,
            output=True,
            frames_per_buffer=TTS_CHUNK_SIZE
        )

        buffer = ""
        sentence_delimiters = {'.', '!', '?', ':', ';'}

        try:
            async for text_chunk in text_stream:
                buffer += text_chunk
                print(text_chunk, end="", flush=True)

                # Verifier si on a une phrase complete
                last_delimiter_pos = -1
                for i, char in enumerate(buffer):
                    if char in sentence_delimiters:
                        last_delimiter_pos = i

                if last_delimiter_pos >= 0:
                    # Extraire la phrase complete
                    sentence = buffer[:last_delimiter_pos + 1].strip()
                    buffer = buffer[last_delimiter_pos + 1:]

                    if sentence:
                        # Generer et jouer l'audio pour cette phrase
                        tts_stream = await gradium.speech.tts_stream(
                            self.gradium_client,
                            setup={"voice_id": VOICE_ID, "output_format": "pcm"},
                            text=sentence
                        )
                        async for audio_chunk in tts_stream.iter_bytes():
                            output_stream.write(audio_chunk)

            # Jouer le reste du buffer s'il y en a
            if buffer.strip():
                print()
                tts_stream = await gradium.speech.tts_stream(
                    self.gradium_client,
                    setup={"voice_id": VOICE_ID, "output_format": "pcm"},
                    text=buffer.strip()
                )
                async for audio_chunk in tts_stream.iter_bytes():
                    output_stream.write(audio_chunk)

            print()

        finally:
            output_stream.stop_stream()
            output_stream.close()

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

    async def conversation_turn(self):
        """Un tour de conversation complet avec streaming."""
        # 1. Enregistrer
        audio_data = self.record_audio()

        if len(audio_data) < STT_CHUNK_SIZE * 4:
            print("[!] Trop court")
            return

        # 2. Transcrire
        user_text = await self.transcribe_audio(audio_data)
        print(f"[Vous] {user_text}")

        if not user_text.strip():
            return

        # 3. LLM + TTS en streaming
        print("[Agent] ", end="")
        text_stream = self.stream_llm_response(user_text)
        await self.stream_tts_and_play(text_stream)

    async def run(self):
        """Boucle principale."""
        print("=" * 50)
        print("  Voice-to-Voice Streaming")
        print("  Gradium + Claude")
        print("=" * 50)
        print("Entree = parler, 'q' = quitter\n")

        while True:
            cmd = input("[Entree/q]: ").strip().lower()
            if cmd == 'q':
                break

            try:
                await self.conversation_turn()
            except Exception as e:
                print(f"[Erreur] {e}")
                import traceback
                traceback.print_exc()

    def cleanup(self):
        self.audio.terminate()


async def main():
    v2v = VoiceToVoiceStreaming()
    try:
        await v2v.run()
    finally:
        v2v.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
