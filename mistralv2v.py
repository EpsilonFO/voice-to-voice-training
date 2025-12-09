#!/usr/bin/env python3
"""
Voice-to-Voice Interface avec Gradium API et Mistral LLM
Version Streaming optimisée pour latence minimale
"""

import asyncio
import gradium
import pyaudio
import numpy as np
import os
import httpx
import json
from typing import AsyncIterator, Optional
import websockets
import base64
import time
from config import *

class GradiumSTTStream:
    """Gestion du streaming STT avec Gradium"""
    
    def __init__(self, api_key: str, server_url: str = "wss://eu.api.gradium.ai/api/speech/asr"):
        self.api_key = api_key
        self.server_url = server_url
        self.websocket = None
        self.request_id = None
        
    async def connect(self):
        """Établir la connexion WebSocket"""
        headers = {
            "x-api-key": self.api_key
        }
        self.websocket = await websockets.connect(self.server_url, extra_headers=headers)
        
        # Envoyer le message de setup
        setup_msg = {
            "type": "setup",
            "model_name": "default",
            "input_format": "pcm"
        }
        await self.websocket.send(json.dumps(setup_msg))
        
        # Attendre le message ready
        response = await self.websocket.recv()
        ready_data = json.loads(response)
        if ready_data["type"] == "ready":
            self.request_id = ready_data["request_id"]
            return True
        else:
            raise Exception(f"STT setup failed: {ready_data}")
    
    async def send_audio(self, audio_data: bytes):
        """Envoyer des données audio au serveur STT"""
        if not self.websocket:
            await self.connect()
            
        # Encoder en base64
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        audio_msg = {
            "type": "audio",
            "audio": audio_b64
        }
        await self.websocket.send(json.dumps(audio_msg))
    
    async def receive_text(self) -> AsyncIterator[str]:
        """Recevoir le texte transcrit en streaming"""
        if not self.websocket:
            await self.connect()
            
        try:
            while True:
                response = await self.websocket.recv()
                print(f"[STT] Message reçu: {response}")  # Debug
                data = json.loads(response)
                
                if data["type"] == "text":
                    yield data["text"]
                elif data["type"] == "end_text":
                    print("[STT] Fin de texte reçue")
                    break
                elif data["type"] == "step":
                    # Message VAD, on peut l'ignorer pour le moment
                    continue
                elif data["type"] == "error":
                    raise Exception(f"STT error: {data.get('message', 'Unknown error')}")
                else:
                    print(f"[STT] Message inconnu: {data}")
        except websockets.exceptions.ConnectionClosed as e:
            print(f"STT connection closed: {e}")
        except json.JSONDecodeError as e:
            print(f"STT JSON error: {e}")
    
    async def close(self):
        """Fermer la connexion"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

class GradiumTTSStream:
    """Gestion du streaming TTS avec Gradium"""
    
    def __init__(self, api_key: str, server_url: str = "wss://eu.api.gradium.ai/api/speech/tts"):
        self.api_key = api_key
        self.server_url = server_url
        self.websocket = None
        self.request_id = None
        
    async def connect(self):
        """Établir la connexion WebSocket"""
        headers = {
            "x-api-key": self.api_key
        }
        self.websocket = await websockets.connect(self.server_url, extra_headers=headers)
        
        # Envoyer le message de setup
        setup_msg = {
            "type": "setup",
            "model_name": "default",
            "voice_id": VOICE_ID,
            "output_format": "pcm"
        }
        await self.websocket.send(json.dumps(setup_msg))
        
        # Attendre le message ready
        response = await self.websocket.recv()
        ready_data = json.loads(response)
        if ready_data["type"] == "ready":
            self.request_id = ready_data["request_id"]
            return True
        else:
            raise Exception(f"TTS setup failed: {ready_data}")
    
    async def send_text(self, text: str):
        """Envoyer du texte à convertir en parole"""
        if not self.websocket:
            await self.connect()
            
        text_msg = {
            "type": "text",
            "text": text
        }
        await self.websocket.send(json.dumps(text_msg))
    
    async def receive_audio(self) -> AsyncIterator[bytes]:
        """Recevoir l'audio généré en streaming"""
        if not self.websocket:
            await self.connect()
            
        try:
            while True:
                response = await self.websocket.recv()
                print(f"[TTS] Message reçu: {response[:100]}...")  # Debug
                data = json.loads(response)
                
                if data["type"] == "audio":
                    audio_data = base64.b64decode(data["audio"])
                    yield audio_data
                elif data["type"] == "end_of_stream":
                    print("[TTS] Fin de stream reçue")
                    break
                elif data["type"] == "error":
                    raise Exception(f"TTS error: {data.get('message', 'Unknown error')}")
                else:
                    print(f"[TTS] Message inconnu: {data}")
        except websockets.exceptions.ConnectionClosed as e:
            print(f"TTS connection closed: {e}")
        except json.JSONDecodeError as e:
            print(f"TTS JSON error: {e}")
    
    async def close(self):
        """Fermer la connexion"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

class MistralLLMStream:
    """Gestion du streaming avec Mistral LLM"""
    
    def __init__(self, api_key: str, model: str = MISTRAL_MODEL):
        self.api_key = api_key
        self.model = model
        self.conversation_history = []
        
    async def stream_response(self, user_text: str) -> AsyncIterator[str]:
        """Stream la réponse du LLM token par token"""
        self.conversation_history.append({"role": "user", "content": user_text})
        
        # Configuration pour Mistral API
        
        
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": self.conversation_history,
            "stream": True,
            "max_tokens": 300,
            "temperature": 0.7
        }
        
        print(f"[LLM] Requête envoyée à {url}")
        print(f"[LLM] Messages: {self.conversation_history}")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=data, stream=True)
                
                if response.status_code != 200:
                    error_msg = f"Erreur HTTP {response.status_code}: {response.text}"
                    print(f"[LLM] {error_msg}")
                    yield error_msg
                    return
                
                full_response = ""
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:].strip())
                            print(f"[LLM] Réponse partielle: {data}")  # Debug
                            if data.get("choices") and data["choices"][0].get("delta", {}).get("content"):
                                content = data["choices"][0]["delta"]["content"]
                                full_response += content
                                yield content
                        except json.JSONDecodeError as e:
                            print(f"[LLM] Erreur JSON: {e}")
                            continue
                
                print(f"[LLM] Réponse complète: {full_response}")
                # Ajouter la réponse complète à l'historique
                self.conversation_history.append({"role": "assistant", "content": full_response})
                
        except Exception as e:
            error_msg = f"Erreur LLM: {e}"
            print(f"[LLM] {error_msg}")
            yield error_msg

class VoiceToVoiceStreaming:
    def __init__(self):
        self.gradium_stt = GradiumSTTStream(api_key=GRADIUM_API_KEY)
        self.gradium_tts = GradiumTTSStream(api_key=GRADIUM_API_KEY)
        self.mistral_llm = MistralLLMStream(api_key=MISTRAL_API_KEY)
        self.audio = pyaudio.PyAudio()
        self.conversation_history = []
        
    async def transcribe_audio(self, audio_data: bytes) -> str:
        """STT avec Gradium en streaming"""
        try:
            print("[STT] Connexion au serveur STT...")
            await self.gradium_stt.connect()
            print("[STT] Connexion établie")
            
            # Envoyer l'audio par chunks
            print(f"[STT] Envoi de {len(audio_data)} bytes d'audio...")
            for i in range(0, len(audio_data), STT_CHUNK_SIZE * 2):
                chunk = audio_data[i:i + STT_CHUNK_SIZE * 2]
                await self.gradium_stt.send_audio(chunk)
            
            print("[STT] Envoi terminé, attente de la transcription...")
            # Envoyer end_of_stream
            await self.gradium_stt.websocket.send(json.dumps({"type": "end_of_stream"}))
            
            # Recevoir le texte transcrit
            text_parts = []
            async for text_chunk in self.gradium_stt.receive_text():
                if text_chunk.strip():
                    print(f"[STT] Reçu: {text_chunk}")
                    text_parts.append(text_chunk)
            
            if text_parts:
                transcription = " ".join(text_parts)
                print(f"[STT] Transcription complète: {transcription}")
                return transcription
            else:
                print("[STT] Aucune transcription reçue")
                return ""
            
        except Exception as e:
            print(f"[STT] Erreur: {e}")
            import traceback
            traceback.print_exc()
            return ""
        finally:
            await self.gradium_stt.close()
    
    async def stream_llm_response(self, user_text: str):
        """Stream la réponse du LLM token par token"""
        try:
            print(f"[LLM] Envoi de la requête à Mistral: {user_text}")
            async for text_chunk in self.mistral_llm.stream_response(user_text):
                if text_chunk.strip():
                    yield text_chunk
        except Exception as e:
            print(f"[LLM] Erreur: {e}")
            import traceback
            traceback.print_exc()
            yield "Désolé, j'ai rencontré une erreur technique."
    
    async def stream_tts_and_play(self, text_stream):
        """
        Reçoit un stream de texte du LLM, accumule par phrases,
        et joue l'audio en streaming dès qu'une phrase est complète.
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
                
                # Vérifier si on a une phrase complète
                last_delimiter_pos = -1
                for i, char in enumerate(buffer):
                    if char in sentence_delimiters:
                        last_delimiter_pos = i
                
                if last_delimiter_pos >= 0:
                    # Extraire la phrase complète
                    sentence = buffer[:last_delimiter_pos + 1].strip()
                    buffer = buffer[last_delimiter_pos + 1:]
                    
                    if sentence:
                        # Générer et jouer l'audio pour cette phrase
                        await self.gradium_tts.connect()
                        await self.gradium_tts.send_text(sentence)
                        
                        async for audio_chunk in self.gradium_tts.receive_audio():
                            output_stream.write(audio_chunk)
                        
                        await self.gradium_tts.close()
            
            # Jouer le reste du buffer s'il y en a
            if buffer.strip():
                print()
                await self.gradium_tts.connect()
                await self.gradium_tts.send_text(buffer.strip())
                
                async for audio_chunk in self.gradium_tts.receive_audio():
                    output_stream.write(audio_chunk)
                
                await self.gradium_tts.close()
            
            print()
            
        finally:
            output_stream.stop_stream()
            output_stream.close()
    
    def record_audio(self, silence_threshold: float = 300,
                     silence_duration: float = 1.2, min_recording_time: float = 1.5) -> bytes:
        """Enregistre avec détection de silence"""
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
        """Un tour de conversation complet avec streaming"""
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
        """Boucle principale"""
        print("=" * 50)
        print("  Voice-to-Voice Streaming")
        print("  Gradium + Mistral")
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
        """Nettoyage"""
        self.audio.terminate()

async def main():
    v2v = VoiceToVoiceStreaming()
    try:
        await v2v.run()
    finally:
        v2v.cleanup()

if __name__ == "__main__":
    asyncio.run(main())