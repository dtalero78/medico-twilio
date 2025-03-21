import asyncio
import base64
import json
import sounddevice as sd
import numpy as np

class AudioPlayer:
    """Reproduce audio en tiempo real usando sounddevice."""
    def __init__(self):
        self.queue = asyncio.Queue()

    async def play_audio(self):
        while True:
            audio_data = await self.queue.get()
            if audio_data is None:
                break
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            sd.play(audio_np, samplerate=8000)
            sd.wait()

    async def add_audio(self, audio_data):
        await self.queue.put(audio_data)

class TwilioStreamReceiver:
    """Convierte el audio entrante de Twilio en un buffer compatible con OpenAI VoicePipeline."""
    def __init__(self, websocket):
        self.websocket = websocket

    async def __aiter__(self):
        """Permite que la clase se use como un generador asíncrono."""
        async for message in self.websocket.iter_text():
            data = json.loads(message)
            if data.get('event') == 'media':
                print(f"[debug] Mensaje de audio recibido: {data}")
                audio_data = base64.b64decode(data['media']['payload'])
                yield audio_data
