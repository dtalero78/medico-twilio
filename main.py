import os
import json
import base64
import asyncio
import websockets
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi.responses import FileResponse


load_dotenv()


# 1) Instancia única de FastAPI
app = FastAPI()

# 1) Instancia mp3
@app.get("/pem.mp3")
def serve_mp3():
    # Devuelve el archivo "pem.mp3" que está en la misma carpeta que main.py
    return FileResponse("pem.mp3", media_type="audio/mpeg")

# 2) Middleware de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia si quieres restringir orígenes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3) Credenciales de Twilio (en .env)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_FROM")

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# 4) Endpoint OPTIONS para solicitudes preflight (CORS)
@app.options("/make-call")
async def options_make_call():
    return JSONResponse(content={}, status_code=200)

# 5) Endpoint para hacer la llamada con Twilio
@app.api_route("/make-call", methods=["OPTIONS", "POST"])
async def make_call(request: Request):
    data = await request.json()
    phone_number = data.get("phone")
    if not phone_number:
        return JSONResponse(content={"error": "Número no proporcionado"}, status_code=400)

    try:
        call = client.calls.create(
            to=phone_number,
            from_=TWILIO_PHONE_NUMBER,
            twiml="""
        <Response>
            <Play>https://953b9-809-84-187-208.ngrok-free.app/pem.mp3</Play>
        </Response>
    """
        )
        return JSONResponse(content={"message": "Llamada en curso", "call_sid": call.sid})
    except Exception as e:
        print("Error al crear la llamada:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ==============================
# 6) Configuración de OpenAI Realtime
# ==============================
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError('Falta la clave de OpenAI (OPENAI_API_KEY) en .env')

PORT = int(os.getenv('PORT', 5050))

# Prompt de OpenAI (recortado a una sola frase)
system_prompt = """Eres un agente creado para enamorar y conquistar a Manuela Bermudez."""
rag_chunks = """ "Daniel y Manuela se conocieron en 2022." """

SYSTEM_MESSAGE = system_prompt + rag_chunks

VOICE = 'echo'
LOG_EVENT_TYPES = [
    'error', 'response.content.done', 'rate_limits.updated',
    'response.done', 'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped', 'input_audio_buffer.speech_started',
    'session.created'
]
SHOW_TIMING_MATH = False

# ==============================
# 7) Endpoint de prueba (GET /)
# ==============================
@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}

# ==============================
# 8) Endpoint para llamadas entrantes (Twilio -> /incoming-call)
# ==============================
@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Devuelve TwiML para conectar con Media Stream."""
    response = VoiceResponse()
    response.say(
        "Hola Manu. Soy un agente que he sido programado por Daniel, el hombre que tanto te ama y admira. "
        "Puedes preguntar cualquier cosa de nuestra relación",
        voice="man", 
        language="es-ES"
    )
    response.pause(length=1)
    response.say("Ya puedes preguntar", voice="man", language="es-ES")
    host = request.url.hostname
    connect = Connect()
    connect.stream(url=f'wss://{host}/media-stream')
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

# ==============================
# 9) WebSocket para el Media Stream (Twilio <-> OpenAI)
# ==============================
@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Maneja la conexión WebSocket entre Twilio y OpenAI Realtime."""
    print("Client connected")
    await websocket.accept()

    async with websockets.connect(
        'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01',
        extra_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
    ) as openai_ws:
        await initialize_session(openai_ws)

        # Variables de estado
        stream_sid = None
        latest_media_timestamp = 0
        last_assistant_item = None
        mark_queue = []
        response_start_timestamp_twilio = None
        
        async def receive_from_twilio():
            """Recibe datos de audio de Twilio y los envía a la API Realtime de OpenAI."""
            nonlocal stream_sid, latest_media_timestamp
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data['event'] == 'media' and openai_ws.open:
                        latest_media_timestamp = int(data['media']['timestamp'])
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": data['media']['payload']
                        }
                        await openai_ws.send(json.dumps(audio_append))
                    elif data['event'] == 'start':
                        stream_sid = data['start']['streamSid']
                        print(f"Incoming stream has started {stream_sid}")
                    elif data['event'] == 'mark':
                        if mark_queue:
                            mark_queue.pop(0)
            except WebSocketDisconnect:
                print("Client disconnected.")
                if openai_ws.open:
                    await openai_ws.close()

        async def send_to_twilio():
            """Recibe eventos de OpenAI Realtime API y reenvía audio a Twilio."""
            nonlocal stream_sid, last_assistant_item, response_start_timestamp_twilio
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)
                    if response['type'] in LOG_EVENT_TYPES:
                        print(f"Received event: {response['type']}", response)

                    # Audio parcial
                    if response.get('type') == 'response.audio.delta' and 'delta' in response:
                        audio_payload = base64.b64encode(
                            base64.b64decode(response['delta'])
                        ).decode('utf-8')
                        audio_delta = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                "payload": audio_payload
                            }
                        }
                        await websocket.send_json(audio_delta)

                        if response_start_timestamp_twilio is None:
                            response_start_timestamp_twilio = latest_media_timestamp

                        if response.get('item_id'):
                            last_assistant_item = response['item_id']

                        await send_mark(websocket, stream_sid)

                    # Interrupción cuando detectamos speech del usuario
                    if response.get('type') == 'input_audio_buffer.speech_started':
                        print("Speech started detected.")
                        if last_assistant_item:
                            print(f"Interrupting response with id: {last_assistant_item}")
                            await handle_speech_started_event()
            except Exception as e:
                print(f"Error in send_to_twilio: {e}")

        async def handle_speech_started_event():
            """Interrumpe la respuesta actual si el usuario empieza a hablar."""
            nonlocal response_start_timestamp_twilio, last_assistant_item
            if mark_queue and response_start_timestamp_twilio is not None:
                elapsed_time = latest_media_timestamp - response_start_timestamp_twilio
                if last_assistant_item:
                    truncate_event = {
                        "type": "conversation.item.truncate",
                        "item_id": last_assistant_item,
                        "content_index": 0,
                        "audio_end_ms": elapsed_time
                    }
                    await openai_ws.send(json.dumps(truncate_event))

                await websocket.send_json({
                    "event": "clear",
                    "streamSid": stream_sid
                })

                mark_queue.clear()
                last_assistant_item = None
                response_start_timestamp_twilio = None

        async def send_mark(connection, stream_sid):
            if stream_sid:
                mark_event = {
                    "event": "mark",
                    "streamSid": stream_sid,
                    "mark": {"name": "responsePart"}
                }
                await connection.send_json(mark_event)
                mark_queue.append('responsePart')

        await asyncio.gather(receive_from_twilio(), send_to_twilio())

# ==============================
# 10) Inicializar sesión con OpenAI Realtime
# ==============================
async def initialize_session(openai_ws):
    """Configura la sesión con OpenAI Realtime API."""
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
        }
    }
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))
    # Si quieres que el bot hable primero, descomenta la siguiente línea
    # await send_initial_conversation_item(openai_ws)

# (Opcional) Función para enviar un mensaje inicial
async def send_initial_conversation_item(openai_ws):
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Hola, soy tu asistente de voz. ¿En qué puedo ayudarte?"
                }
            ]
        }
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))

# ==============================
# 11) Ejecutar la aplicación
# ==============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
