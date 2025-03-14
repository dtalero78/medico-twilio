import os
import json
import base64
import asyncio
import websockets
import httpx
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from dotenv import load_dotenv

load_dotenv()

# ===================================================
# 1) Instancia única de FastAPI y configuración básica
# ===================================================
app = FastAPI()

# Endpoint para servir el archivo mp3
@app.get("/pem.mp3")
def serve_mp3():
    return FileResponse("pem.mp3", media_type="audio/mpeg")

# Middleware de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modifica si deseas restringir orígenes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================================================
# 2) Configuración de Twilio
# ===================================================
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_FROM")

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Endpoint OPTIONS para solicitudes preflight en /make-call
@app.options("/make-call")
async def options_make_call():
    return JSONResponse(content={}, status_code=200)

# ===================================================
# 3) Endpoint para iniciar llamada saliente
# ===================================================
@app.api_route("/make-call", methods=["OPTIONS", "POST"])
async def make_call(request: Request):
    """
    Recibe un JSON con:
      - "phone": número de celular a marcar.
      - "ref": identificador del paciente.
    Genera un TwiML dinámico que, al contestar, conecta al endpoint /media-stream pasando el parámetro ref.
    """
    data = await request.json()
    phone_number = data.get("phone")
    ref = data.get("ref", "")
    if not phone_number:
        return JSONResponse(content={"error": "Número no proporcionado"}, status_code=400)
    
    # Usamos el hostname del request para construir la URL del stream
    host = request.url.hostname or "tu-dominio.com"
    
    # Generamos el TwiML dinámico
    twiml = f"""
<Response>
  <Connect>
    <Stream url="wss://{host}/media-stream?ref={ref}"/>
  </Connect>
</Response>
    """
    try:
        call = client.calls.create(
            to=phone_number,
            from_=TWILIO_PHONE_NUMBER,
            twiml=twiml
        )
        return JSONResponse(content={"message": "Llamada en curso", "call_sid": call.sid})
    except Exception as e:
        print("Error al crear la llamada:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ===================================================
# 4) Configuración de OpenAI Realtime
# ===================================================
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError('Falta la clave de OpenAI (OPENAI_API_KEY) en .env')

PORT = int(os.getenv('PORT', 5050))

# Mensaje base de instrucciones (fallback)
system_prompt = "Eres un agente médico de salud laboral para verificar datos médicos de un paciente."
rag_chunks = " Pregunta por los sintomas que son los siguientes:"
SYSTEM_MESSAGE = system_prompt + rag_chunks

VOICE = 'echo'
LOG_EVENT_TYPES = [
    'error', 'response.content.done', 'rate_limits.updated',
    'response.done', 'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped', 'input_audio_buffer.speech_started',
    'session.created'
]
SHOW_TIMING_MATH = False

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}

# ===================================================
# 5) Función para obtener datos del paciente (según ref)
# ===================================================
async def fetch_patient_data(patient_id: str):
    url = f"https://www.bsl.com.co/_functions/chatbot?_id={patient_id}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()

# ===================================================
# 6) WebSocket para el Media Stream (Twilio <-> OpenAI)
# ===================================================
@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    print("Cliente conectado en /media-stream")
    await websocket.accept()

    # Extraer el parámetro ref para identificar al paciente
    patient_id = websocket.query_params.get('ref')
    if patient_id:
        try:
            patient_data = await fetch_patient_data(patient_id)
            # Se asume que la respuesta contiene las claves: primerNombre, encuestaSalud y antecedentesFamiliares
            primerNombre = patient_data.get("primerNombre", "Paciente")
            encuestaSalud = ", ".join(patient_data.get("encuestaSalud", []))
            antecedentesFamiliares = ", ".join(patient_data.get("antecedentesFamiliares", []))
            instructions = (
                f"Eres un asistente de salud ocupacional de BSL. Pregunta a este paciente sobre su historial de salud. "
                f"El paciente se llama {primerNombre}. Historial de salud: {encuestaSalud}. "
                f"Antecedentes familiares: {antecedentesFamiliares}. "
                "Salúdalo por su nombre, sé profesional, cálido y empático, y sé breve. "
                "Al finalizar, despídete e indica que en breve se comunicarán para entregar su certificado."
            )
        except Exception as e:
            print("Error al obtener datos del paciente:", e)
            instructions = SYSTEM_MESSAGE
    else:
        instructions = SYSTEM_MESSAGE

    async with websockets.connect(
        'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01',
        extra_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
    ) as openai_ws:
        await initialize_session(openai_ws, instructions)

        # Variables de estado para gestionar el stream
        stream_sid = None
        latest_media_timestamp = 0
        last_assistant_item = None
        mark_queue = []
        response_start_timestamp_twilio = None

        async def receive_from_twilio():
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
                        print(f"Stream iniciada: {stream_sid}")
                    elif data['event'] == 'mark':
                        if mark_queue:
                            mark_queue.pop(0)
            except Exception as e:
                print("Error en receive_from_twilio:", e)
                if openai_ws.open:
                    await openai_ws.close()

        async def send_to_twilio():
            nonlocal stream_sid, last_assistant_item, response_start_timestamp_twilio
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)
                    if response['type'] in LOG_EVENT_TYPES:
                        print(f"Evento recibido: {response['type']}", response)
                    # Procesar audio parcial recibido desde OpenAI y reenviarlo a Twilio
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
                    # Si se detecta inicio de habla del usuario, interrumpir respuesta actual
                    if response.get('type') == 'input_audio_buffer.speech_started':
                        print("Inicio de habla detectado.")
                        if last_assistant_item:
                            print(f"Interrumpiendo respuesta con id: {last_assistant_item}")
                            await handle_speech_started_event()
            except Exception as e:
                print("Error en send_to_twilio:", e)

        async def handle_speech_started_event():
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

# ===================================================
# 7) Función para inicializar la sesión con OpenAI Realtime
# ===================================================
async def initialize_session(openai_ws, instructions):
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "instructions": instructions,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
        }
    }
    print('Enviando actualización de sesión:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))

# ===================================================
# 8) Ejecutar la aplicación
# ===================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
