import os
import json
import base64
import asyncio
import websockets
import httpx
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

@app.get("/pem.mp3")
def serve_mp3():
    return FileResponse("pem.mp3", media_type="audio/mpeg")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_FROM")

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

@app.options("/make-call")
async def options_make_call():
    return JSONResponse(content={}, status_code=200)

@app.api_route("/make-call", methods=["OPTIONS", "POST"])
async def make_call(request: Request):
    data = await request.json()
    phone_number = data.get("phone")
    ref = data.get("ref", "")
    if not phone_number:
        return JSONResponse(content={"error": "Número no proporcionado"}, status_code=400)
    
    host = request.url.hostname or "tu-dominio.com"
    
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

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError('Falta la clave de OpenAI (OPENAI_API_KEY) en .env')

PORT = int(os.getenv('PORT', 5050))

VOICE = 'echo'
LOG_EVENT_TYPES = [
    'error', 'response.content.done', 'rate_limits.updated',
    'response.done', 'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped', 'input_audio_buffer.speech_started',
    'session.created'
]

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}

async def fetch_patient_data(patient_id: str):
    url = f"https://www.bsl.com.co/_functions/chatbot?_id={patient_id}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    print("Cliente conectado en /media-stream")
    await websocket.accept()

    # 1) Extraemos ref (patient_id)
    patient_id = websocket.query_params.get('ref')
    if patient_id:
        try:
            patient_data = await fetch_patient_data(patient_id)
            primerNombre = patient_data.get("primerNombre", "Paciente")
            encuestaSalud = ", ".join(patient_data.get("encuestaSalud", []))
            antecedentesFamiliares = ", ".join(patient_data.get("antecedentesFamiliares", []))

            # Construimos el prompt
            system_prompt = (
                f"Eres un asistente de salud ocupacional de BSL. "
                f"El paciente se llama {primerNombre}. "
                f"Su historial de salud: {encuestaSalud}. "
                f"Sus antecedentes familiares: {antecedentesFamiliares}. "
                "Al iniciar la llamada, salúdalo por su nombre y pregunta específicamente "
                "por cada uno de los puntos de su historial de salud y de sus antecedentes familiares. "
                "Verifica si hay novedades o cambios en cada ítem. "
                "Sé profesional, cálido y empático, y sé breve. "
                "Al finalizar, despídete indicando que pronto se comunicarán para entregar su certificado."
            )
        except Exception as e:
            print("Error al obtener datos del paciente:", e)
            system_prompt = "Eres un asistente de salud ocupacional de BSL."
    else:
        system_prompt = "Eres un asistente de salud ocupacional de BSL."

    # 2) Conectamos con OpenAI Realtime
    async with websockets.connect(
        'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01',
        extra_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
    ) as openai_ws:
        # Inicializamos la sesión (sin instructions, para no enmascarar el system message)
        await initialize_session(openai_ws)

        # 3) Enviamos el system message
        await send_system_message_item(openai_ws, system_prompt)

        # 4) Enviamos un mensaje de usuario que dispare la respuesta del bot
        await send_initial_user_item(openai_ws)

        # Variables de estado
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

                    # Procesar audio parcial
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

                    # Interrupción si el usuario empieza a hablar
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
                await connection.send_json(json.dumps(mark_event))
                mark_queue.append('responsePart')

        await asyncio.gather(receive_from_twilio(), send_to_twilio())


# ----------------------------------------------------------------------
# FUNCIONES DE INICIALIZACIÓN DE SESIÓN Y ENVÍO DE MENSAJES
# ----------------------------------------------------------------------

async def initialize_session(openai_ws):
    """
    Inicia la sesión con el Realtime API sin poner instructions aquí,
    para no enmascarar el system message que crearemos manualmente.
    """
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
        }
    }
    print("Enviando session.update sin instructions.")
    await openai_ws.send(json.dumps(session_update))

async def send_system_message_item(openai_ws, system_prompt: str):
    """
    Envía un mensaje de rol 'system' con el prompt detallado.
    Esto obliga al modelo a seguir estas directrices como prioridad.
    """
    system_message_event = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt
                }
            ]
        }
    }
    print("Enviando system message con el prompt:", system_prompt)
    await openai_ws.send(json.dumps(system_message_event))
    # Forzamos la respuesta para que el modelo tome en cuenta este system message
    await openai_ws.send(json.dumps({"type": "response.create"}))

async def send_initial_user_item(openai_ws):
    """
    Envía un mensaje de usuario que detona la primera respuesta del bot,
    basándose en el system message ya presente.
    """
    user_message_event = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "La llamada ha sido contestada. Por favor saluda al paciente "
                        "por su nombre y pregunta por su historial y antecedentes."
                    )
                }
            ]
        }
    }
    print("Enviando user message inicial para detonar respuesta.")
    await openai_ws.send(json.dumps(user_message_event))
    # Generamos la respuesta del bot
    await openai_ws.send(json.dumps({"type": "response.create"}))

# ----------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5050)))
