import os
import json
import random
import asyncio
import httpx
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from twilio.rest import Client
from dotenv import load_dotenv

from agents import Agent, function_tool
from agents.voice import (
    AudioInput,
    SingleAgentVoiceWorkflow,
    SingleAgentWorkflowCallbacks,
    VoicePipeline,
)
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

from util import AudioPlayer, TwilioStreamReceiver

load_dotenv()

app = FastAPI()

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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# -----------------------------------------------------------------------------
# TOOLS
# -----------------------------------------------------------------------------

@function_tool
def get_weather(city: str) -> str:
    """Obtiene el clima para una ciudad dada."""
    print(f"[debug] get_weather llamado con ciudad: {city}")
    opciones = ["soleado", "nublado", "lluvioso", "nevando"]
    return f"El clima en {city} es {random.choice(opciones)}."

# -----------------------------------------------------------------------------
# AGENTES
# -----------------------------------------------------------------------------

spanish_agent = Agent(
    name="Spanish",
    handoff_description="Un agente que habla en español.",
    instructions=prompt_with_handoff_instructions(
        "Estás hablando con un paciente. Sé profesional, cálido y empático. Habla en español.",
    ),
    model="gpt-4o-mini",
)

assistant_agent = Agent(
    name="BSL Assistant",
    instructions=prompt_with_handoff_instructions(
        "Eres un asistente de salud ocupacional de BSL. Responde siempre en español. Si el usuario habla en español, transfiere al agente Spanish.",
    ),
    model="gpt-4o-mini",
    handoffs=[spanish_agent],
    tools=[get_weather],
)

# -----------------------------------------------------------------------------
# CALLBACKS PARA DEBUG
# -----------------------------------------------------------------------------

class WorkflowCallbacks(SingleAgentWorkflowCallbacks):
    def on_run(self, workflow: SingleAgentVoiceWorkflow, transcription: str) -> None:
        print(f"[debug] Transcripción recibida: {transcription}")

# -----------------------------------------------------------------------------
# ENDPOINT PARA HACER LLAMADA
# -----------------------------------------------------------------------------

@app.api_route("/make-call", methods=["POST"])
async def make_call(request: Request):
    data = await request.json()
    phone_number = data.get("phone")
    ref = data.get("ref", "")
    if not phone_number:
        return JSONResponse(content={"error": "Número no proporcionado"}, status_code=400)

    # Para testing en localhost, usamos "ws" en lugar de "wss" si es necesario.
    host = request.url.hostname or "tu-dominio.com"
    protocol = "ws" if host in ["localhost", "127.0.0.1"] else "wss"

    twiml = f"""
<Response>
    <Connect>
        <Stream url="{protocol}://{host}/media-stream?ref={ref}"/>
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

# -----------------------------------------------------------------------------
# ENDPOINT WEBSOCKET PARA STREAM DE AUDIO DESDE TWILIO
# -----------------------------------------------------------------------------

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    print("[debug] Conexión de Twilio establecida")
    await websocket.accept()

    # Capturamos ID del paciente desde el parámetro 'ref'
    patient_id = websocket.query_params.get('ref')
    patient_prompt = "Eres un asistente de salud ocupacional de BSL."

    if patient_id:
        try:
            url = f"https://www.bsl.com.co/_functions/chatbot?_id={patient_id}"
            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(url)
                response.raise_for_status()
                patient_data = response.json()

            nombre = patient_data.get("primerNombre", "Paciente")
            salud = ", ".join(patient_data.get("encuestaSalud", []))
            antecedentes = ", ".join(patient_data.get("antecedentesFamiliares", []))

            patient_prompt = (
                f"Eres un asistente de salud ocupacional de BSL. El paciente se llama {nombre}. "
                f"Su historial de salud: {salud}. Sus antecedentes familiares: {antecedentes}. "
                "Salúdalo por su nombre, pregunta por cada ítem de salud y antecedentes, y despídete cordialmente."
            )
        except Exception as e:
            print("Error obteniendo datos del paciente:", e)

    # Inicia el pipeline de voz con prompt generado
    voice_workflow = SingleAgentVoiceWorkflow(
        agent=assistant_agent,
        instructions=patient_prompt,
        callbacks=WorkflowCallbacks()
    )

    pipeline = VoicePipeline(workflow=voice_workflow)
    audio_input = AudioInput(buffer=TwilioStreamReceiver(websocket))

    # Instanciar AudioPlayer y ejecutar su reproducción en segundo plano
    player = AudioPlayer()
    asyncio.create_task(player.play_audio())

    result = await pipeline.run(audio_input)

    async for event in result.stream():
        if event.type == "voice_stream_event_audio":
            print("[debug] Recibiendo audio del pipeline")
            await player.add_audio(event.data)
        elif event.type == "voice_stream_event_lifecycle":
            print(f"[debug] Evento de ciclo de vida: {event.event}")

@app.get("/")
async def index():
    return {"message": "Servidor de agente de voz BSL en ejecución."}
