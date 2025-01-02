import os
from io import BytesIO
import httpx
from openai import AsyncOpenAI
from chainlit.element import ElementBased
import chainlit as cl
from dotenv import load_dotenv
import requests

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LITERAL_API_KEY = os.getenv("LITERAL_API_KEY")
SERVER_URL = os.getenv("SERVER_URL")
SPEECHACE_API_KEY = os.getenv("SPEECHACE_API_KEY")

print(ELEVENLABS_API_KEY)
print(ELEVENLABS_VOICE_ID)
print(OPENAI_API_KEY)
print(LITERAL_API_KEY)
print(SERVER_URL)
print(SPEECHACE_API_KEY)

cl.instrument_openai()

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID or not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY, ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID must be set")

actions = [
    cl.Action(name="Housing", value="1", description="Finding a Suitable House for Rent for Grad Students at a University"),
    cl.Action(name="Educational Opportunities", value="2", description="Interview for Participation in a Cultural Program in the Yucatán Peninsula, Mexico"),
    cl.Action(name="Health and Wellness", value="3", description="Discussing Health Issues at a University Health Facility in a Spanish-Speaking Country"),
    cl.Action(name="Zoom Interview", value="4", description="virtual format of a Zoom interview practice session, highlighting the importance of clear communication in Spanish, preparation for professional settings, and useful feedback from the career counselor")
]

actionsSummary = [
    cl.Action(name="Summary", value="summary", description="Summarize the conversation"),
]

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if "voiceflow" in username or "princeton" in username: 
        return cl.User(
            identifier=username, metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None

async def speech_to_text(audio_file):
    response = await client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )

    return response.text


async def generate_text_answer(transcription, fluency_info=None):
    user = cl.user_session.get("user")
    sessionId = cl.user_session.get("id")
    scenraioId = cl.user_session.get("scenario")
    url = f"{SERVER_URL}/v1/interact"
    headers = {
        "Content-Type": "application/json",
        }

    data = {
        "input": transcription,
        "userId": user.identifier,
        "sessionId":  sessionId,
        "scenarioId": scenraioId,
        "language": cl.user_session.get("language"),
        "fluencyInfo": fluency_info != None and fluency_info or {}
    }
    print("URL: "+url)
    async with httpx.AsyncClient(timeout=25.0) as client:
        response = await client.post(url, json=data, headers=headers)
        response.raise_for_status()  # Ensure we notice bad responses
        return response.json()["result"]
    
async def fluencyDetection(audioFile: str, user_id: str):
    # Set the URL and parameters
    url = "https://api.speechace.co/api/scoring/speech/v9/json"
    params = {
        "key": SPEECHACE_API_KEY,
        "dialect": cl.user_session.get("locale").lower(),
        "user_id": user_id,
        "include_ielts_feedback": "1"
    }
    # Define the file and additional data
    files = {
        "user_audio_file": open(audioFile, "rb")
    }

    # Send the POST request
    response = requests.post(url, params=params, files=files)

    # Print the response
    print(response.json())
    return response.json()

async def text_to_speech(text: str, mime_type: str):
    CHUNK_SIZE = 1024

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"

    headers = {
    "Accept": mime_type,
    "Content-Type": "application/json",
    "xi-api-key": ELEVENLABS_API_KEY
    }

    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    async with httpx.AsyncClient(timeout=25.0) as client:
        response = await client.post(url, json=data, headers=headers)
        response.raise_for_status()  # Ensure we notice bad responses

        buffer = BytesIO()
        buffer.name = f"output_audio.{mime_type.split('/')[1]}"

        async for chunk in response.aiter_bytes(chunk_size=CHUNK_SIZE):
            if chunk:
                buffer.write(chunk)
        
        buffer.seek(0)
        return buffer.name, buffer.read()
        

@cl.on_chat_start
async def start():
    await cl.Message(
        content="Welcome to Fluentia. Press `p` to talk! Please select the scenario you want to practice.",
        actions=actions
    ).send()


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):

    if chunk.isStart:
        buffer = BytesIO()
        # This is required for whisper to recognize the file type
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        # Initialize the session for a new audio stream
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    # TODO: Use Gladia to transcribe chunks as they arrive would decrease latency
    # see https://docs-v1.gladia.io/reference/live-audio
    
    # For now, write the chunks to a buffer and transcribe the whole audio at the end
    cl.user_session.get("audio_buffer").write(chunk.data)

def storeAudioFile(audio_buffer):
    # Specify the output file name
    output_file = f".files/input/{audio_buffer.name}"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write the contents of the BytesIO object to the file
    with open(output_file, "wb") as file:
        file.write(audio_buffer.getvalue())

    return output_file

async def setLangueage(transcription):
    url = f"{SERVER_URL}/v1/detectLanguage"
    headers = {
        "Content-Type": "application/json",
        }

    data = {
        "input": transcription,
    }
    print("language set to: "+url)
    async with httpx.AsyncClient(timeout=25.0) as client:
        response = await client.post(url, json=data, headers=headers)
        response.raise_for_status()  # Ensure we notice bad responses
        cl.user_session.set("locale", response.json()["result"]["locale"])
        cl.user_session.set("language", response.json()["result"]["language"])
    print("language set to: "+response.json()["result"]["language"])
    print("locale set to: "+response.json()["result"]["locale"])

@cl.on_audio_end
async def on_audio_end(elements: list[ElementBased]):
    scenario = cl.user_session.get("scenario")
    if scenario is None:
        await cl.Message(
            content="Please select a scenario first.",
            actions=actions
        ).send()
        return

    # Get the audio buffer from the session
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)  # Move the file pointer to the beginning
    audio_file = audio_buffer.read()
    audio_mime_type: str = cl.user_session.get("audio_mime_type")

    input_audio_el = cl.Audio(
        mime=audio_mime_type, content=audio_file
    )

    output_file = storeAudioFile(audio_buffer)

    whisper_input = (audio_buffer.name, audio_file, audio_mime_type)
    transcription = await speech_to_text(whisper_input)

    if cl.user_session.get("language") == None:
        await setLangueage(transcription)

    await cl.Message(
        author="You", 
        type="user_message",
        content=transcription,
        elements=[input_audio_el, *elements]
    ).send()

    # Call to Speechace API to get the fluency information
    fluency_info = await fluencyDetection(output_file, cl.user_session.get("user").identifier)

    result = await generate_text_answer(transcription, fluency_info)
    fluentia_info = extract_fluentia_info(result)

    elements = [
        cl.Text(name="Fluentia Information", content=fluentia_info, display="inline")
    ]


    # Generate audio response
    output_name, output_audio = await text_to_speech(result["nextInteraction"], audio_mime_type)
    
    output_audio_el = cl.Audio(
        auto_play=True,
        mime=audio_mime_type,
        content=output_audio,
    )
    answer_message = await cl.Message(
        content=result["nextInteraction"],
        elements=elements,
        actions=actionsSummary
        ).send()

    answer_message.elements = [output_audio_el]
    await answer_message.update()

@cl.on_message
async def on_message(message: cl.Message):

    scenario = cl.user_session.get("scenario")
    if scenario is None:
        await cl.Message(
            content="Please select a scenario first.",
            actions=actions
        ).send()
        return
    if cl.user_session.get("language") == None:
        await setLangueage(message.content)
    result = await generate_text_answer(message.content)
    fluentia_info = extract_fluentia_info(result)
    elements = [
        cl.Text(name="Fluentia Information", content=fluentia_info, display="inline")
    ]
    await cl.Message(
        content=result["nextInteraction"],
        elements=elements,
        actions=actionsSummary
    ).send()


@cl.action_callback("Housing")
async def on_action(action: cl.Action):

    cl.user_session.set("scenario", action.value)

    await cl.Message(
        content="Perfect, let's start! Ask me a question about housing. in the language you want to practice.",
    ).send()

@cl.action_callback("Educational Opportunities")
async def on_action(action: cl.Action):

    cl.user_session.set("scenario", action.value)

    await cl.Message(
        content="Perfect, let's start! Ask me a question about Educational Opportunities. in the language you want to practice.",
    ).send()

@cl.action_callback("Health and Wellness")
async def on_action(action: cl.Action):

    cl.user_session.set("scenario", action.value)

    await cl.Message(
        content="Perfect, let's start! Ask me a question about Health and Wellness. in the language you want to practice.",
    ).send()

@cl.action_callback("Zoom Interview")
async def on_action(action: cl.Action):

    cl.user_session.set("scenario", action.value)

    await cl.Message(
        content="Perfect, let's start this interview! what is your name?. in the language you want to practice.",
    ).send()

@cl.action_callback("Summary")
async def on_action(action: cl.Action):

    user = cl.user_session.get("user")
    sessionId = cl.user_session.get("id")
    url = f"{SERVER_URL}/v1/summarize"
    headers = {
        "Content-Type": "application/json",
        }

    data = {
        "userId": user.identifier,
        "sessionId":  sessionId,
    }
    print("URL: "+url)
    async with httpx.AsyncClient(timeout=25.0) as client:
        response = await client.post(url, json=data, headers=headers)
        response.raise_for_status()  # Ensure we notice bad responses
        await cl.Message(
            content=response.json()["result"],
        ).send()

def extract_fluentia_info(result):
    fluentia_info = ""

    if "errors" in result and result["errors"] !="":
        fluentia_info += "**Errors detected:** "+result["errors"]

    if "solution" in result and result["solution"] !="":
        fluentia_info += "\n**Correct Sentence:** "+result["solution"]

    if "tip" in result and result["tip"] !="":
        fluentia_info += "\n**Tip:** "+result["tip"]

    if "sentiment" in result and result["sentiment"] !="":
        fluentia_info += "\n**Sentiment:** "+result["sentiment"]

    if "correctness" in result and result["correctness"] !="":
        fluentia_info += "\n**Correctness:** "+str(result["correctness"])
    if "speechAndPronunciationFluency" in result and result["speechAndPronunciationFluency"] !="":
        fluentia_info += "\**Speech and Pronunciation Fluency:** "+str(result["speechAndPronunciationFluency"])
    print(fluentia_info)
    return fluentia_info