import os
import io
import wave
import httpx
import numpy as np
from openai import AsyncOpenAI
from chainlit.element import ElementBased
import chainlit as cl
from dotenv import load_dotenv
import requests
import uuid
import audioop

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
    cl.Action(name="Housing", payload={"key": "1"}, tooltip="Finding a Suitable House for Rent for Grad Students at a University"),
    cl.Action(name="Educational Opportunities", payload={"key": "2"}, tooltip="Interview for Participation in a Cultural Program in the Yucatán Peninsula, Mexico"),
    cl.Action(name="Health and Wellness", payload={"key": "3"}, tooltip="Discussing Health Issues at a University Health Facility in a Spanish-Speaking Country"),
    cl.Action(name="Zoom Interview", payload={"key": "4"}, tooltip="virtual format of a Zoom interview practice session, highlighting the importance of clear communication in Spanish, preparation for professional settings, and useful feedback from the career counselor")
]

actionsSummary = [
    cl.Action(name="Summary", payload={"key": "summary"}, tooltip="Summarize the conversation"),
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

        buffer = io.BytesIO()
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

@cl.on_audio_start
async def on_audio_start():
    try:
        scenario = cl.user_session.get("scenario")
        if scenario is None:
            await cl.Message(
                content="Please select a scenario first.",
                actions=actions
            ).send()
            return False
        else:
            cl.user_session.set("audio_chunks", [])
            return True
    except Exception as e:
        await cl.ErrorMessage(content=f"Failed to connect to OpenAI realtime: {e}").send()
        return False


# Define a threshold for detecting silence and a timeout for ending a turn
SILENCE_THRESHOLD = 500  # Adjust based on your audio level (e.g., lower for quieter audio)
SILENCE_TIMEOUT = 2000.0    # Seconds of silence to consider the turn finished

# Variables to track state
last_elapsed_time = None
silent_duration_ms = 0
is_speaking = False

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    audio_chunks = cl.user_session.get("audio_chunks")

    global last_elapsed_time, silent_duration_ms, is_speaking

    # If this is the first chunk, initialize timers and state
    if last_elapsed_time is None:
        last_elapsed_time = chunk.elapsedTime
        is_speaking = True
        print("Audio stream started")
        return
    # Calculate the time difference between this chunk and the previous one
    time_diff_ms = chunk.elapsedTime - last_elapsed_time
    last_elapsed_time = chunk.elapsedTime

    # Compute the RMS (root mean square) energy of the audio chunk
    audio_energy = audioop.rms(chunk.data, 2)  # Assumes 16-bit audio (2 bytes per sample)

    if audio_energy < SILENCE_THRESHOLD:
        # Audio is considered silent
        silent_duration_ms += time_diff_ms
        if silent_duration_ms >= SILENCE_TIMEOUT and is_speaking:
            print("Turn finished: Silence detected")
            is_speaking = False
            print("Processing audio")
            #await process_audio()
    else:
        # Audio is not silent, reset silence timer and mark as speaking
        silent_duration_ms = 0
        if not is_speaking:
            print("Speaking resumed")
            is_speaking = True

    
    if audio_chunks is not None:
        audio_chunk= np.frombuffer(chunk.data, dtype=np.int16)
        audio_chunks.append(audio_chunk)


def storeAudioFile(audio_buffer):
    # Specify the output file name
    output_file = f".files/input/input_audio"+str(uuid.uuid4())+".wav"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write the contents of the BytesIO object to the file
    with open(output_file, "wb") as file:
        file.write(audio_buffer)

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
async def on_audio_end():
    print("Audio stream ended")
    #await process_audio()

async def process_audio():

    # Get the audio buffer from the session
    if audio_chunks:=cl.user_session.get("audio_chunks"):
       # Concatenate all chunks
        concatenated = np.concatenate(list(audio_chunks))
        
        # Create an in-memory binary stream
        wav_buffer = io.BytesIO()
        
        # Create WAV file with proper parameters
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wav_file.setframerate(24000)  # sample rate (24kHz PCM)
            wav_file.writeframes(concatenated.tobytes())
        
        # Reset buffer position
        wav_buffer.seek(0)
        
        cl.user_session.set("audio_chunks", [])

    audio_buffer = wav_buffer.getvalue()

    input_audio_el = cl.Audio(content=audio_buffer, mime="audio/wav", )

    output_file = storeAudioFile(audio_buffer)

    whisper_input = ("audio.wav", audio_buffer, "audio/wav")
    transcription = await speech_to_text(whisper_input)

    if cl.user_session.get("language") == None:
        await setLangueage(transcription)

    await cl.Message(
        author="You", 
        type="user_message",
        content=transcription,
        elements=[input_audio_el]
    ).send()

    # Call to Speechace API to get the fluency information
    fluency_info = await fluencyDetection(output_file, cl.user_session.get("user").identifier)

    result = await generate_text_answer(transcription, fluency_info)
    fluentia_info = extract_fluentia_info(result)

    elements = [
        cl.Text(name="Fluentia Information", content=fluentia_info, display="inline")
    ]


    # Generate audio response
    output_name, output_audio = await text_to_speech(result["nextInteraction"], "audio/wav")
    
    output_audio_el = cl.Audio(
        auto_play=True,
        mime="audio/wav",
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

    cl.user_session.set("scenario", action.payload["key"])

    await cl.Message(
        content="Perfect, let's start! Ask me a question about housing. in the language you want to practice.",
    ).send()

@cl.action_callback("Educational Opportunities")
async def on_action(action: cl.Action):

    cl.user_session.set("scenario", action.payload["key"])

    await cl.Message(
        content="Perfect, let's start! Ask me a question about Educational Opportunities. in the language you want to practice.",
    ).send()

@cl.action_callback("Health and Wellness")
async def on_action(action: cl.Action):

    cl.user_session.set("scenario", action.payload["key"])

    await cl.Message(
        content="Perfect, let's start! Ask me a question about Health and Wellness. in the language you want to practice.",
    ).send()

@cl.action_callback("Zoom Interview")
async def on_action(action: cl.Action):

    cl.user_session.set("scenario", action.payload["key"])

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