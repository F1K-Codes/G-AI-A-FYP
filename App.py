import gc
from pathlib import Path  
import json  
import sys  
import tiktoken  
import wave 
import numpy as np  
import audioop  
import torch  
from torch.utils.data import Dataset, DataLoader 
import asyncio  
import concurrent.futures   
from typing import Dict, Any   
from faster_whisper import WhisperModel  
import os   
import tempfile   
import io   
import psutil
import threading   
import chainlit as cl  
from gtts import gTTS   
from dotenv import load_dotenv   
import time   
import joblib   
from The_Guts import (
    SilenceDetector,
    generate,   
    text_to_token_ids,   
    token_ids_to_text,   
    text_to_speech,  
    speech_to_text,  
    loaded_models,
    load_default_model,
    load_model,
    unload_model,
    check_memory_usage,
    Available_Models,
    query_router,
    FarmBotSSHBridge,
    generate_with_farmbot_model,
    generate_with_plant_model,
    classify_query,
    process_audio,
    generate_text_answer,
    handle_model_command,
    get_farmbot_bridge,
    is_model_command
)
# For other PCs go to the Guts.py line 50 and change to correct drive.
# To run script: chainlit run App.py

# Determine if CUDA (GPU) is available, otherwise use CPU.
this_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

# How often (in number of generations) to check memory usage.
Memory_Check_Interval = 10

# Audio recording and silence detection parameters.
MAX_RECORD_SEC = 15             # Maximum recording length in seconds.
SILENCE_THRESHOLD_SEC = 1.5     # Silence duration to trigger auto‑stop.
VAD_MODE = 1                    # WebRTC VAD aggressiveness mode (0-3).
FRAME_DURATION_MS = 30          # Duration of each audio frame in milliseconds.


@cl.on_chat_start
async def start():
    """Initialize user session when a new chat starts, clears message history and generation counter,
    loads a query classifier if available, sets FarmBot connection status, loads the default AI model,
    attempts to auto‑connect to the FarmBot SSH bridge, displays a welcome message. """
    cl.user_session.set("message_history", [])
    cl.user_session.set("generation_count", 0)
    
    # Load pre‑trained query classifier (if present).
    classifier_path = Path("query_classifier.pkl")
    if classifier_path.exists():
        cl.user_session.set("classifier", joblib.load(classifier_path))
    else:
        cl.user_session.set("classifier", None)
    
    cl.user_session.set("farmbot_connected", False)
    await load_default_model() 
    
    # Build a status message showing which models are currently loaded.
    model_status = []
    for model_id, info in Available_Models.items():
        status = "Loaded" if model_id in loaded_models else "Not Loaded"
        model_status.append(f"{'status'} - {info['name']}")
    model_info = "\n".join(model_status)
    
    # Try to connect to the FarmBot bridge automatically.
    try:
        bridge = await get_farmbot_bridge()
        if not bridge.is_connected:
            await cl.Message(content="Connecting to Farmbot Bridge...").send() 
            await bridge.connect()
            cl.user_session.set("farmbot_connected", bridge.is_connected)
            status = "connected" if bridge.is_connected else "failed"
            await cl.Message(content=f"FarmBot bridge connect status: {status}.").send()
        
        else:
            await cl.Message(content="FarmBot bridge already connected.").send()
    except Exception as e:
        await cl.Message(content=f"Error connecting to FarmBot bridge: {e}").send()

    time.sleep(5)  
    await cl.Message(
        content="Im G-AI-A, the Plant Care Assistant! Please select a model to get started: \n"
                "To see the list of commands click on Readme in the corner."
    ).send()
    
    await cl.ChatSettings(disable_input=True).send()

@cl.on_audio_start
async def on_audio_start():
    """Prepare for audio recording when the user starts speaking, initializes empty audio chunks list,
    creates a SilenceDetector to monitor for silence, starts a timer that will force‑stop recording after MAX_RECORD_SEC."""
    cl.user_session.set("audio_chunks", [])
    cl.user_session.set("recording_active", True)
    cl.user_session.set("stop_reason", None)
    
    detector = SilenceDetector(
        sample_rate=16000,
        frame_duration_ms=FRAME_DURATION_MS,
        silence_threshold_sec=SILENCE_THRESHOLD_SEC,
        vad_mode=VAD_MODE
    )
    cl.user_session.set("silence_detector", detector)
    return True

@cl.on_audio_end
async def on_audio_end():
    """Handle the end of an audio recording (user stopped manually).
    Cancels the force‑stop timer and triggers processing of the recorded audio."""
    if cl.user_session.get("recording_active"):
        cl.user_session.set("recording_active", False)
        await process_audio()

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    """Process each incoming audio chunk during recording, appends the chunk to the list,
    feeds the chunk to the silence detector, if silence threshold is reached, stops recording and triggers processing.
    """
    if not cl.user_session.get("recording_active"):
        return

    audio_bytes = chunk.data
    detector = cl.user_session.get("silence_detector")
    chunks = cl.user_session.get("audio_chunks")
    chunks.append(np.frombuffer(audio_bytes, dtype=np.int16))

    if detector.process_audio(audio_bytes):
        cl.user_session.set("recording_active", False)
        cl.user_session.set("stop_reason", "silence")
        await process_audio()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle a text message from the user, checks if the message is a FarmBot action command and executes it,
    handles model‑management commands, classifies the query (plant care vs. FarmBot) and generates a text response,
    converts the response to speech and sends it back as an audio element.
    """
    if cl.user_session.get("recording_active"):
        cl.user_session.set("recording_active", False)
        # Discard any accumulated audio chunks
        cl.user_session.set("audio_chunks", [])
         
    content = message.content.strip()
    text_lower = content.lower()
    
    # Detect FarmBot action keywords and ensure bridge is connected.
    if any(keyword in text_lower for keyword in ['move', 'calibrate', 'home', 'water', 'tool', 'estop', 'mount', 'unmount']):
        bridge = cl.user_session.get("ros_bridge")
        if not (bridge and bridge.is_connected):
            await cl.Message(content="FarmBot bridge is not connected. Connect to the bridge to use this command.").send()
            return
    
    # Attempt to parse the message as a direct action command.
    action, params = query_router.parse_action_command(content)
    if action:
        print(f"Parsed action: {action} with params: {params}")
        bridge = cl.user_session.get("ros_bridge")
        if bridge and bridge.is_connected:
            method = getattr(bridge, action, None)
            if method:
                success = method(**params)
                if success:
                    await cl.Message(content=f"Command '{action}' sent to FarmBot.").send()
                else:
                    await cl.Message(content=f"Failed to send '{action}'.").send()
            else:
                await cl.Message(content=f"Unknown action: {action}").send()
        else:
            await cl.Message(content="FarmBot not connected.").send()
        return
    
    # Handle model/bridge management commands.
    if is_model_command(text_lower):
        if "connect" in text_lower and "bridge" in text_lower:
            await cl.Message(content="Connecting to FarmBot...").send()
            bridge = await get_farmbot_bridge()
            if not bridge.is_connected:
                await bridge.connect()
                cl.user_session.set("farmbot_connected", bridge.is_connected)
                status = "connected" if bridge.is_connected else "failed"
                await cl.Message(content=f"FarmBot bridge {status}.").send()
            else:
                await cl.Message(content="FarmBot bridge already connected.").send()
            return
        
        elif "disconnect" in text_lower and "bridge" in text_lower:
            bridge = cl.user_session.get("ros_bridge")
            if bridge and bridge.is_connected:
                bridge.shutdown()
                cl.user_session.set("farmbot_connected", False)
                await cl.Message(content="FarmBot bridge disconnected.").send()
            else:
                await cl.Message(content="FarmBot bridge not connected.").send()
            return
        
        else:
            response = await handle_model_command(content)
            await cl.Message(content=response).send()
            return
    
    # Classify the query and generate a response using the appropriate model.
    query_type = await classify_query(content)
    async with cl.Step(name=f"Generating response with {query_type} model"):
        answer = await generate_text_answer(content)
        
    if not answer:
        answer = "Sorry, I couldn't generate a response to that."
 
    generation_count = cl.user_session.get("generation_count", 0) + 1
    cl.user_session.set("generation_count", generation_count)
    
    # Periodically check system memory usage.
    if generation_count % Memory_Check_Interval == 0:
        if not check_memory_usage():
            await cl.Message(content="Memory usage is high. Consider unloading a model.").send()
          
    # Store the conversation history.
    message_history = cl.user_session.get("message_history") or []
    message_history.append({"role": "user", "content": message.content})
    message_history.append({"role": "assistant", "content": answer})
    cl.user_session.set("message_history", message_history)
    
    # Convert the answer to speech and send it along with the text.
    audio_bytes = await text_to_speech(answer)
    audio_element = cl.Audio(
        auto_play=True,
        mime="audio/mpeg",
        content = audio_bytes)    
    await cl.Message(content=answer, elements=[audio_element]).send()
    