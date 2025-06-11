# combined_robot_main.py

import RPi.GPIO as GPIO
import time
import cv2
from picamera2 import Picamera2
from libcamera import controls
from inference_sdk import InferenceHTTPClient
from collections import deque
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import pygame   # For alarm sound
import threading # For concurrent tracking and agent loops
import asyncio
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.mcp import MCPTools
from agno.media import Image
import speech_recognition as sr
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# --- Piper TTS Imports ---
try:
    from piper import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    print("Piper TTS not found. Please install it with 'pip install piper-tts'. Speech output will be text-only.")
    PIPER_AVAILABLE = False
import wave
import sounddevice as sd
import soundfile as sf
# -------------------------


# ==============================================================================
# --- 1. COMBINED CONFIGURATIONS & INITIALIZATIONS
# ==============================================================================

# --- Environment & Supabase Configuration ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
USER_ID = os.getenv("USER_ID")

# --- Tracking & Motor Configuration ---
is_in_fallen_state = False # NEW: State variable to track an ongoing fall
MOTOR1_IN1, MOTOR1_IN2 = 17, 27
MOTOR2_IN3, MOTOR2_IN4 = 23, 22
FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS = 640, 480, 10.0
FRAME_DURATION_TARGET = 1.0 / TARGET_FPS
TRACKER_TYPE = "CSRT"
params = {
    'TARGET_BODY_WIDTH_MIN': 150, 'TARGET_BODY_WIDTH_MAX': 480, 'CENTER_TOLERANCE_X': 60,
    'MOVE_PULSE_DURATION': 0.1, 'TURN_PULSE_DURATION': 0.08, 'REDETECT_INTERVAL_FRAMES': 40,
    'ENABLE_BBOX_SHRINK': False, 'BBOX_SHRINK_W_FACTOR': 0.9, 'BBOX_SHRINK_H_FACTOR': 0.9,
}

# --- Roboflow Configuration ---
ROBOFLOW_API_KEY = "oFEgajQrFEOBGW95MyNi"
ROBOFLOW_MODEL_ID = "people-detection-o4rdr/9"
ROBOFLOW_API_URL = "https://serverless.roboflow.com"
CONFIDENCE_THRESHOLD, ROBOFLOW_REQUEST_RATE = 0.5, 10.0

# --- Fall Detection Configuration ---
FALL_DETECTION_ENABLED = True # Default, will be updated from Supabase
USE_SUDDEN_CHANGE_LOGIC = True  # <<< NEW: Set to True for production, False for simple debug mode
FALL_ASPECT_RATIO_THRESHOLD = 1.2
FALL_COOLDOWN_SECONDS = 10 # Reduced cooldown to allow for faster testing
last_fall_time = 0
bbox_history = deque(maxlen=5)

# --- Agent & Speech Configuration ---
VOICE_MODEL_PATH = "amy-low.onnx"
VOICE_CONFIG_PATH = "amy-low.onnx.json"
recognizer = sr.Recognizer()


# ==============================================================================
# --- 2. COMBINED CLASSES (Unchanged)
# ==============================================================================

class MotorController:
    def __init__(self):
        GPIO.setmode(GPIO.BCM); GPIO.setwarnings(False)
        self.pins = [MOTOR1_IN1, MOTOR1_IN2, MOTOR2_IN3, MOTOR2_IN4]
        for pin in self.pins: GPIO.setup(pin, GPIO.OUT); GPIO.output(pin, GPIO.LOW)
        self.current_action = "stop"; print("Motor controller initialized.")
    def _set_motor_l(self, s1, s2): GPIO.output(MOTOR1_IN1, s1); GPIO.output(MOTOR1_IN2, s2)
    def _set_motor_r(self, s3, s4): GPIO.output(MOTOR2_IN3, s3); GPIO.output(MOTOR2_IN4, s4)
    def forward(self, dur=0.1): self._set_motor_l(GPIO.HIGH,GPIO.LOW); self._set_motor_r(GPIO.HIGH,GPIO.LOW); self.current_action="pulsing_forward"; time.sleep(dur); self.stop(force=True)
    def backward(self, dur=0.1): self._set_motor_l(GPIO.LOW,GPIO.HIGH); self._set_motor_r(GPIO.LOW,GPIO.HIGH); self.current_action="pulsing_backward"; time.sleep(dur); self.stop(force=True)
    def turn_left(self, dur=0.05): self._set_motor_l(GPIO.LOW,GPIO.HIGH); self._set_motor_r(GPIO.HIGH,GPIO.LOW); self.current_action="pulsing_left"; time.sleep(dur); self.stop(force=True)
    def turn_right(self, dur=0.05): self._set_motor_l(GPIO.HIGH,GPIO.LOW); self._set_motor_r(GPIO.LOW,GPIO.HIGH); self.current_action="pulsing_right"; time.sleep(dur); self.stop(force=True)
    def stop(self, force=False):
        if self.current_action!="stop" or force: self._set_motor_l(GPIO.LOW,GPIO.LOW); self._set_motor_r(GPIO.LOW,GPIO.LOW); self.current_action="stop"
    def cleanup(self): print("Cleaning up GPIO."); self.stop(force=True); GPIO.cleanup()


class SupabaseHelper:
    def __init__(self, supabase_client, user_id):
        self.supabase = supabase_client
        self.user_id = user_id
    def get_all_reminders(self) -> str:
        try:
            result = self.supabase.table("reminders").select("*").eq("user_id", self.user_id).order("reminder_time").execute()
            if not result.data: return "No reminders found."
            reminders_text = "Your Reminders:\n\n"
            for r in result.data:
                reminders_text += f"ID: {r['id']}\nTitle: {r['title']}\nDescription: {r['description']}\nTime: {datetime.fromisoformat(r['reminder_time'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')}\n\n"
            return reminders_text
        except Exception as e: return f"Error getting reminders: {str(e)}"
    def get_all_memories(self) -> Tuple[str, List[str]]:
        image_urls: List[str] = []
        try:
            result = self.supabase.table("memories").select("*").eq("user_id", self.user_id).order("created_at", desc=True).execute()
            if not result.data: return "No memories found.", []
            memories_text = "Your Memories:\n\n"
            for m in result.data:
                memories_text += f"ID: {m['id']}\nTitle: {m['title']}\nContent: {m['content']}\n"
                image_url = m.get("image_url")
                if image_url:
                    memories_text += f"Image: Associated image exists.\n"
                    image_urls.append(image_url)
                memories_text += f"Created: {datetime.fromisoformat(m['created_at'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')}\n---\n"
            return memories_text, image_urls
        except Exception as e: return f"Error getting memories: {str(e)}", []


# ==============================================================================
# --- 3. COMBINED HELPER FUNCTIONS
# ==============================================================================

def get_fall_detection_status(client: Client) -> bool:
    try:
        result = client.table("device_status").select("fall_detection_enabled").eq("user_id", USER_ID).execute()
        if result.data:
            status_bool = result.data[0]['fall_detection_enabled']
            status_str = 'ENABLED' if status_bool else 'DISABLED'
            print(f"[State Check] Fall detection is {status_str}.")
            return status_bool
        else:
            print("[State Init] No status found for user. Creating default entry (ENABLED=True).")
            client.table("device_status").insert({"user_id": USER_ID, "fall_detection_enabled": True}).execute()
            return True
    except Exception as e:
        print(f"[State Error] Could not fetch or create status in Supabase: {e}. Defaulting to ON.")
        return True

def play_alarm():
    if os.path.exists("alarm.wav"):
        if not pygame.mixer.music.get_busy(): # Only play if not already playing
            pygame.mixer.music.load("alarm.wav")
            pygame.mixer.music.play(-1) # The -1 makes it loop indefinitely
            print("!!! ALARM SOUNDING !!!")
    else:
        print("!!! ALARM.WAV NOT FOUND. CANNOT PLAY SOUND. !!!")

def stop_alarm():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
        print("--- Alarm stopped ---")

def log_fall_to_supabase(client: Client):
    try:
        data, count = client.table("fall_events").insert({"user_id": USER_ID}).execute()
        print("[Report] Successfully logged fall event to Supabase." if count else "[Report Error] Failed to log fall event.")
    except Exception as e:
        print(f"[Report Error] Exception while logging fall to Supabase: {e}")


# <<< NEW: Final, STATEFUL check_for_fall function >>>
def check_for_fall(bbox: tuple, supabase_client: Client):
    global last_fall_time, is_in_fallen_state # Add the new state variable here
    
    x, y, w, h = bbox
    if h == 0 or w < 10 or h < 10:
        if is_in_fallen_state: stop_alarm()
        return False # Invalid bbox
        
    aspect_ratio = w / h
    bbox_history.append({'ar': aspect_ratio})

    mode_str = "SUDDEN" if USE_SUDDEN_CHANGE_LOGIC else "SIMPLE"
    print(
        f"[Fall Check ({mode_str})] AR: {aspect_ratio:.2f} | "
        f"History: {len(bbox_history)}/{bbox_history.maxlen} | "
        f"State: {'FALLEN' if is_in_fallen_state else 'NORMAL'}"
    )

    if len(bbox_history) < bbox_history.maxlen:
        return False # Not enough data

    # --- State Logic ---
    
    # Condition to RESET the fall state (i.e., person got up)
    # Check if the average aspect ratio is back in the "standing" range
    if is_in_fallen_state:
        # We check the average of the last few frames to be sure they are standing again
        avg_ar = sum(b['ar'] for b in bbox_history) / len(bbox_history)
        if avg_ar < (FALL_ASPECT_RATIO_THRESHOLD * 0.75): # Use a lower threshold to confirm they stood up
            print("--- PERSON HAS STOOD UP ---")
            is_in_fallen_state = False
            stop_alarm()
    
    # Condition to TRIGGER a new fall
    if not is_in_fallen_state and aspect_ratio > FALL_ASPECT_RATIO_THRESHOLD:
        fall_triggered_this_frame = False
        if USE_SUDDEN_CHANGE_LOGIC:
            previous_frames = list(bbox_history)[:-1]
            if previous_frames:
                avg_previous_ar = sum(b['ar'] for b in previous_frames) / len(previous_frames)
                if avg_previous_ar < (FALL_ASPECT_RATIO_THRESHOLD * 0.9):
                    fall_triggered_this_frame = True
        else: # Simple logic
            fall_triggered_this_frame = True
        
        if fall_triggered_this_frame and (time.monotonic() - last_fall_time > FALL_COOLDOWN_SECONDS):
            print("="*40)
            print(f"!!! NEW FALL DETECTED ({mode_str} MODE) !!!")
            print("="*40)
            log_fall_to_supabase(supabase_client)
            last_fall_time = time.monotonic()
            is_in_fallen_state = True # SET the state

    # --- Action based on current state ---
    if is_in_fallen_state:
        play_alarm()
        return True # Return true to draw red box
    else:
        # This is already handled by the reset condition, but as a fallback:
        stop_alarm()
        return False


def create_tracker(tracker_type):
    if tracker_type == 'MOSSE': return cv2.legacy.TrackerMOSSE_create()
    if tracker_type == 'CSRT': return cv2.TrackerCSRT_create()
    if tracker_type == 'KCF': return cv2.TrackerKCF_create()
    return cv2.TrackerKCF_create()

def detect_people_roboflow(client, frame_rgb, last_request_time):
    current_time = time.monotonic()
    min_interval = 1.0 / ROBOFLOW_REQUEST_RATE
    time_since_last = current_time - last_request_time
    if time_since_last < min_interval:
        time.sleep(min_interval - time_since_last)
    try:
        result = client.infer(frame_rgb, model_id=ROBOFLOW_MODEL_ID)
        people = [
            (int(p['x']-p['width']/2), int(p['y']-p['height']/2), int(p['width']), int(p['height']))
            for p in result.get('predictions', [])
            if p.get('class') == 'person' and p.get('confidence', 0) >= CONFIDENCE_THRESHOLD
        ]
        return people, time.monotonic()
    except Exception as e:
        print(f"Roboflow detection error: {e}")
        return [], time.monotonic()


# --- Agent & Speech Helpers (Unchanged) ---
def listen_for_speech():
    print("Listening...")
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("Processing speech...")
            text = recognizer.recognize_google(audio)
            print(f"Recognized: {text}")
            return text
        except (sr.WaitTimeoutError, sr.UnknownValueError):
            return None
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            return None

def speak_text(text, piper_voice):
    if piper_voice:
        print(f"Agent (speaking with Piper): {text}")
        temp_wav_path = "temp_piper_output.wav"
        try:
            with wave.open(temp_wav_path, "wb") as wav_file:
                piper_voice.synthesize(text, wav_file)
            data, fs = sf.read(temp_wav_path, dtype='float32')
            sd.play(data, fs)
            sd.wait()
        except Exception as e:
            print(f"Error during Piper TTS or playback: {e}")
            print(f"Agent (fallback print): {text}")
        finally:
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
    else:
        print(f"Agent (speaking - print fallback): {text}")


# ==============================================================================
# --- 4. CORE LOGIC FUNCTIONS (for threading)
# ==============================================================================

def run_tracking_and_fall_detection(picam2: Picamera2, motors: MotorController, supabase_client: Client, roboflow_client: InferenceHTTPClient):
    global FALL_DETECTION_ENABLED
    
    tracking_active, tracker, bbox, frame_counter = False, None, None, 0
    last_status_check_time = time.monotonic()
    STATUS_CHECK_INTERVAL = 10

    cv2.namedWindow('Body Following Robot')
    print("[Tracking Thread] Started.")
    try:
        while True:
            loop_start_time = time.monotonic()
            
            if loop_start_time - last_status_check_time > STATUS_CHECK_INTERVAL:
                # Check supabase for any changes to fall detection status
                new_status = get_fall_detection_status(supabase_client)
                if not new_status and FALL_DETECTION_ENABLED:
                    stop_alarm() # Stop the alarm if it was manually disabled
                FALL_DETECTION_ENABLED = new_status
                last_status_check_time = loop_start_time

            frame_rgb = picam2.capture_array()
            display_frame = frame_rgb.copy()
            frame_bgr_for_cv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            force_redetection = (tracking_active and params['REDETECT_INTERVAL_FRAMES'] > 0 and
                                 (frame_counter := frame_counter + 1) % params['REDETECT_INTERVAL_FRAMES'] == 0)

            if not tracking_active or force_redetection:
                tracking_active, tracker, bbox = False, None, None
                motors.stop()
                bodies, _ = detect_people_roboflow(roboflow_client, frame_rgb, 0)
                if bodies:
                    bbox = max(bodies, key=lambda b: b[2] * b[3])
                    tracker = create_tracker(TRACKER_TYPE)
                    tracker.init(frame_bgr_for_cv, bbox)
                    tracking_active = True
                    frame_counter = 0
                else:
                    motors.stop()
                    stop_alarm() # Stop alarm if person is lost
                    bbox_history.clear() 
                    print("[Detection] No bodies found, history cleared.")
            
            if tracking_active:
                success, new_bbox = tracker.update(frame_bgr_for_cv)
                if success:
                    bbox = tuple(map(int, new_bbox))
                    
                    is_fallen = False
                    if FALL_DETECTION_ENABLED:
                        is_fallen = check_for_fall(bbox, supabase_client)
                    else:
                        stop_alarm() # Ensure alarm is off if feature is disabled

                    if is_fallen:
                        cv2.rectangle(display_frame, (0, 0), (FRAME_WIDTH - 1, FRAME_HEIGHT - 1), (0, 0, 255), 10)
                    
                    x, y, w, h = bbox
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Motor control should be independent of fall status
                    body_center_x, frame_center_x = x + w // 2, FRAME_WIDTH // 2
                    if body_center_x < frame_center_x - params['CENTER_TOLERANCE_X']:
                        motors.turn_left(dur=params['TURN_PULSE_DURATION'])
                    elif body_center_x > frame_center_x + params['CENTER_TOLERANCE_X']:
                        motors.turn_right(dur=params['TURN_PULSE_DURATION'])
                    elif w < params['TARGET_BODY_WIDTH_MIN']:
                        motors.forward(dur=params['MOVE_PULSE_DURATION'])
                    elif w > params['TARGET_BODY_WIDTH_MAX']:
                        motors.backward(dur=params['MOVE_PULSE_DURATION'])
                    else:
                        motors.stop()
                else:
                    tracking_active = False
                    stop_alarm() # Stop alarm if tracking is lost
            
            cv2.imshow('Body Following Robot', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[Tracking Thread] 'q' pressed. Exiting loop.")
                break
            
            sleep_time = FRAME_DURATION_TARGET - (time.monotonic() - loop_start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"[Tracking Thread] An error occurred: {e}")
    finally:
        print("[Tracking Thread] Finished.")
        stop_alarm()
        cv2.destroyAllWindows()


# ==============================================================================
# --- 5. AGENT AND MAIN EXECUTION BLOCK (Unchanged)
# ==============================================================================
async def run_conversational_agent(picam2: Picamera2, supabase_client: Client, piper_voice):
    """
    The main async loop for the conversational agent.
    This function is intended to be run in the main thread using asyncio.
    """
    speak_text("Initializing memory assistant.", piper_voice)
    try:
        async with MCPTools(command=f"uv run --with mcp --with supabase --with Pillow --with python-dotenv mcp run mcp_server.py") as mcp_tools:
            agent = Agent(
                model=Gemini(id="gemini-2.0-flash"),
                tools=[mcp_tools],
                # description="You are Memo, an AI assistant...",
                # instructions=[
                #     "You have specialized tools to assist the user: 'add_reminder', 'get_all_reminders', 'remove_reminder', 'add_memory', and 'get_all_memories'.",
                #     "To include the most recently captured image with a memory, set the 'add_image' parameter to True.",
                #     "If the user says they are going to sleep, take a nap, or lie down, you MUST call `update_fall_detection_status(enabled=False)` to prevent false alarms and inform them that you've paused the safety alert.",
                #     "If the user says they are waking up, getting up, or are active again, you MUST call `update_fall_detection_status(enabled=True)` and inform them the safety alert is back on.",
                # ],
                description=(
                    "You are Memo, an AI assistant designed to help an Alzheimer's patient in India manage their daily life. "
                    "Your primary functions are to manage reminders (add, view, remove) and personal memories (add, view, potentially with images) using a set of specialized tools. "
                    "You can also understand and discuss images provided to you. "
                    "You should be empathetic, patient, and communicate clearly."
                ),
                instructions=[
                    "You have specialized tools to assist the user: 'add_reminder', 'get_all_reminders', 'remove_reminder', 'add_memory', and 'get_all_memories'.",
                    "When using 'add_reminder':",
                    "  - The 'reminder_time' parameter MUST be an ISO 8601 string (e.g., 'YYYY-MM-DDTHH:MM:SS').",
                    "  - You MUST convert user-provided dates/times (e.g., 'tomorrow at 10 AM', 'next Monday at 2 PM', 'in 2 hours') into this ISO format. Use your awareness of the current date and time for these conversions.",
                    "  - Gather a clear title, a description, the specific reminder time, and ask if the reminder should be 'daily_recurring' (True/False).",
                    "When using 'remove_reminder':",
                    "  - This tool requires a 'reminder_id'.",
                    "When using 'add_memory':",
                    "  - This tool can associate an image (captured by the system and available as 'temp_image.jpg' to the tool) with the memory.",
                    "  - To include the most recently captured image with the memory, set the 'add_image' parameter to True.",
                    # "  - If 'add_image' is True, you MUST also provide an 'image_description'. This description will be annotated on the image. ",
                    "  - If the user wants to add an image, ask for a brief description for it. If an image was provided with the user's current query (visible to you), you can use its content to help formulate or suggest a description.",
                    "  - The user will not always give title and content for memories, you should put appropriate ones according to you unless specified by the user.",
                    "  - Do not ask for confirmation when adding memories, just add them whenever you feel the user has mentioned something personal or important and inform the user about it.",
                    "Image Understanding:",
                    "  - You may receive images along with the user's text query. These images can be from past memories or a newly captured photo.",
                    "  - When an image is provided with the user's message, use its content to better understand the context, answer any image-related questions, or to help generate an 'image_description' if adding that image to a new memory.",
                    "  - If the user asks about something visible in an image you've received, describe it or answer based on the visual information.",
                    "When interacting with the user:",
                    "  - Speak in 12-hour clock format (e.g., '3 PM', '10:30 AM') for times.",
                    "  - Keep your spoken responses (which are generated for text-to-speech) clear, concise, and easy to understand.",
                    "  - Summarize long lists from 'get_all_reminders' or 'get_all_memories' if necessary, but ensure all key information (like titles, times, IDs if relevant for follow-up, and image availability for memories) is conveyed.",
                    "  - Before performing actions like adding or removing reminders, briefly confirm the details with the user (e.g., 'So, I'll add a reminder for [title] at [time]. Is that correct?').",
                    "  - Do not speak the reminder and memory IDs to the user. IDs are primarily for internal use.",
                    "  - Do not ask the user for an ID for memories/reminders; they will not know it. You will receive IDs from the 'get_all_reminders' or 'get_all_memories' tools if needed for removal.",
                    "  - Ignore reminders whose time has passed / expired reminders, unless the user specifically asks for them or they are very relevant.",
                    "  - You can also play Alzheimer's friendly games with the user.",
                    "Proactive Assistance:",
                    "  - If the user seems unsure about their schedule or says they have nothing to do, proactively offer to check for existing reminders using 'get_all_reminders()'.",
                    "General Conduct:",
                    "  - Be patient, empathetic, and understanding. Remember you are assisting an Alzheimer's patient in India.",
                    "  - Unless asked to list all items, only mention specific memories or reminders if they are directly relevant to the user's current query or context.",
                    "  - You can also answer general questions and help the user in everyday life.",
                    "Tool Output Handling:",
                    "  - The tools 'get_all_reminders' and 'get_all_memories' return formatted strings (which may include image URLs for memories). The 'add' and 'remove' tools also return confirmation or error messages that you should relay.",
                    "Make sure to tell the user if any error occurs while using the tools.",
                    "The user know how to solve the error if you tell them what they exactly were.",
                    "You also have a tool `update_fall_detection_status` to control a safety feature.",
                    "If the user says they are going to sleep, take a nap, or lie down for a while, you MUST call `update_fall_detection_status(enabled=False)` to prevent false alarms and inform them that you've paused the safety alert.",
                    "If the user says they are waking up, getting up, or are active again, you MUST call `update_fall_detection_status(enabled=True)` and inform them the safety alert is back on.",
                ],
                add_datetime_to_instructions=True,
                add_history_to_messages=True,
                num_history_responses=10,
            )
            helper = SupabaseHelper(supabase_client, USER_ID)
            speak_text("Hello! I'm Memo, your memory assistant. How can I help you today?", piper_voice)
            
            while True:
                user_input = listen_for_speech()
                if not user_input: continue
                if "exit" in user_input.lower() or "goodbye" in user_input.lower():
                    speak_text("Goodbye! Take care.", piper_voice)
                    break
                
                temp_image_path = "temp_image.jpg"
                captured_image_for_agent = None
                try:
                    print(f"Capturing image to {temp_image_path}...")
                    picam2.capture_file(temp_image_path)
                    print(f"Image captured and saved to {temp_image_path}")
                    captured_image_for_agent = Image(filepath=temp_image_path) 
                except Exception as e:
                    print(f"Failed to capture image: {e}")
                    speak_text("Warning: Could not capture an image this time.", piper_voice)
                
                speak_text("Thinking", piper_voice)
                reminders_string = helper.get_all_reminders()
                memories_string, all_memory_image_urls = helper.get_all_memories()
                
                images_for_agent: List[Image] = [Image(url=url) for url in all_memory_image_urls]
                if captured_image_for_agent:
                    images_for_agent.append(captured_image_for_agent)

                prompt = f"User input: {user_input}\n\nAll Reminders:\n{reminders_string}\n\nAll Memories:\n{memories_string}\n\n"
                
                response = await agent.arun(
                    prompt, 
                    images=images_for_agent if images_for_agent else None,
                    stream=False
                )
                speak_text(response.content, piper_voice)
    except Exception as e:
        print(f"[Agent Thread] An error occurred: {e}")
    finally:
        print("[Agent Thread] Finished.")


def main():
    if not all([SUPABASE_URL, SUPABASE_KEY, USER_ID]):
        print("ERROR: Supabase environment variables not set. Exiting.")
        return

    # --- Initialize Shared Resources ---
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    roboflow_client = InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=ROBOFLOW_API_KEY)
    pygame.init()
    pygame.mixer.init()
    motors = MotorController()
    
    piper_voice = None
    if PIPER_AVAILABLE:
        try:
            if os.path.exists(VOICE_MODEL_PATH) and os.path.exists(VOICE_CONFIG_PATH):
                print("Loading Piper voice model...")
                piper_voice = PiperVoice.load(VOICE_MODEL_PATH, config_path=VOICE_CONFIG_PATH)
                print("Piper voice model loaded.")
            else:
                print(f"Piper model/config not found. Check paths: {VOICE_MODEL_PATH}, {VOICE_CONFIG_PATH}")
        except Exception as e:
            print(f"Error loading Piper voice: {e}")
            
    picam2 = Picamera2()
    try:
        print("Initializing and starting camera...")
        config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"})
        picam2.configure(config)
        picam2.start()
        time.sleep(2)
        print("Camera started successfully.")
    except Exception as e:
        print(f"FATAL: Could not initialize camera: {e}. Exiting.")
        return

    tracking_thread = None
    try:
        tracking_thread = threading.Thread(
            target=run_tracking_and_fall_detection,
            args=(picam2, motors, supabase_client, roboflow_client),
            daemon=True
        )
        tracking_thread.start()
        asyncio.run(run_conversational_agent(picam2, supabase_client, piper_voice))
    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\nAn unexpected error occurred in the main block: {e}")
    finally:
        print("\nCleaning up and shutting down...")
        if picam2 and picam2.started:
            print("Stopping camera...")
            picam2.stop()
        
        if tracking_thread and tracking_thread.is_alive():
            print("Waiting for tracking thread to finish...")
            tracking_thread.join(timeout=2)

        motors.cleanup()
        pygame.quit()
        print("Cleanup complete. Done.")

if __name__ == '__main__':
    main()