import argparse
import datetime as dt
import json
import platform
from pathlib import Path
import re
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from urllib import error as url_error
from urllib import request as url_request
from urllib.parse import quote_plus

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    import pyaudio  # type: ignore  # noqa: F401

    HAS_PYAUDIO = True
except Exception:
    HAS_PYAUDIO = False

try:
    import pyautogui
except Exception:
    pyautogui = None

try:
    from PIL import ImageGrab
except Exception:
    ImageGrab = None

try:
    import pytesseract
except Exception:
    pytesseract = None


class LocalLLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: int = 60,
        max_tokens: int = 96,
        keep_alive: str = "30m",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.keep_alive = keep_alive

    def _fetch_json(self, endpoint: str) -> dict | None:
        req = url_request.Request(f"{self.base_url}{endpoint}", method="GET")
        try:
            with url_request.urlopen(req, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (url_error.URLError, TimeoutError, ValueError):
            return None

    def _post_json(self, endpoint: str, body: dict) -> dict | None:
        req = url_request.Request(
            f"{self.base_url}{endpoint}",
            method="POST",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with url_request.urlopen(req, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (url_error.URLError, TimeoutError, ValueError):
            return None

    def service_is_available(self) -> bool:
        return self._fetch_json("/api/tags") is not None

    def model_is_available(self) -> bool:
        tags_payload = self._fetch_json("/api/tags")
        if not tags_payload:
            return False

        for model_info in tags_payload.get("models", []):
            name = model_info.get("name", "")
            if name == self.model:
                return True
        return False

    def generate(self, user_message: str, memory: dict) -> str | None:
        goal = memory.get("user_goal") or "not set"
        task = memory.get("ongoing_task") or "not set"
        notes = memory.get("notes") or []
        recent_commands = memory.get("previous_commands") or []

        system_prompt = (
            "You are JARVIS, an advanced AI personal assistant running locally on the user's machine. "
            "Be calm, intelligent, concise, polite, slightly witty, and practical. "
            "Do not claim actions were executed if they were not. "
            "Keep answers useful and conversational, with technical confidence."
        )

        context_block = (
            f"Known user goal: {goal}\n"
            f"Known active task: {task}\n"
            f"Saved notes: {notes}\n"
            f"Recent commands: {recent_commands}\n"
        )

        prompt = (
            f"{system_prompt}\n\n"
            f"Conversation context:\n{context_block}\n"
            f"User message: {user_message}\n"
            "Assistant response:"
        )

        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": self.max_tokens,
            },
            "keep_alive": self.keep_alive,
        }

        payload = self._post_json("/api/generate", body)
        if not payload:
            return None
        generated = payload.get("response", "").strip()
        return generated or None

    def chat(self, messages: list[dict]) -> str | None:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.45,
                "num_predict": self.max_tokens,
            },
            "keep_alive": self.keep_alive,
        }
        payload = self._post_json("/api/chat", body)
        if not payload:
            return None

        message = payload.get("message", {})
        content = message.get("content", "").strip()
        return content or None

    def prewarm(self) -> None:
        body = {
            "model": self.model,
            "prompt": "Reply with: Ready.",
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 8,
            },
            "keep_alive": self.keep_alive,
        }

        req = url_request.Request(
            f"{self.base_url}/api/generate",
            method="POST",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with url_request.urlopen(req, timeout=min(self.timeout_seconds, 45)):
                pass
        except (url_error.URLError, TimeoutError, ValueError):
            pass


class JarvisAssistant:
    def __init__(
        self,
        use_voice: bool = True,
        use_microphone: bool | None = None,
        wake_word: str = "jarvis",
        llm_client: LocalLLMClient | None = None,
        mic_index: int | None = None,
        listen_timeout: float = 3.0,
        phrase_time_limit: float = 6.0,
        mic_calibration_seconds: float = 0.2,
        memory_file: str = ".jarvis_memory.json",
        max_history_turns: int = 12,
    ) -> None:
        self.use_voice = use_voice
        self.use_microphone = use_voice if use_microphone is None else use_microphone
        self.wake_word = wake_word.lower().strip()
        self.tts_engine = None
        self.voice_warning_printed = False
        self.llm_client = llm_client
        self.mic_index = mic_index
        self.listen_timeout = max(listen_timeout, 1.0)
        self.phrase_time_limit = max(phrase_time_limit, 1.0)
        self.mic_calibration_seconds = max(mic_calibration_seconds, 0.0)
        self.memory_file = Path(memory_file)
        self.max_history_turns = max(max_history_turns, 6)
        self.recognizer = sr.Recognizer() if self.use_microphone and sr is not None else None
        self.microphone = None
        self.microphone_calibrated = False
        self.memory = {
            "user_goal": None,
            "ongoing_task": None,
            "notes": [],
            "previous_commands": [],
            "reminders": [],
            "actuators": {},
        }
        self.profile = {
            "name": None,
            "preferences": [],
            "persona_style": "calm, confident, helpful",
        }
        self.conversation_history: list[dict] = []
        self.speech_queue: queue.Queue[str] = queue.Queue()
        self.tts_worker: threading.Thread | None = None
        self.tts_stop_event = threading.Event()
        self.tts_speaking_event = threading.Event()
        self.reminder_worker: threading.Thread | None = None
        self.reminder_stop_event = threading.Event()
        # Default off to avoid the mic interrupting Jarvis with room/speaker audio.
        self.barge_in_enabled = False

        if self.recognizer is not None:
            try:
                self.microphone = sr.Microphone(device_index=self.mic_index)
            except Exception:
                self.recognizer = None
                self.microphone = None
                self.use_microphone = False

        if self.use_voice and pyttsx3 is not None:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty("rate", 185)
                self.tts_engine.setProperty("volume", 1.0)
            except Exception:
                self.tts_engine = None

        if self.tts_engine is not None:
            self.start_tts_worker()

        self.load_persistent_memory()
        self.start_reminder_worker()

    def _restore_tts_engine(self) -> bool:
        if not self.use_voice or pyttsx3 is None:
            return False

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 185)
            engine.setProperty("volume", 1.0)
            self.tts_engine = engine
            self.start_tts_worker()
            return True
        except Exception:
            self.tts_engine = None
            return False

    def _notify_voice_unavailable_once(self) -> None:
        if self.use_voice and not self.voice_warning_printed:
            print("Jarvis: Voice output is unavailable. Run with --text-only or fix Windows TTS settings.")
            self.voice_warning_printed = True

    def load_persistent_memory(self) -> None:
        if not self.memory_file.exists():
            return

        try:
            payload = json.loads(self.memory_file.read_text(encoding="utf-8"))
        except Exception:
            return

        saved_memory = payload.get("memory", {}) if isinstance(payload, dict) else {}
        saved_profile = payload.get("profile", {}) if isinstance(payload, dict) else {}
        saved_history = payload.get("conversation_history", []) if isinstance(payload, dict) else []

        if isinstance(saved_memory, dict):
            self.memory["user_goal"] = saved_memory.get("user_goal")
            self.memory["ongoing_task"] = saved_memory.get("ongoing_task")
            self.memory["notes"] = list(saved_memory.get("notes", []))[-8:]
            loaded_reminders = list(saved_memory.get("reminders", []))
            valid_reminders = []
            for item in loaded_reminders:
                if not isinstance(item, dict):
                    continue
                if not item.get("id") or not item.get("title") or not item.get("due_at"):
                    continue
                if item.get("status") not in {"pending", "done"}:
                    item["status"] = "pending"
                valid_reminders.append(item)
            self.memory["reminders"] = valid_reminders[-40:]

            loaded_actuators = saved_memory.get("actuators", {})
            valid_actuators: dict[str, dict] = {}
            if isinstance(loaded_actuators, dict):
                for raw_name, payload in loaded_actuators.items():
                    if not isinstance(raw_name, str):
                        continue
                    name = self.normalize_actuator_name(raw_name)
                    if not name:
                        continue

                    state = "off"
                    if isinstance(payload, dict):
                        raw_state = str(payload.get("state", "off")).lower()
                        state = "on" if raw_state == "on" else "off"

                    valid_actuators[name] = {
                        "state": state,
                        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
                    }

            self.memory["actuators"] = valid_actuators

        if isinstance(saved_profile, dict):
            self.profile["name"] = saved_profile.get("name")
            self.profile["preferences"] = list(saved_profile.get("preferences", []))[-10:]
            self.profile["persona_style"] = saved_profile.get("persona_style", self.profile["persona_style"])

        if isinstance(saved_history, list):
            filtered_history = []
            for item in saved_history:
                if not isinstance(item, dict):
                    continue
                role = item.get("role")
                content = item.get("content")
                if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                    filtered_history.append({"role": role, "content": content.strip()})
            self.conversation_history = filtered_history[-(self.max_history_turns * 2) :]

    def save_persistent_memory(self) -> None:
        payload = {
            "memory": {
                "user_goal": self.memory.get("user_goal"),
                "ongoing_task": self.memory.get("ongoing_task"),
                "notes": self.memory.get("notes", [])[-8:],
                "reminders": self.memory.get("reminders", [])[-40:],
                "actuators": self.memory.get("actuators", {}),
            },
            "profile": {
                "name": self.profile.get("name"),
                "preferences": self.profile.get("preferences", [])[-10:],
                "persona_style": self.profile.get("persona_style", "calm, confident, helpful"),
            },
            "conversation_history": self.conversation_history[-(self.max_history_turns * 2) :],
        }

        try:
            self.memory_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def append_history(self, role: str, content: str) -> None:
        cleaned = content.strip()
        if not cleaned:
            return
        self.conversation_history.append({"role": role, "content": cleaned})
        self.conversation_history = self.conversation_history[-(self.max_history_turns * 2) :]

    def display_name(self) -> str:
        name = self.profile.get("name")
        return str(name).strip() if name else "there"

    def prefers_concise_responses(self) -> bool:
        prefs = self.profile.get("preferences", [])
        prefs_text = " ".join(str(item).lower() for item in prefs)
        return any(word in prefs_text for word in ["short", "concise", "brief"])

    def format_llm_response(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return cleaned

        if not self.prefers_concise_responses():
            return cleaned

        # Strip markdown-like list formatting and compress to a few direct sentences.
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        normalized = " ".join(line.lstrip("-*0123456789. ") for line in lines)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", normalized) if s.strip()]
        concise = " ".join(sentences[:3]).strip()
        if not concise:
            concise = normalized[:320].strip()
        return concise

    def start_tts_worker(self) -> None:
        if self.tts_engine is None:
            return
        if self.tts_worker is not None and self.tts_worker.is_alive():
            return

        self.tts_stop_event.clear()
        self.tts_worker = threading.Thread(target=self._tts_worker_loop, daemon=True)
        self.tts_worker.start()

    def _tts_worker_loop(self) -> None:
        while not self.tts_stop_event.is_set():
            try:
                text = self.speech_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if text is None:
                break

            if self.tts_engine is None and not self._restore_tts_engine():
                if self.use_voice and self._speak_with_windows_fallback(text):
                    continue
                self._notify_voice_unavailable_once()
                continue

            self.tts_speaking_event.set()
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception:
                self.tts_engine = None
                self._restore_tts_engine()
            finally:
                self.tts_speaking_event.clear()

    def is_speaking(self) -> bool:
        return self.tts_speaking_event.is_set()

    def interrupt_speech(self) -> None:
        if self.tts_engine is not None:
            try:
                self.tts_engine.stop()
            except Exception:
                pass

        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break

        self.tts_speaking_event.clear()

    def shutdown(self) -> None:
        self.interrupt_speech()
        self.tts_stop_event.set()
        if self.tts_worker is not None and self.tts_worker.is_alive():
            self.speech_queue.put_nowait(None)
        self.reminder_stop_event.set()

    def _speak_with_windows_fallback(self, text: str) -> bool:
        escaped = text.replace("'", "''")
        ps_script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Speak('{escaped}')"
        )
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return False

    def speak(self, text: str) -> None:
        print(f"Jarvis: {text}")
        if not self.use_voice:
            return

        worker_alive = self.tts_worker is not None and self.tts_worker.is_alive()
        if worker_alive:
            if self.tts_engine is not None or self._restore_tts_engine():
                self.speech_queue.put_nowait(text)
                return

        if self.tts_engine is not None:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                return
            except Exception:
                self.tts_engine = None
                if self._restore_tts_engine():
                    worker_alive = self.tts_worker is not None and self.tts_worker.is_alive()
                    if worker_alive:
                        self.speech_queue.put_nowait(text)
                        return

        if self._speak_with_windows_fallback(text):
            return

        self._notify_voice_unavailable_once()

    def emit_action(self, action_call: str, explanation: str) -> None:
        print("ACTION:")
        print(action_call)
        print("\nEXPLANATION:")
        print(explanation)

    def llm_status(self) -> tuple[bool, str]:
        if self.llm_client is None:
            return False, "Local LLM is disabled."
        if not self.llm_client.service_is_available():
            return False, f"Ollama service is not reachable on {self.llm_client.base_url}."
        if not self.llm_client.model_is_available():
            return False, f'Ollama is running, but model "{self.llm_client.model}" is not installed.'
        return True, f'Local LLM ready with model "{self.llm_client.model}".'

    def build_llm_messages(self, user_message: str) -> list[dict]:
        prefs = self.profile.get("preferences", [])[-5:]
        notes = self.memory.get("notes", [])[-5:]
        goal = self.memory.get("user_goal") or "Not set"
        task = self.memory.get("ongoing_task") or "Not set"
        prefs_text = " ".join(str(item).lower() for item in prefs)
        concise_mode = any(word in prefs_text for word in ["short", "concise", "brief"])

        style_hint = (
            "User prefers short practical answers. Keep responses to 2-4 concise sentences unless they ask for detail."
            if concise_mode
            else "Default to concise answers and expand only when asked."
        )

        system_prompt = (
            "You are JARVIS, an advanced local personal AI companion. "
            "Speak naturally like a trusted human assistant: calm, intelligent, concise, and proactive. "
            "Avoid robotic wording. Keep responses practical and personable. "
            "Ask clarifying questions if a request is ambiguous. "
            "Offer a smart next action when helpful. "
            "Never claim you executed actions unless the user confirms execution happened in their environment. "
            "When giving advice, prefer concrete steps over abstract statements. "
            "Avoid markdown bullet lists unless the user explicitly asks for a list. "
            f"{style_hint}"
        )

        profile_context = (
            f"User name: {self.display_name()}\n"
            f"Current goal: {goal}\n"
            f"Current task: {task}\n"
            f"Known preferences: {prefs if prefs else 'None yet'}\n"
            f"Saved notes: {notes if notes else 'None yet'}\n"
        )

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Conversation memory:\n{profile_context}"},
        ]
        messages.extend(self.conversation_history[-(self.max_history_turns * 2) :])
        messages.append({"role": "user", "content": user_message})
        return messages

    def reply_with_llm(self, command: str) -> bool:
        if self.llm_client is None:
            return False

        print("Jarvis: Thinking...")
        messages = self.build_llm_messages(command)
        response = self.llm_client.chat(messages)
        if not response:
            return False

        response = self.format_llm_response(response)

        self.append_history("user", command)
        self.append_history("assistant", response)
        self.save_persistent_memory()
        self.speak(response)
        return True

    def remember_command(self, command: str) -> None:
        self.memory["previous_commands"].append(command)
        self.memory["previous_commands"] = self.memory["previous_commands"][-8:]
        self.save_persistent_memory()

    def update_context_from_command(self, command: str) -> None:
        goal_markers = ["my goal is ", "i am trying to ", "i'm trying to ", "i want to "]
        task_markers = ["i am working on ", "i'm working on "]
        name_markers = ["my name is ", "call me "]
        preference_markers = ["i prefer ", "my preference is "]

        for marker in name_markers:
            if command.startswith(marker):
                value = command.replace(marker, "", 1).strip(" .")
                if value:
                    self.profile["name"] = value.title()
                    self.save_persistent_memory()
                return

        for marker in preference_markers:
            if command.startswith(marker):
                value = command.replace(marker, "", 1).strip(" .")
                if value:
                    prefs = self.profile.get("preferences", [])
                    prefs.append(value)
                    self.profile["preferences"] = prefs[-10:]
                    self.save_persistent_memory()
                return

        for marker in goal_markers:
            if marker in command:
                value = command.split(marker, 1)[1].strip(" .")
                if value:
                    self.memory["user_goal"] = value
                    self.save_persistent_memory()
                return

        for marker in task_markers:
            if marker in command:
                value = command.split(marker, 1)[1].strip(" .")
                if value:
                    self.memory["ongoing_task"] = value
                    self.save_persistent_memory()
                return

        if command.startswith("remember that "):
            note = command.replace("remember that ", "", 1).strip(" .")
            if note:
                self.memory["notes"].append(note)
                self.memory["notes"] = self.memory["notes"][-8:]
                self.save_persistent_memory()

    def build_plan(self, objective: str) -> list[str]:
        cleaned = objective.strip()
        return [
            f"Define a clear success target for {cleaned}.",
            "Break the work into small milestones with a realistic order.",
            "Execute the first milestone and verify results quickly.",
            "Review progress, adjust, and continue to completion.",
        ]

    def normalize_actuator_name(self, name: str) -> str:
        cleaned = re.sub(r"[^a-z0-9_\-\s]", "", name.lower()).strip()
        cleaned = re.sub(r"\s+", "_", cleaned)
        return cleaned[:40]

    def humanize_actuator_name(self, name: str) -> str:
        return name.replace("_", " ")

    def actuator_store(self) -> dict:
        store = self.memory.get("actuators")
        if not isinstance(store, dict):
            store = {}
            self.memory["actuators"] = store
        return store

    def register_actuator(self, raw_name: str) -> tuple[bool, str, str | None]:
        name = self.normalize_actuator_name(raw_name)
        if not name:
            return False, "Please provide a valid actuator name.", None

        actuators = self.actuator_store()
        if name in actuators:
            return True, f"Actuator {self.humanize_actuator_name(name)} is already registered.", name

        actuators[name] = {
            "state": "off",
            "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
        self.save_persistent_memory()
        return True, f"Actuator {self.humanize_actuator_name(name)} registered and set to off.", name

    def set_actuator_state(self, raw_name: str, state: str) -> tuple[bool, str, str | None]:
        name = self.normalize_actuator_name(raw_name)
        desired = "on" if state.lower() == "on" else "off"
        if not name:
            return False, "Please provide a valid actuator name.", None

        actuators = self.actuator_store()
        if name not in actuators:
            actuators[name] = {
                "state": "off",
                "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
            }

        actuators[name]["state"] = desired
        actuators[name]["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
        self.save_persistent_memory()
        return True, f"Actuator {self.humanize_actuator_name(name)} is now {desired}.", name

    def toggle_actuator(self, raw_name: str) -> tuple[bool, str, str | None]:
        name = self.normalize_actuator_name(raw_name)
        if not name:
            return False, "Please provide a valid actuator name.", None

        actuators = self.actuator_store()
        current = "off"
        if name in actuators:
            current = "on" if actuators[name].get("state") == "on" else "off"

        next_state = "off" if current == "on" else "on"
        return self.set_actuator_state(name, next_state)

    def list_actuator_statuses(self) -> list[str]:
        actuators = self.actuator_store()
        if not actuators:
            return []

        lines = []
        for name in sorted(actuators.keys()):
            state = "on" if actuators.get(name, {}).get("state") == "on" else "off"
            lines.append(f"{self.humanize_actuator_name(name)}: {state}")
        return lines

    def get_actuator_status(self, raw_name: str) -> tuple[bool, str, str | None]:
        name = self.normalize_actuator_name(raw_name)
        if not name:
            return False, "Please provide a valid actuator name.", None

        actuators = self.actuator_store()
        if name not in actuators:
            return False, f"Actuator {self.humanize_actuator_name(name)} is not registered.", name

        state = "on" if actuators[name].get("state") == "on" else "off"
        return True, f"Actuator {self.humanize_actuator_name(name)} is {state}.", name

    def write_text_to_active_window(
        self,
        text: str,
        delay_seconds: float = 1.0,
        typing_interval: float = 0.01,
    ) -> tuple[bool, str]:
        if pyautogui is None:
            return False, "Desktop typing control is unavailable. Install pyautogui."

        if not text.strip():
            return False, "I need some text to write."

        try:
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            pyautogui.write(text, interval=max(typing_interval, 0.0))
            return True, "Done. I typed the requested text."
        except Exception as ex:
            return False, f"I could not type text: {ex}"

    def press_keyboard_key(self, key_name: str) -> tuple[bool, str]:
        if pyautogui is None:
            return False, "Keyboard control is unavailable. Install pyautogui."

        key = key_name.strip().lower().replace(" ", "")
        alias = {
            "return": "enter",
            "esc": "escape",
            "spacebar": "space",
            "pgup": "pageup",
            "pgdn": "pagedown",
        }
        key = alias.get(key, key)
        if not key:
            return False, "Please provide a key to press."

        try:
            pyautogui.press(key)
            return True, f"Pressed {key}."
        except Exception as ex:
            return False, f"I could not press that key: {ex}"

    def trigger_hotkey(self, keys: list[str]) -> tuple[bool, str]:
        if pyautogui is None:
            return False, "Hotkey control is unavailable. Install pyautogui."

        normalized = [key.strip().lower().replace(" ", "") for key in keys if key.strip()]
        if len(normalized) < 2:
            return False, "Provide at least two keys, for example: hotkey ctrl+s"

        try:
            pyautogui.hotkey(*normalized)
            return True, f"Sent hotkey {'+'.join(normalized)}."
        except Exception as ex:
            return False, f"I could not trigger that hotkey: {ex}"

    def capture_screen(self) -> tuple[bool, str, Path | None]:
        if ImageGrab is None:
            return False, "Screen capture is unavailable. Install Pillow.", None

        captures_dir = Path("captures")
        captures_dir.mkdir(parents=True, exist_ok=True)
        file_path = captures_dir / f"screen_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        try:
            try:
                image = ImageGrab.grab(all_screens=True)
            except TypeError:
                image = ImageGrab.grab()
            image.save(file_path)
            resolved = file_path.resolve()
            return True, f"Captured your screen to {resolved}.", resolved
        except Exception as ex:
            return False, f"I could not capture the screen: {ex}", None

    def read_screen_text(self) -> tuple[bool, str]:
        if ImageGrab is None:
            return False, "Screen reading is unavailable. Install Pillow first."
        if pytesseract is None:
            return False, "Screen OCR needs pytesseract plus the Tesseract OCR app installed on Windows."
        if shutil.which("tesseract") is None:
            return False, "Screen OCR needs Tesseract installed and available in PATH."

        try:
            try:
                image = ImageGrab.grab(all_screens=True)
            except TypeError:
                image = ImageGrab.grab()
            raw_text = pytesseract.image_to_string(image)
        except Exception as ex:
            return False, f"I could not read screen text: {ex}"

        text = raw_text.strip()
        if not text:
            return True, "I checked the screen but did not detect readable text."

        print("SCREEN TEXT:")
        print(text)
        preview = " ".join(text.split())
        if len(preview) > 220:
            preview = preview[:220].rstrip() + "..."
        return True, f"I read this on screen: {preview}"

    def parse_datetime_iso(self, value: str | None) -> dt.datetime | None:
        if not value:
            return None
        try:
            return dt.datetime.fromisoformat(value)
        except Exception:
            return None

    def format_due_time(self, due_at: dt.datetime) -> str:
        now = dt.datetime.now()
        if due_at.date() == now.date():
            return due_at.strftime("today at %I:%M %p")
        if due_at.date() == (now.date() + dt.timedelta(days=1)):
            return due_at.strftime("tomorrow at %I:%M %p")
        return due_at.strftime("%Y-%m-%d %I:%M %p")

    def parse_reminder_request(self, command: str) -> tuple[str, dt.datetime] | None:
        if not command.startswith("remind me to "):
            return None

        now = dt.datetime.now()
        payload = command.replace("remind me to ", "", 1).strip()
        title = payload
        due_at: dt.datetime | None = None

        relative_match = re.search(r"\s+in\s+(\d+)\s+(minute|minutes|hour|hours|day|days)\s*$", payload)
        if relative_match:
            amount = int(relative_match.group(1))
            unit = relative_match.group(2)
            if "minute" in unit:
                due_at = now + dt.timedelta(minutes=amount)
            elif "hour" in unit:
                due_at = now + dt.timedelta(hours=amount)
            else:
                due_at = now + dt.timedelta(days=amount)
            title = payload[: relative_match.start()].strip(" ,.")

        tomorrow_match = re.search(r"\s+tomorrow\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*$", payload)
        if tomorrow_match and due_at is None:
            hour = int(tomorrow_match.group(1))
            minute = int(tomorrow_match.group(2) or "0")
            meridiem = (tomorrow_match.group(3) or "").lower()
            if meridiem == "pm" and hour < 12:
                hour += 12
            if meridiem == "am" and hour == 12:
                hour = 0
            tomorrow = now + dt.timedelta(days=1)
            due_at = tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)
            title = payload[: tomorrow_match.start()].strip(" ,.")

        at_match = re.search(r"\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*$", payload)
        if at_match and due_at is None:
            hour = int(at_match.group(1))
            minute = int(at_match.group(2) or "0")
            meridiem = (at_match.group(3) or "").lower()
            if meridiem == "pm" and hour < 12:
                hour += 12
            if meridiem == "am" and hour == 12:
                hour = 0
            due_at = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if due_at <= now:
                due_at += dt.timedelta(days=1)
            title = payload[: at_match.start()].strip(" ,.")

        if not title:
            title = "Untitled reminder"

        if due_at is None:
            due_at = now + dt.timedelta(minutes=30)

        return title, due_at

    def add_reminder(self, title: str, due_at: dt.datetime) -> dict:
        reminder = {
            "id": str(uuid.uuid4())[:8],
            "title": title,
            "due_at": due_at.isoformat(timespec="seconds"),
            "status": "pending",
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "last_notified_at": None,
        }
        reminders = self.memory.get("reminders", [])
        reminders.append(reminder)
        self.memory["reminders"] = reminders[-40:]
        self.save_persistent_memory()
        return reminder

    def list_pending_reminders(self) -> list[dict]:
        reminders = self.memory.get("reminders", [])
        pending = []
        for item in reminders:
            if item.get("status") == "pending":
                pending.append(item)
        pending.sort(key=lambda x: x.get("due_at", ""))
        return pending

    def complete_reminder(self, reminder_id: str) -> bool:
        reminders = self.memory.get("reminders", [])
        for item in reminders:
            if item.get("id") == reminder_id and item.get("status") == "pending":
                item["status"] = "done"
                item["completed_at"] = dt.datetime.now().isoformat(timespec="seconds")
                self.save_persistent_memory()
                return True
        return False

    def delete_reminder(self, reminder_id: str) -> bool:
        reminders = self.memory.get("reminders", [])
        before = len(reminders)
        reminders = [item for item in reminders if item.get("id") != reminder_id]
        self.memory["reminders"] = reminders
        changed = len(reminders) != before
        if changed:
            self.save_persistent_memory()
        return changed

    def check_due_reminders(self) -> None:
        now = dt.datetime.now()
        due_reminders: list[dict] = []
        reminders = self.memory.get("reminders", [])

        for item in reminders:
            if item.get("status") != "pending":
                continue

            due_at = self.parse_datetime_iso(item.get("due_at"))
            if due_at is None or due_at > now:
                continue

            last_notified = self.parse_datetime_iso(item.get("last_notified_at"))
            if last_notified and (now - last_notified).total_seconds() < 300:
                continue

            item["last_notified_at"] = now.isoformat(timespec="seconds")
            due_reminders.append(item)

        if due_reminders:
            self.save_persistent_memory()

        for item in due_reminders:
            self.speak(f"Reminder due: {item.get('title')}. Say complete reminder {item.get('id')} when done.")

    def start_reminder_worker(self) -> None:
        if self.reminder_worker is not None and self.reminder_worker.is_alive():
            return
        self.reminder_stop_event.clear()
        self.reminder_worker = threading.Thread(target=self._reminder_worker_loop, daemon=True)
        self.reminder_worker.start()

    def _reminder_worker_loop(self) -> None:
        while not self.reminder_stop_event.is_set():
            self.check_due_reminders()
            for _ in range(10):
                if self.reminder_stop_event.is_set():
                    return
                time.sleep(1)

    def plan_actions_from_text(self, command: str) -> list[dict]:
        lowered = command.lower().strip()
        if not lowered:
            return []

        chunks = [part.strip() for part in re.split(r"\s+and then\s+|\s+then\s+", lowered) if part.strip()]
        if not chunks:
            chunks = [lowered]

        actions: list[dict] = []
        for chunk in chunks:
            if chunk.startswith("open "):
                target = chunk.replace("open ", "", 1).strip()
                if not target:
                    continue

                known_sites = {
                    "google": "https://www.google.com",
                    "youtube": "https://www.youtube.com",
                    "github": "https://github.com",
                }

                if target.startswith("http://") or target.startswith("https://"):
                    actions.append({"type": "open_website", "url": target})
                elif target in known_sites:
                    actions.append({"type": "open_website", "url": known_sites[target]})
                else:
                    actions.append({"type": "open_application", "app": target})
                continue

            if chunk.startswith("search "):
                query = chunk.replace("search ", "", 1).strip()
                if query:
                    actions.append({"type": "search_web", "query": query})
                continue

            if "list files" in chunk or "show files" in chunk:
                actions.append({"type": "list_files"})
                continue

            parsed_reminder = self.parse_reminder_request(chunk)
            if parsed_reminder is not None:
                title, due_at = parsed_reminder
                actions.append({"type": "add_reminder", "title": title, "due_at": due_at.isoformat()})

        return actions

    def execute_action_plan(self, actions: list[dict]) -> bool:
        if not actions:
            return False

        for action in actions:
            action_type = action.get("type")

            if action_type == "open_website":
                url = action.get("url", "")
                if not isinstance(url, str) or not url:
                    continue
                escaped = url.replace('"', '\\"')
                self.emit_action(f'open_website("{escaped}")', "Opening website from planned action.")
                self.open_website(url)
                self.speak(f"Opened {url}.")
                continue

            if action_type == "open_application":
                app = action.get("app", "")
                if not isinstance(app, str) or not app:
                    continue
                escaped = app.replace('"', '\\"')
                self.emit_action(f'open_application("{escaped}")', "Opening app from planned action.")
                if self.open_application(app):
                    self.speak(f"Opened {app}.")
                else:
                    self.speak(f"I could not open {app}.")
                continue

            if action_type == "search_web":
                query = action.get("query", "")
                if not isinstance(query, str) or not query:
                    continue
                escaped = query.replace('"', '\\"')
                self.emit_action(f'search_web("{escaped}")', "Searching web from planned action.")
                self.search_web(query)
                self.speak(f"Searched for {query}.")
                continue

            if action_type == "list_files":
                self.emit_action('list_files(".")', "Listing files from planned action.")
                files = self.list_files(".")
                if not files:
                    self.speak("I could not find files in this folder.")
                else:
                    preview = ", ".join(files[:8])
                    self.speak(f"I found {len(files)} items. First results: {preview}.")
                continue

            if action_type == "add_reminder":
                title = action.get("title", "")
                due_at_raw = action.get("due_at")
                due_at = self.parse_datetime_iso(str(due_at_raw) if due_at_raw is not None else None)
                if not isinstance(title, str) or not title or due_at is None:
                    continue
                reminder = self.add_reminder(title, due_at)
                self.emit_action(
                    f'add_reminder("{title.replace("\"", "\\\"")}", "{due_at.isoformat(timespec="seconds")}")',
                    "Creating reminder from planned action.",
                )
                self.speak(
                    f"Reminder set for {title}, due {self.format_due_time(due_at)}. ID {reminder.get('id')}."
                )

        return True

    def open_application(self, app_name: str) -> bool:
        app_map = {
            "notepad": ["notepad"],
            "calculator": ["calc"],
            "calc": ["calc"],
            "paint": ["mspaint"],
            "explorer": ["explorer"],
            "terminal": ["powershell"],
            "powershell": ["powershell"],
            "code": ["code"],
            "vscode": ["code"],
        }

        launch_cmd = app_map.get(app_name)
        if not launch_cmd:
            return False

        try:
            subprocess.Popen(launch_cmd)
            return True
        except Exception:
            return False

    def open_website(self, url: str) -> None:
        webbrowser.open(url)

    def search_web(self, query: str) -> None:
        webbrowser.open(f"https://www.google.com/search?q={quote_plus(query)}")

    def list_files(self, folder: str = ".") -> list[str]:
        base = Path(folder)
        if not base.exists() or not base.is_dir():
            return []
        return sorted(item.name for item in base.iterdir())

    def listen(self) -> str | None:
        if not self.use_microphone or sr is None or self.recognizer is None or self.microphone is None:
            return None

        try:
            with self.microphone as source:
                print("Listening...")
                if not self.microphone_calibrated and self.mic_calibration_seconds > 0:
                    self.recognizer.adjust_for_ambient_noise(source, duration=self.mic_calibration_seconds)
                    self.microphone_calibrated = True
                audio = self.recognizer.listen(
                    source,
                    timeout=self.listen_timeout,
                    phrase_time_limit=self.phrase_time_limit,
                )
            text = self.recognizer.recognize_google(audio)
            print(f"You: {text}")
            return text
        except sr.UnknownValueError:
            return None
        except sr.WaitTimeoutError:
            return None
        except Exception:
            return None

    def get_text_input(self) -> str:
        try:
            return input("You: ").strip()
        except EOFError:
            return "exit"

    def normalize_command(self, raw: str) -> str:
        cmd = raw.lower().strip()
        if cmd.startswith(self.wake_word):
            cmd = cmd[len(self.wake_word) :].strip(" ,")
        return cmd

    def strip_wake_word_preserve_case(self, raw: str) -> str:
        cmd = raw.strip()
        if cmd.lower().startswith(self.wake_word):
            cmd = cmd[len(self.wake_word) :].strip(" ,")
        return cmd

    def handle_command(self, command: str, raw_command: str | None = None) -> bool:
        if not command:
            return True

        raw_text_command = self.strip_wake_word_preserve_case(raw_command) if raw_command else command

        self.remember_command(command)
        self.update_context_from_command(command)

        if command in {"stop", "stop talking", "quiet", "be quiet", "pause"}:
            self.interrupt_speech()
            self.speak("Understood. I will stay quiet.")
            return True

        if command in {"enable barge in", "enable barge-in"}:
            self.barge_in_enabled = True
            self.speak("Barge-in is now enabled.")
            return True

        if command in {"disable barge in", "disable barge-in"}:
            self.barge_in_enabled = False
            self.speak("Barge-in is now disabled.")
            return True

        if self.is_speaking() and self.barge_in_enabled:
            self.interrupt_speech()

        if command in {"see screen", "see my screen", "capture screen", "screenshot", "take screenshot"}:
            ok, message, capture_path = self.capture_screen()
            if ok and capture_path is not None:
                escaped_path = str(capture_path).replace("\\", "\\\\").replace('"', '\\"')
                self.emit_action(
                    f'capture_screen("{escaped_path}")',
                    "Capturing current screen from the desktop.",
                )
            self.speak(message)
            return True

        if command in {"read screen", "see screen text", "read my screen", "what is on my screen"}:
            ok, message = self.read_screen_text()
            self.speak(message)
            return True

        if command.startswith(("write ", "type ")):
            match = re.match(r"^(?:write|type)\s+(.+)$", raw_text_command, flags=re.IGNORECASE)
            text_to_write = match.group(1).strip() if match else command.split(" ", 1)[1].strip()
            if not text_to_write:
                self.speak("Tell me what to write.")
                return True

            print("Jarvis: Focus your target app now. Typing in one second.")
            ok, message = self.write_text_to_active_window(text_to_write, delay_seconds=1.0)
            if ok:
                escaped_text = text_to_write.replace("\\", "\\\\").replace('"', '\\"')
                self.emit_action(
                    f'write_text("{escaped_text}")',
                    "Typing text into the active desktop window.",
                )
            self.speak(message)
            return True

        if command.startswith("press "):
            key_name = command.replace("press ", "", 1).strip()
            ok, message = self.press_keyboard_key(key_name)
            if ok:
                escaped_key = key_name.replace("\\", "\\\\").replace('"', '\\"')
                self.emit_action(
                    f'press_key("{escaped_key}")',
                    "Pressing keyboard key on the active window.",
                )
            self.speak(message)
            return True

        if command.startswith("hotkey "):
            combo = command.replace("hotkey ", "", 1).strip()
            keys = [part for part in re.split(r"\s*\+\s*", combo) if part]
            ok, message = self.trigger_hotkey(keys)
            if ok:
                escaped_combo = "+".join(keys).replace("\\", "\\\\").replace('"', '\\"')
                self.emit_action(
                    f'hotkey("{escaped_combo}")',
                    "Sending desktop hotkey combination.",
                )
            self.speak(message)
            return True

        if command.startswith(("my name is ", "call me ")):
            self.speak(f"Great to meet you, {self.display_name()}. I will remember that.")
            return True

        if command.startswith(("i prefer ", "my preference is ")):
            self.speak("Got it. I saved your preference.")
            return True

        if "what is my name" in command or "who am i" in command:
            if self.profile.get("name"):
                self.speak(f"You are {self.display_name()}.")
            else:
                self.speak("You have not told me your name yet. Say: my name is ...")
            return True

        if "show my preferences" in command or "what are my preferences" in command:
            prefs = self.profile.get("preferences", [])
            if not prefs:
                self.speak("You have no saved preferences yet.")
                return True
            self.speak("Here are your saved preferences.")
            for idx, pref in enumerate(prefs, start=1):
                print(f"{idx}. {pref}")
            return True

        if "forget my name" in command:
            self.profile["name"] = None
            self.save_persistent_memory()
            self.speak("Done. I forgot your name.")
            return True

        if command.startswith(("my goal is ", "i am trying to ", "i'm trying to ", "i want to ")):
            goal = self.memory.get("user_goal")
            if goal:
                self.speak(f"Understood. I will keep your goal in mind: {goal}.")
            return True

        if command.startswith(("i am working on ", "i'm working on ")):
            task = self.memory.get("ongoing_task")
            if task:
                self.speak(f"Perfect. I will track that task: {task}.")
            return True

        if command.startswith("remember that "):
            self.speak("Noted. I saved that.")
            return True

        if any(word in command for word in ["exit", "quit", "goodbye", "shutdown"]):
            self.save_persistent_memory()
            self.speak("Shutting down. Talk to you later.")
            return False

        if command in {"help", "commands", "what can you do"}:
            self.speak(
                "I can hold human-like conversations, remember your preferences, help plan goals, "
                "and run actions on your machine. Try: my name is Alex, I prefer concise answers, "
                "plan build my bot, remind me to stretch in 30 minutes, execute open google then search local ai, "
                "see screen, read screen, write Hello world, hotkey ctrl+s, "
                "register actuator pump, turn on pump, actuator status pump, "
                "llm status, and exit."
            )
            return True

        if command in {"actuators", "list actuators", "show actuators", "actuator status"}:
            statuses = self.list_actuator_statuses()
            if not statuses:
                self.speak("No actuators registered yet. Say: register actuator pump.")
                return True
            self.speak("Here are your actuator states.")
            for line in statuses:
                print(line)
            return True

        if command.startswith("register actuator "):
            actuator_name = command.replace("register actuator ", "", 1).strip()
            ok, message, canonical_name = self.register_actuator(actuator_name)
            if ok and canonical_name is not None:
                self.emit_action(
                    f'register_actuator("{canonical_name}")',
                    "Registering actuator in local control map.",
                )
            self.speak(message)
            return True

        actuator_action = re.match(r"^actuator\s+(.+?)\s+(on|off|toggle|status)$", command)
        if actuator_action:
            actuator_name = actuator_action.group(1).strip()
            action = actuator_action.group(2)
            if action == "status":
                ok, message, canonical_name = self.get_actuator_status(actuator_name)
                if ok and canonical_name is not None:
                    self.emit_action(
                        f'get_actuator_status("{canonical_name}")',
                        "Querying actuator state.",
                    )
                self.speak(message)
                return True

            if action == "toggle":
                ok, message, canonical_name = self.toggle_actuator(actuator_name)
                if ok and canonical_name is not None:
                    state = self.actuator_store().get(canonical_name, {}).get("state", "off")
                    self.emit_action(
                        f'set_actuator_state("{canonical_name}", "{state}")',
                        "Toggling actuator state.",
                    )
                self.speak(message)
                return True

            ok, message, canonical_name = self.set_actuator_state(actuator_name, action)
            if ok and canonical_name is not None:
                self.emit_action(
                    f'set_actuator_state("{canonical_name}", "{action}")',
                    "Setting actuator state.",
                )
            self.speak(message)
            return True

        turn_action = re.match(r"^turn\s+(on|off)\s+(.+)$", command)
        if turn_action:
            state = turn_action.group(1)
            actuator_name = turn_action.group(2).strip()
            ok, message, canonical_name = self.set_actuator_state(actuator_name, state)
            if ok and canonical_name is not None:
                self.emit_action(
                    f'set_actuator_state("{canonical_name}", "{state}")',
                    "Setting actuator state.",
                )
            self.speak(message)
            return True

        toggle_action = re.match(r"^toggle\s+(.+)$", command)
        if toggle_action:
            actuator_name = toggle_action.group(1).strip()
            ok, message, canonical_name = self.toggle_actuator(actuator_name)
            if ok and canonical_name is not None:
                state = self.actuator_store().get(canonical_name, {}).get("state", "off")
                self.emit_action(
                    f'set_actuator_state("{canonical_name}", "{state}")',
                    "Toggling actuator state.",
                )
            self.speak(message)
            return True

        if command.startswith(("execute ", "do this ", "do this:", "run plan ")):
            request = (
                command.replace("do this:", "", 1)
                .replace("do this ", "", 1)
                .replace("execute ", "", 1)
                .replace("run plan ", "", 1)
                .strip()
            )
            actions = self.plan_actions_from_text(request)
            if not actions:
                self.speak("I could not build an action plan from that request yet.")
                return True
            self.speak("On it. Executing your plan now.")
            self.execute_action_plan(actions)
            return True

        if " and then " in command or command.startswith("first "):
            actions = self.plan_actions_from_text(command)
            if len(actions) >= 2:
                self.speak("Understood. I will handle those steps now.")
                self.execute_action_plan(actions)
                return True

        if "llm status" in command or "model status" in command:
            _, status_text = self.llm_status()
            self.speak(status_text)
            return True

        if "what is my goal" in command or "what's my goal" in command:
            goal = self.memory.get("user_goal")
            if goal:
                self.speak(f"Your current goal is {goal}.")
            else:
                self.speak("I do not have a saved goal yet. Tell me by saying: my goal is ...")
            return True

        if "what am i working on" in command:
            task = self.memory.get("ongoing_task")
            if task:
                self.speak(f"You are working on {task}.")
            else:
                self.speak("You have not told me your active task yet.")
            return True

        if "show notes" in command or "my notes" in command:
            notes = self.memory.get("notes", [])
            if not notes:
                self.speak("No notes saved yet.")
                return True

            self.speak("Here are your recent notes.")
            for idx, note in enumerate(notes, start=1):
                print(f"{idx}. {note}")
            return True

        if "last commands" in command or "recent commands" in command:
            recent = self.memory.get("previous_commands", [])
            if not recent:
                self.speak("No recent commands found.")
                return True

            self.speak("Here are your recent commands.")
            for idx, item in enumerate(recent, start=1):
                print(f"{idx}. {item}")
            return True

        parsed_reminder = self.parse_reminder_request(command)
        if parsed_reminder is not None:
            title, due_at = parsed_reminder
            reminder = self.add_reminder(title, due_at)
            self.emit_action(
                f'add_reminder("{title.replace("\"", "\\\"")}", "{due_at.isoformat(timespec="seconds")}")',
                "Creating reminder.",
            )
            self.speak(
                f"Reminder set for {title}, due {self.format_due_time(due_at)}. "
                f"Reminder ID {reminder.get('id')}."
            )
            return True

        if command in {"show reminders", "list reminders", "my reminders"}:
            pending = self.list_pending_reminders()
            if not pending:
                self.speak("You have no pending reminders.")
                return True
            self.speak("Here are your pending reminders.")
            for item in pending[:15]:
                due_at = self.parse_datetime_iso(item.get("due_at"))
                due_text = self.format_due_time(due_at) if due_at else item.get("due_at")
                print(f"{item.get('id')}: {item.get('title')} -> {due_text}")
            return True

        if command.startswith("complete reminder "):
            reminder_id = command.replace("complete reminder ", "", 1).strip()
            if self.complete_reminder(reminder_id):
                self.speak(f"Reminder {reminder_id} marked complete.")
            else:
                self.speak("I could not find that pending reminder ID.")
            return True

        if command.startswith("delete reminder "):
            reminder_id = command.replace("delete reminder ", "", 1).strip()
            if self.delete_reminder(reminder_id):
                self.speak(f"Reminder {reminder_id} deleted.")
            else:
                self.speak("I could not find that reminder ID.")
            return True

        if "time" in command:
            now = dt.datetime.now().strftime("%I:%M %p")
            self.speak(f"It is {now}.")
            return True

        if "date" in command:
            today = dt.datetime.now().strftime("%A, %B %d, %Y")
            self.speak(f"Today is {today}.")
            return True

        if "open youtube" in command:
            self.emit_action('open_website("https://www.youtube.com")', "Opening YouTube for you.")
            self.open_website("https://www.youtube.com")
            self.speak("Opening YouTube for you.")
            return True

        if "open google" in command:
            self.emit_action('open_website("https://www.google.com")', "Opening Google for you.")
            self.open_website("https://www.google.com")
            self.speak("Opening Google for you.")
            return True

        if "open github" in command:
            self.emit_action('open_website("https://github.com")', "Opening GitHub for you.")
            self.open_website("https://github.com")
            self.speak("Opening GitHub for you.")
            return True

        if command.startswith("search "):
            query = command.replace("search ", "", 1).strip()
            if query:
                escaped_query = query.replace('"', '\\"')
                self.emit_action(f'search_web("{escaped_query}")', f"Searching the web for {query}.")
                self.search_web(query)
                self.speak(f"Searching the web for {query}.")
            else:
                self.speak("What should I search for?")
            return True

        if command == "plan" or command.startswith("plan ") or "help me plan" in command:
            if command == "plan":
                objective = ""
            elif command.startswith("plan "):
                objective = command.replace("plan ", "", 1).strip()
            else:
                objective = command.replace("help me plan", "", 1).strip()
            if not objective:
                objective = self.memory.get("user_goal") or self.memory.get("ongoing_task")
            if not objective:
                self.speak("Sure. What exactly do you want me to plan?")
                return True

            steps = self.build_plan(objective)
            self.speak(f"Here is a quick plan for {objective}.")
            for idx, step in enumerate(steps, start=1):
                print(f"{idx}. {step}")
            self.speak("If you want, I can help you execute step one right now.")
            return True

        if "list files" in command or "show files" in command:
            self.emit_action('list_files(".")', "Listing files in your current folder.")
            files = self.list_files(".")
            if not files:
                self.speak("I could not find files in this folder.")
                return True

            preview = ", ".join(files[:10])
            self.speak(f"I found {len(files)} items. First results: {preview}.")
            return True

        if command.startswith("open "):
            target = command.replace("open ", "", 1).strip()
            if not target:
                self.speak("What would you like me to open?")
                return True

            if target.startswith("http://") or target.startswith("https://"):
                escaped_target = target.replace('"', '\\"')
                self.emit_action(f'open_website("{escaped_target}")', "Opening that website for you.")
                self.open_website(target)
                self.speak("Opening that website for you.")
                return True

            if self.open_application(target):
                escaped_target = target.replace('"', '\\"')
                self.emit_action(f'open_application("{escaped_target}")', f"Opening {target} for you.")
                self.speak(f"Opening {target} for you.")
            else:
                self.speak(
                    "I cannot open that yet. Try notepad, calculator, paint, explorer, terminal, or code."
                )
            return True

        if "system status" in command or "system info" in command:
            info = f"{platform.system()} {platform.release()}, Python {platform.python_version()}"
            self.speak(info)
            return True

        if any(greet in command for greet in ["hello", "hi", "hey"]):
            current_task = self.memory.get("ongoing_task") or self.memory.get("user_goal")
            if current_task:
                self.speak(f"Hey {self.display_name()}. Ready to continue with {current_task}?")
            else:
                self.speak(f"Hey {self.display_name()}. What are we working on today?")
            return True

        if "how are you" in command:
            self.speak("Operating smoothly. Ready when you are.")
            return True

        if "who are you" in command:
            self.speak("I am Jarvis, your local digital operating companion.")
            return True

        if "thank" in command:
            self.speak("Always a pleasure.")
            return True

        if self.reply_with_llm(command):
            return True

        self.speak(
            "I can respond conversationally or run commands. "
            "If you want an action, ask me to open, search, list files, or plan a task. "
            "You can also ask for llm status."
        )
        return True


def list_microphones() -> int:
    if sr is None:
        print("SpeechRecognition is not installed.")
        return 1

    try:
        names = sr.Microphone.list_microphone_names()
    except Exception as ex:
        print(f"Microphone listing failed: {ex}")
        return 1

    if not names:
        print("No microphones detected.")
        return 1

    print("Available microphones:")
    for idx, name in enumerate(names):
        print(f"{idx}: {name}")
    return 0


def run_self_test(
    use_voice: bool,
    use_microphone: bool,
    mic_index: int | None,
    llm_client: LocalLLMClient | None = None,
) -> int:
    print("Running Jarvis self-test...")
    print(f"Python: {sys.version.split()[0]}")
    print(f"SpeechRecognition installed: {sr is not None}")
    print(f"PyAudio installed: {HAS_PYAUDIO}")
    print(f"pyttsx3 installed: {pyttsx3 is not None}")
    print(f"Voice output enabled: {use_voice}")
    print(f"Microphone input enabled: {use_microphone}")
    print(f"Microphone index: {mic_index if mic_index is not None else 'default'}")

    if use_microphone and sr is not None:
        try:
            names = sr.Microphone.list_microphone_names()
            print(f"Microphones detected: {len(names)}")
            if mic_index is not None and 0 <= mic_index < len(names):
                print(f"Selected microphone: {names[mic_index]}")
        except Exception as ex:
            print(f"Microphone check failed: {ex}")
            return 1

    if llm_client is None:
        print("Local LLM: disabled")
    else:
        print(f"Local LLM service reachable: {llm_client.service_is_available()}")
        print(f"Local LLM model available ({llm_client.model}): {llm_client.model_is_available()}")

    print("Self-test complete.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Jarvis-style assistant")
    parser.add_argument("--text-only", action="store_true", help="Disable microphone and voice output")
    parser.add_argument("--keyboard-input", action="store_true", help="Type commands while still allowing voice output")
    parser.add_argument("--list-mics", action="store_true", help="List available microphones and exit")
    parser.add_argument("--mic-index", type=int, default=None, help="Microphone device index to use")
    parser.add_argument("--self-test", action="store_true", help="Run environment checks and exit")
    parser.add_argument("--no-llm", action="store_true", help="Disable local Ollama conversation fallback")
    parser.add_argument("--model", default="llama3.2:1b", help="Ollama model name to use for local chat")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama server URL")
    parser.add_argument("--llm-timeout", type=int, default=60, help="Timeout in seconds for local LLM responses")
    parser.add_argument("--llm-max-tokens", type=int, default=96, help="Max tokens generated by local LLM")
    parser.add_argument("--memory-file", default=".jarvis_memory.json", help="Path to persistent Jarvis memory file")
    parser.add_argument("--history-turns", type=int, default=12, help="Number of recent conversation turns to retain")
    parser.add_argument("--listen-timeout", type=float, default=3.0, help="Seconds to wait for speech before retry")
    parser.add_argument("--phrase-time-limit", type=float, default=6.0, help="Max seconds for each spoken command")
    parser.add_argument(
        "--mic-calibration-seconds",
        type=float,
        default=0.2,
        help="One-time microphone ambient calibration duration",
    )
    args = parser.parse_args()

    use_voice = not args.text_only
    use_microphone = use_voice and not args.keyboard_input

    if use_microphone and sr is None:
        print("Microphone mode requires SpeechRecognition. Install dependencies from requirements.txt.")
        return 1

    if use_microphone and not HAS_PYAUDIO:
        print("Microphone mode requires PyAudio in the current Python environment.")
        print("Use the project Python 3.12 virtual environment for full voice conversation:")
        print("  .\\.venv\\Scripts\\python.exe jarvis.py")
        print("Or switch to --keyboard-input / --text-only if you cannot change environments.")
        return 1

    if args.list_mics:
        return list_microphones()

    llm_client = None
    if not args.no_llm:
        llm_client = LocalLLMClient(
            base_url=args.ollama_url,
            model=args.model,
            timeout_seconds=max(args.llm_timeout, 10),
            max_tokens=max(args.llm_max_tokens, 32),
        )

    if args.self_test:
        return run_self_test(
            use_voice=use_voice,
            use_microphone=use_microphone,
            mic_index=args.mic_index,
            llm_client=llm_client,
        )

    assistant = JarvisAssistant(
        use_voice=use_voice,
        use_microphone=use_microphone,
        llm_client=llm_client,
        mic_index=args.mic_index,
        memory_file=args.memory_file,
        max_history_turns=max(args.history_turns, 6),
        listen_timeout=args.listen_timeout,
        phrase_time_limit=args.phrase_time_limit,
        mic_calibration_seconds=args.mic_calibration_seconds,
    )
    assistant.speak(f"Jarvis online. Good to see you, {assistant.display_name()}. Say help for commands.")
    assistant.check_due_reminders()

    if use_voice and args.keyboard_input:
        assistant.speak("Keyboard input mode enabled. Type commands, and I will reply out loud.")
    elif use_voice and not assistant.use_microphone:
        assistant.speak("Microphone is unavailable. Switching to keyboard input mode.")

    if llm_client is not None:
        llm_ready, llm_message = assistant.llm_status()
        assistant.speak(llm_message)
        if not llm_ready:
            assistant.speak(f'If needed, run: ollama pull {args.model}')
        else:
            # Warm the model in the background to reduce first-response latency.
            threading.Thread(target=llm_client.prewarm, daemon=True).start()

    while True:
        if assistant.use_microphone:
            raw = assistant.listen()
            if not raw:
                continue
        else:
            raw = assistant.get_text_input()
        command = assistant.normalize_command(raw)
        keep_running = assistant.handle_command(command, raw_command=raw)
        if not keep_running:
            assistant.shutdown()
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
