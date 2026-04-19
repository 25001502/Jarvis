import argparse
import ast
import datetime as dt
import json
import math
import os
import platform
from pathlib import Path
import random
import re
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
        }
        self.profile = {
            "name": None,
            "preferences": [],
            "persona_style": "calm, confident, helpful",
        }
        self.conversation_history: list[dict] = []
        self.reminder_worker: threading.Thread | None = None
        self.reminder_stop_event = threading.Event()
        self.barge_in_enabled = True
        self.speech_lock = threading.RLock()
        self.tts_speaking_event = threading.Event()

        if self.recognizer is not None:
            try:
                self.microphone = sr.Microphone(device_index=self.mic_index)
            except Exception as ex:
                print(f"[MIC ERROR] Failed to open microphone: {ex}")
                self.recognizer = None
                self.microphone = None
                self.use_microphone = False

        if self.use_voice and pyttsx3 is not None:
            try:
                if platform.system().lower().startswith("win"):
                    self.tts_engine = pyttsx3.init("sapi5")
                else:
                    self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty("rate", 185)
                self.tts_engine.setProperty("volume", 1.0)
            except Exception as ex:
                print(f"[TTS INIT ERROR] {ex}")
                self.tts_engine = None

        self.load_persistent_memory()
        self.start_reminder_worker()

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

        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        normalized = " ".join(line.lstrip("-*0123456789. ") for line in lines)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", normalized) if s.strip()]
        concise = " ".join(sentences[:3]).strip()
        if not concise:
            concise = normalized[:320].strip()
        return concise

    def is_speaking(self) -> bool:
        return self.tts_speaking_event.is_set()

    def interrupt_speech(self) -> None:
        if self.tts_engine is not None:
            try:
                self.tts_engine.stop()
            except Exception as ex:
                print(f"[TTS STOP ERROR] {ex}")
        self.tts_speaking_event.clear()

    def shutdown(self) -> None:
        self.interrupt_speech()
        self.reminder_stop_event.set()

    def _speak_with_system_fallback(self, text: str) -> bool:
        escaped = text.replace("'", "''")
        system_name = platform.system().lower()

        if "windows" in system_name:
            ps_script = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.Speak('{escaped}')"
            )
            commands = [["powershell", "-NoProfile", "-Command", ps_script]]
        elif "darwin" in system_name:
            commands = [["say", text]]
        else:
            commands = [["spd-say", text], ["espeak", text]]

        for cmd in commands:
            try:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=20,
                    check=False,
                )
                if result.returncode == 0:
                    return True
            except Exception:
                continue
        return False

    def speak(self, text: str) -> None:
        text = text.strip()
        if not text:
            return

        print(f"Jarvis: {text}")

        if not self.use_voice:
            return

        if self.tts_engine is not None:
            try:
                with self.speech_lock:
                    self.tts_speaking_event.set()
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                    self.tts_speaking_event.clear()
                    return
            except Exception as ex:
                self.tts_speaking_event.clear()
                print(f"[TTS ERROR] pyttsx3 failed: {ex}")
                self.tts_engine = None

        try:
            self.tts_speaking_event.set()
            if self._speak_with_system_fallback(text):
                self.tts_speaking_event.clear()
                return
        except Exception as ex:
            print(f"[TTS FALLBACK ERROR] {ex}")
        finally:
            self.tts_speaking_event.clear()

        if not self.voice_warning_printed:
            print("Jarvis: Voice output is unavailable. Install/configure a local TTS engine for your OS.")
            self.voice_warning_printed = True

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

    def evaluate_expression(self, expression: str) -> float | None:
        if not expression or len(expression) > 120:
            return None
        if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s%]+", expression):
            return None

        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
            ast.USub,
            ast.UAdd,
            ast.Constant,
            ast.Load,
        )

        try:
            tree = ast.parse(expression, mode="eval")
        except Exception:
            return None

        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return None
            if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
                return None

        try:
            result = eval(compile(tree, "<jarvis-calc>", "eval"), {"__builtins__": {}}, {})
        except Exception:
            return None

        if not isinstance(result, (int, float)):
            return None
        return float(result)

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

        relative_match = re.search(
            r"\s+in\s+(\d+)\s+(second|seconds|minute|minutes|hour|hours|day|days)\s*$",
            payload,
        )
        if relative_match:
            amount = int(relative_match.group(1))
            unit = relative_match.group(2)
            if "second" in unit:
                due_at = now + dt.timedelta(seconds=amount)
            elif "minute" in unit:
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
                escaped_title = title.replace('"', '\\"')
                due_at_text = due_at.isoformat(timespec="seconds")
                self.emit_action(
                    f'add_reminder("{escaped_title}", "{due_at_text}")',
                    "Creating reminder from planned action.",
                )
                self.speak(
                    f"Reminder set for {title}, due {self.format_due_time(due_at)}. ID {reminder.get('id')}."
                )

        return True

    def open_application(self, app_name: str) -> bool:
        system_name = platform.system().lower()

        windows_map = {
            "notepad": ["notepad"],
            "calculator": ["calc"],
            "calc": ["calc"],
            "paint": ["mspaint"],
            "explorer": ["explorer"],
            "file explorer": ["explorer"],
            "terminal": ["powershell"],
            "powershell": ["powershell"],
            "cmd": ["cmd"],
            "command prompt": ["cmd"],
            "code": ["code"],
            "vscode": ["code"],
            "visual studio code": ["code"],
            "task manager": ["taskmgr"],
            "settings": ["start", "ms-settings:"],
            "control panel": ["control"],
            "snipping tool": ["snippingtool"],
            "wordpad": ["wordpad"],
            "browser": ["start", "https://"],
            "edge": ["start", "msedge:"],
            "chrome": ["start", "chrome"],
            "firefox": ["start", "firefox"],
            "spotify": ["start", "spotify:"],
            "discord": ["start", "discord:"],
        }

        linux_map = {
            "terminal": ["x-terminal-emulator"],
            "files": ["xdg-open", "."],
            "file manager": ["xdg-open", "."],
            "browser": ["xdg-open", "https://"],
            "calculator": ["gnome-calculator"],
            "calc": ["gnome-calculator"],
            "text editor": ["gedit"],
            "code": ["code"],
            "vscode": ["code"],
            "visual studio code": ["code"],
            "firefox": ["firefox"],
            "chrome": ["google-chrome"],
        }

        mac_map = {
            "terminal": ["open", "-a", "Terminal"],
            "finder": ["open", "-a", "Finder"],
            "browser": ["open", "-a", "Safari"],
            "safari": ["open", "-a", "Safari"],
            "chrome": ["open", "-a", "Google Chrome"],
            "firefox": ["open", "-a", "Firefox"],
            "calculator": ["open", "-a", "Calculator"],
            "calc": ["open", "-a", "Calculator"],
            "notes": ["open", "-a", "Notes"],
            "code": ["open", "-a", "Visual Studio Code"],
            "vscode": ["open", "-a", "Visual Studio Code"],
            "visual studio code": ["open", "-a", "Visual Studio Code"],
            "settings": ["open", "-a", "System Preferences"],
            "spotify": ["open", "-a", "Spotify"],
            "discord": ["open", "-a", "Discord"],
        }

        if "windows" in system_name:
            app_map = windows_map
        elif "darwin" in system_name:
            app_map = mac_map
        else:
            app_map = linux_map

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

    # ── Weather via wttr.in (no API key needed) ──────────────────────

    def fetch_weather(self, location: str = "") -> str | None:
        loc = quote_plus(location.strip()) if location.strip() else ""
        url = f"https://wttr.in/{loc}?format=%l:+%C+%t+%h+humidity+wind+%w&m"
        req = url_request.Request(url, headers={"User-Agent": "Jarvis/1.0"})
        try:
            with url_request.urlopen(req, timeout=10) as resp:
                text = resp.read().decode("utf-8").strip()
                if "Unknown location" in text or "Sorry" in text:
                    return None
                return text
        except Exception:
            return None

    # ── Wikipedia summary (no API key needed) ────────────────────────

    def fetch_wikipedia_summary(self, topic: str) -> str | None:
        encoded = quote_plus(topic.strip())
        url = (
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        )
        req = url_request.Request(url, headers={"User-Agent": "Jarvis/1.0"})
        try:
            with url_request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                extract = data.get("extract", "").strip()
                if not extract:
                    return None
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", extract) if s.strip()]
                return " ".join(sentences[:4])
        except Exception:
            return None

    # ── Countdown timer ──────────────────────────────────────────────

    def start_timer(self, seconds: int, label: str = "Timer") -> None:
        def _timer_callback() -> None:
            time.sleep(seconds)
            self.speak(f"{label} is up! {seconds} seconds have elapsed.")

        thread = threading.Thread(target=_timer_callback, daemon=True)
        thread.start()

    def parse_timer_request(self, command: str) -> tuple[int, str] | None:
        match = re.search(
            r"(?:set\s+(?:a\s+)?timer|timer)\s+(?:for\s+)?(\d+)\s*(second|seconds|sec|minute|minutes|min|hour|hours|hr)",
            command,
        )
        if not match:
            return None
        amount = int(match.group(1))
        unit = match.group(2).lower()
        if "min" in unit:
            amount *= 60
        elif "hour" in unit or "hr" in unit:
            amount *= 3600
        label = f"{match.group(1)} {match.group(2)} timer"
        return amount, label

    # ── System info helpers ──────────────────────────────────────────

    def get_ip_address(self) -> str | None:
        try:
            req = url_request.Request(
                "https://api.ipify.org?format=json",
                headers={"User-Agent": "Jarvis/1.0"},
            )
            with url_request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("ip")
        except Exception:
            return None

    def get_disk_usage(self) -> str:
        total, used, free = shutil.disk_usage("/")
        total_gb = total / (1 << 30)
        used_gb = used / (1 << 30)
        free_gb = free / (1 << 30)
        pct = (used / total) * 100 if total else 0
        return f"Disk: {used_gb:.1f} GB used of {total_gb:.1f} GB ({pct:.0f}% used), {free_gb:.1f} GB free"

    def get_uptime(self) -> str | None:
        system_name = platform.system().lower()
        try:
            if "linux" in system_name:
                with open("/proc/uptime") as f:
                    secs = float(f.read().split()[0])
            elif "darwin" in system_name:
                result = subprocess.run(
                    ["sysctl", "-n", "kern.boottime"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                match = re.search(r"sec\s*=\s*(\d+)", result.stdout)
                if not match:
                    return None
                boot = int(match.group(1))
                secs = time.time() - boot
            elif "windows" in system_name:
                result = subprocess.run(
                    ["wmic", "os", "get", "LastBootUpTime"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                lines = [l.strip() for l in result.stdout.splitlines() if l.strip() and l.strip() != "LastBootUpTime"]
                if not lines:
                    return None
                raw = lines[0].split(".")[0]
                boot_dt = dt.datetime.strptime(raw, "%Y%m%d%H%M%S")
                secs = (dt.datetime.now() - boot_dt).total_seconds()
            else:
                return None

            days = int(secs // 86400)
            hours = int((secs % 86400) // 3600)
            mins = int((secs % 3600) // 60)
            parts = []
            if days:
                parts.append(f"{days} day{'s' if days != 1 else ''}")
            if hours:
                parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
            parts.append(f"{mins} minute{'s' if mins != 1 else ''}")
            return ", ".join(parts)
        except Exception:
            return None

    # ── Random / coin flip ───────────────────────────────────────────

    def coin_flip(self) -> str:
        return random.choice(["heads", "tails"])

    def roll_dice(self, sides: int = 6) -> int:
        return random.randint(1, max(sides, 2))

    def pick_random(self, options: list[str]) -> str:
        return random.choice(options)

    # ── Jokes ────────────────────────────────────────────────────────

    _JOKES = [
        "Why do programmers prefer dark mode? Because light attracts bugs.",
        "There are 10 types of people in the world: those who understand binary and those who don't.",
        "A SQL query walks into a bar, sees two tables, and asks: Can I join you?",
        "Why was the JavaScript developer sad? Because he didn't Node how to Express himself.",
        "What is a computer's least favorite food? Spam.",
        "How many programmers does it take to change a light bulb? None, that is a hardware problem.",
        "Why did the developer go broke? Because he used up all his cache.",
        "What is the object-oriented way to become wealthy? Inheritance.",
        "Why do Java developers wear glasses? Because they cannot C#.",
        "!false. It is funny because it is true.",
    ]

    def tell_joke(self) -> str:
        return random.choice(self._JOKES)

    # ── Enhanced calculator with math functions ──────────────────────

    def evaluate_expression_enhanced(self, expression: str) -> float | None:
        if not expression or len(expression) > 200:
            return None

        expr = expression.strip()
        expr = re.sub(r"(\d)\s*\*\*\s*(\d)", r"\1**\2", expr)
        expr = re.sub(r"\^", "**", expr)

        safe_funcs = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log10,
            "ln": math.log,
            "abs": abs,
            "round": round,
            "floor": math.floor,
            "ceil": math.ceil,
            "pi": math.pi,
            "e": math.e,
        }

        if not re.fullmatch(r"[0-9a-z\.\+\-\*\/\(\)\s%,]+", expr):
            return None

        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Call,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.Constant,
            ast.Load,
            ast.Name,
        )

        try:
            tree = ast.parse(expr, mode="eval")
        except Exception:
            return None

        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return None
            if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
                return None
            if isinstance(node, ast.Name) and node.id not in safe_funcs:
                return None

        try:
            result = eval(  # noqa: S307
                compile(tree, "<jarvis-calc>", "eval"),
                {"__builtins__": {}},
                safe_funcs,
            )
        except Exception:
            return None

        if not isinstance(result, (int, float)):
            return None
        return float(result)

    # ── Unit conversion ──────────────────────────────────────────────

    _UNIT_CONVERSIONS: dict[tuple[str, str], float] = {
        ("km", "miles"): 0.621371,
        ("miles", "km"): 1.60934,
        ("kg", "lbs"): 2.20462,
        ("lbs", "kg"): 0.453592,
        ("cm", "inches"): 0.393701,
        ("inches", "cm"): 2.54,
        ("m", "feet"): 3.28084,
        ("feet", "m"): 0.3048,
        ("c", "f"): None,  # special formula
        ("f", "c"): None,  # special formula
        ("liters", "gallons"): 0.264172,
        ("gallons", "liters"): 3.78541,
    }

    def convert_unit(self, value: float, from_unit: str, to_unit: str) -> float | None:
        fu = from_unit.lower().rstrip("s").replace("kilomet", "km").replace("meter", "m")
        tu = to_unit.lower().rstrip("s").replace("kilomet", "km").replace("meter", "m")

        # Temperature special cases
        if fu in ("c", "celsius") and tu in ("f", "fahrenheit"):
            return value * 9 / 5 + 32
        if fu in ("f", "fahrenheit") and tu in ("c", "celsius"):
            return (value - 32) * 5 / 9

        # Normalize common synonyms
        synonyms = {
            "kilometer": "km", "mi": "miles", "mile": "miles",
            "kilogram": "kg", "pound": "lbs", "lb": "lbs",
            "centimeter": "cm", "inch": "inches",
            "foot": "feet", "ft": "feet",
            "liter": "liters", "gallon": "gallons", "gal": "gallons",
        }
        fu = synonyms.get(fu, fu)
        tu = synonyms.get(tu, tu)

        factor = self._UNIT_CONVERSIONS.get((fu, tu))
        if factor is not None:
            return value * factor
        return None

    def parse_conversion(self, command: str) -> tuple[float, str, str] | None:
        match = re.search(
            r"convert\s+([\d.]+)\s+(\w+)\s+(?:to|in|into)\s+(\w+)",
            command,
        )
        if not match:
            return None
        try:
            value = float(match.group(1))
        except ValueError:
            return None
        return value, match.group(2), match.group(3)

    # ── Clipboard ────────────────────────────────────────────────────

    def copy_to_clipboard(self, text: str) -> bool:
        system_name = platform.system().lower()
        try:
            if "windows" in system_name:
                process = subprocess.Popen(
                    ["clip"], stdin=subprocess.PIPE, shell=False,
                )
                process.communicate(text.encode("utf-16le"))
                return process.returncode == 0
            elif "darwin" in system_name:
                process = subprocess.Popen(
                    ["pbcopy"], stdin=subprocess.PIPE,
                )
                process.communicate(text.encode("utf-8"))
                return process.returncode == 0
            else:
                for cmd in [["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]:
                    try:
                        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                        process.communicate(text.encode("utf-8"))
                        if process.returncode == 0:
                            return True
                    except FileNotFoundError:
                        continue
                return False
        except Exception:
            return False

    def read_clipboard(self) -> str | None:
        system_name = platform.system().lower()
        try:
            if "windows" in system_name:
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", "Get-Clipboard"],
                    capture_output=True, text=True, timeout=5, check=False,
                )
                return result.stdout.strip() if result.returncode == 0 else None
            elif "darwin" in system_name:
                result = subprocess.run(
                    ["pbpaste"], capture_output=True, text=True, timeout=5, check=False,
                )
                return result.stdout.strip() if result.returncode == 0 else None
            else:
                for cmd in [["xclip", "-selection", "clipboard", "-o"], ["xsel", "--clipboard", "--output"]]:
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
                        if result.returncode == 0:
                            return result.stdout.strip()
                    except FileNotFoundError:
                        continue
                return None
        except Exception:
            return None

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
        except Exception as ex:
            print(f"[LISTEN ERROR] {ex}")
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

    def handle_command(self, command: str) -> bool:
        if not command:
            return True

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
                "I can hold conversations, remember your preferences, plan goals, run actions on your machine, "
                "check the weather, look up Wikipedia, set timers, do math and unit conversions, "
                "flip a coin, tell jokes, show system info, and manage your clipboard. "
                "Try: weather London, wiki Python, timer for 5 minutes, convert 100 km to miles, "
                "flip a coin, tell me a joke, system info, my ip, disk usage, "
                "copy to clipboard hello world, or say help for more."
            )
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
            escaped_title = title.replace('"', '\\"')
            due_at_text = due_at.isoformat(timespec="seconds")
            self.emit_action(
                f'add_reminder("{escaped_title}", "{due_at_text}")',
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

        if command.startswith("calculate ") or command.startswith("what is "):
            expression = command.replace("calculate ", "", 1)
            if command.startswith("what is "):
                expression = command.replace("what is ", "", 1).strip().rstrip("?")
            value = self.evaluate_expression_enhanced(expression.strip())
            if value is None:
                value = self.evaluate_expression(expression.strip())
            if value is None:
                if command.startswith("calculate "):
                    self.speak("I could not parse that calculation. Try: calculate sqrt(144) or calculate 2^10.")
                    return True
            else:
                if isinstance(value, float) and value.is_integer():
                    self.speak(f"The result is {int(value)}.")
                else:
                    self.speak(f"The result is {round(value, 6)}.")
                return True
        if command.startswith("calculate "):
            self.speak("I could not parse that calculation. Try: calculate sqrt(144) or calculate 2^10.")
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
            parts = [f"{platform.system()} {platform.release()}, Python {platform.python_version()}"]
            uptime = self.get_uptime()
            if uptime:
                parts.append(f"Uptime: {uptime}")
            parts.append(self.get_disk_usage())
            self.speak(". ".join(parts))
            return True

        # ── Weather ──────────────────────────────────────────────────
        if command.startswith("weather") or command.startswith("what's the weather"):
            location = ""
            if command.startswith("weather "):
                location = command.replace("weather ", "", 1).strip()
            elif command.startswith("what's the weather in "):
                location = command.replace("what's the weather in ", "", 1).strip()
            elif command.startswith("what's the weather"):
                location = ""
            self.speak("Checking the weather...")
            result = self.fetch_weather(location)
            if result:
                self.speak(result)
            else:
                self.speak("I could not get weather data right now. Try: weather London.")
            return True

        # ── Wikipedia ────────────────────────────────────────────────
        if command.startswith(("wiki ", "wikipedia ", "look up ", "define ")):
            for prefix in ("wikipedia ", "wiki ", "look up ", "define "):
                if command.startswith(prefix):
                    topic = command.replace(prefix, "", 1).strip()
                    break
            if topic:
                self.speak(f"Looking up {topic}...")
                summary = self.fetch_wikipedia_summary(topic)
                if summary:
                    self.speak(summary)
                else:
                    self.speak(f"I could not find a Wikipedia article for {topic}.")
            else:
                self.speak("What should I look up?")
            return True

        # ── Timer ────────────────────────────────────────────────────
        timer_result = self.parse_timer_request(command)
        if timer_result is not None:
            seconds, label = timer_result
            self.start_timer(seconds, label)
            if seconds >= 3600:
                display = f"{seconds // 3600} hour{'s' if seconds >= 7200 else ''}"
            elif seconds >= 60:
                display = f"{seconds // 60} minute{'s' if seconds >= 120 else ''}"
            else:
                display = f"{seconds} second{'s' if seconds != 1 else ''}"
            self.speak(f"Timer set for {display}. I will notify you when it is up.")
            return True

        # ── Unit conversion ──────────────────────────────────────────
        conv = self.parse_conversion(command)
        if conv is not None:
            value, from_unit, to_unit = conv
            result = self.convert_unit(value, from_unit, to_unit)
            if result is not None:
                if isinstance(result, float) and result == int(result):
                    self.speak(f"{value} {from_unit} is {int(result)} {to_unit}.")
                else:
                    self.speak(f"{value} {from_unit} is {round(result, 4)} {to_unit}.")
            else:
                self.speak(f"I do not know how to convert {from_unit} to {to_unit} yet.")
            return True

        # ── Coin flip / dice / random pick ───────────────────────────
        if "flip a coin" in command or "coin flip" in command or command == "flip":
            result = self.coin_flip()
            self.speak(f"It is {result}.")
            return True

        if command.startswith("roll a die") or command.startswith("roll a dice") or command == "roll":
            sides_match = re.search(r"(\d+)\s*sid", command)
            sides = int(sides_match.group(1)) if sides_match else 6
            result = self.roll_dice(sides)
            self.speak(f"You rolled a {result}.")
            return True

        if command.startswith("pick ") or command.startswith("choose "):
            parts_text = command.replace("pick ", "", 1).replace("choose ", "", 1).strip()
            options = [o.strip() for o in re.split(r",|\bor\b", parts_text) if o.strip()]
            if len(options) >= 2:
                choice = self.pick_random(options)
                self.speak(f"I pick {choice}.")
                return True

        # ── Jokes ────────────────────────────────────────────────────
        if "joke" in command or "make me laugh" in command or "funny" in command:
            self.speak(self.tell_joke())
            return True

        # ── IP address ───────────────────────────────────────────────
        if "my ip" in command or "ip address" in command:
            self.speak("Looking up your public IP address...")
            ip = self.get_ip_address()
            if ip:
                self.speak(f"Your public IP address is {ip}.")
            else:
                self.speak("I could not determine your public IP address.")
            return True

        # ── Disk usage ───────────────────────────────────────────────
        if "disk usage" in command or "disk space" in command or "storage" in command:
            self.speak(self.get_disk_usage())
            return True

        # ── Uptime ───────────────────────────────────────────────────
        if "uptime" in command:
            uptime = self.get_uptime()
            if uptime:
                self.speak(f"System uptime is {uptime}.")
            else:
                self.speak("I could not determine system uptime.")
            return True

        # ── Clipboard ────────────────────────────────────────────────
        if command.startswith("copy to clipboard ") or command.startswith("clipboard copy "):
            text = command.replace("copy to clipboard ", "", 1).replace("clipboard copy ", "", 1).strip()
            if text:
                if self.copy_to_clipboard(text):
                    self.speak("Copied to clipboard.")
                else:
                    self.speak("I could not copy to clipboard. A clipboard tool may not be installed.")
            else:
                self.speak("What should I copy to the clipboard?")
            return True

        if command in {"paste clipboard", "read clipboard", "what's on my clipboard", "clipboard"}:
            content = self.read_clipboard()
            if content:
                preview = content[:200]
                self.speak(f"Your clipboard contains: {preview}")
            else:
                self.speak("Your clipboard is empty or I cannot access it.")
            return True

        # ── Google shortcut ──────────────────────────────────────────
        if command.startswith("google "):
            query = command.replace("google ", "", 1).strip()
            if query:
                escaped_query = query.replace('"', '\\"')
                self.emit_action(f'search_web("{escaped_query}")', f"Searching Google for {query}.")
                self.search_web(query)
                self.speak(f"Searching Google for {query}.")
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
            "Try: weather, wiki, timer, calculate, convert, flip a coin, joke, system info, "
            "open, search, list files, plan, or say help."
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

    if use_voice:
        print(f"pyttsx3 engine available: {pyttsx3 is not None}")

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
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Disable microphone input (voice output remains enabled)",
    )
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

    use_voice = True
    use_microphone = not args.text_only and not args.keyboard_input

    if args.self_test:
        llm_client = None
        if not args.no_llm:
            llm_client = LocalLLMClient(
                base_url=args.ollama_url,
                model=args.model,
                timeout_seconds=max(args.llm_timeout, 10),
                max_tokens=max(args.llm_max_tokens, 32),
            )
        return run_self_test(
            use_voice=use_voice,
            use_microphone=use_microphone,
            mic_index=args.mic_index,
            llm_client=llm_client,
        )

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
            threading.Thread(target=llm_client.prewarm, daemon=True).start()

    try:
        while True:
            if assistant.use_microphone:
                raw = assistant.listen()
                if not raw:
                    continue
            else:
                raw = assistant.get_text_input()

            command = assistant.normalize_command(raw)
            keep_running = assistant.handle_command(command)
            if not keep_running:
                assistant.shutdown()
                return 0
    except KeyboardInterrupt:
        assistant.speak("Stopping Jarvis.")
        assistant.shutdown()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
