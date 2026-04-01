import argparse
import datetime as dt
import json
import platform
from pathlib import Path
import subprocess
import sys
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


class LocalLLMClient:
    def __init__(self, base_url: str, model: str, timeout_seconds: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def _fetch_json(self, endpoint: str) -> dict | None:
        req = url_request.Request(f"{self.base_url}{endpoint}", method="GET")
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
                "temperature": 0.4,
            },
        }

        req = url_request.Request(
            f"{self.base_url}/api/generate",
            method="POST",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with url_request.urlopen(req, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            generated = payload.get("response", "").strip()
            return generated or None
        except (url_error.URLError, TimeoutError, ValueError):
            return None


class JarvisAssistant:
    def __init__(
        self,
        use_voice: bool = True,
        wake_word: str = "jarvis",
        llm_client: LocalLLMClient | None = None,
    ) -> None:
        self.use_voice = use_voice
        self.wake_word = wake_word.lower().strip()
        self.tts_engine = None
        self.llm_client = llm_client
        self.memory = {
            "user_goal": None,
            "ongoing_task": None,
            "notes": [],
            "previous_commands": [],
        }

        if self.use_voice and pyttsx3 is not None:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty("rate", 185)
            except Exception:
                self.tts_engine = None

    def speak(self, text: str) -> None:
        print(f"Jarvis: {text}")
        if self.use_voice and self.tts_engine is not None:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception:
                pass

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

    def reply_with_llm(self, command: str) -> bool:
        if self.llm_client is None:
            return False

        llm_ready, _ = self.llm_status()
        if not llm_ready:
            return False

        response = self.llm_client.generate(command, self.memory)
        if not response:
            return False

        self.speak(response)
        return True

    def remember_command(self, command: str) -> None:
        self.memory["previous_commands"].append(command)
        self.memory["previous_commands"] = self.memory["previous_commands"][-8:]

    def update_context_from_command(self, command: str) -> None:
        goal_markers = ["my goal is ", "i am trying to ", "i'm trying to ", "i want to "]
        task_markers = ["i am working on ", "i'm working on "]

        for marker in goal_markers:
            if marker in command:
                value = command.split(marker, 1)[1].strip(" .")
                if value:
                    self.memory["user_goal"] = value
                return

        for marker in task_markers:
            if marker in command:
                value = command.split(marker, 1)[1].strip(" .")
                if value:
                    self.memory["ongoing_task"] = value
                return

        if command.startswith("remember that "):
            note = command.replace("remember that ", "", 1).strip(" .")
            if note:
                self.memory["notes"].append(note)
                self.memory["notes"] = self.memory["notes"][-8:]

    def build_plan(self, objective: str) -> list[str]:
        cleaned = objective.strip()
        return [
            f"Define a clear success target for {cleaned}.",
            "Break the work into small milestones with a realistic order.",
            "Execute the first milestone and verify results quickly.",
            "Review progress, adjust, and continue to completion.",
        ]

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
        if not self.use_voice or sr is None:
            return None

        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                print("Listening...")
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=6, phrase_time_limit=8)
            text = recognizer.recognize_google(audio)
            print(f"You: {text}")
            return text
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

    def handle_command(self, command: str) -> bool:
        if not command:
            return True

        self.remember_command(command)
        self.update_context_from_command(command)

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
            self.speak("Shutting down. Talk to you later.")
            return False

        if command in {"help", "commands", "what can you do"}:
            self.speak(
                "I can chat naturally, remember context, plan tasks, and run computer actions. "
                "Try: time, date, open google, search python automation, list files, plan build my bot, "
                "llm status, and exit."
            )
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
                self.speak(f"Hello. Ready to continue with {current_task}?")
            else:
                self.speak("Hello. What would you like to work on today?")
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


def run_self_test(use_voice: bool, llm_client: LocalLLMClient | None = None) -> int:
    print("Running Jarvis self-test...")
    print(f"Python: {sys.version.split()[0]}")
    print(f"SpeechRecognition installed: {sr is not None}")
    print(f"pyttsx3 installed: {pyttsx3 is not None}")

    if use_voice and sr is not None:
        try:
            names = sr.Microphone.list_microphone_names()
            print(f"Microphones detected: {len(names)}")
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
    parser.add_argument("--self-test", action="store_true", help="Run environment checks and exit")
    parser.add_argument("--no-llm", action="store_true", help="Disable local Ollama conversation fallback")
    parser.add_argument("--model", default="llama3.2:1b", help="Ollama model name to use for local chat")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama server URL")
    parser.add_argument("--llm-timeout", type=int, default=120, help="Timeout in seconds for local LLM responses")
    args = parser.parse_args()

    use_voice = not args.text_only
    llm_client = None
    if not args.no_llm:
        llm_client = LocalLLMClient(
            base_url=args.ollama_url,
            model=args.model,
            timeout_seconds=max(args.llm_timeout, 10),
        )

    if args.self_test:
        return run_self_test(use_voice=use_voice, llm_client=llm_client)

    assistant = JarvisAssistant(use_voice=use_voice, llm_client=llm_client)
    assistant.speak("Jarvis online. Say help for commands. Say exit to stop.")

    if llm_client is not None:
        llm_ready, llm_message = assistant.llm_status()
        assistant.speak(llm_message)
        if not llm_ready:
            assistant.speak(f'If needed, run: ollama pull {args.model}')

    while True:
        if use_voice:
            raw = assistant.listen()
            if not raw:
                continue
        else:
            raw = assistant.get_text_input()
        command = assistant.normalize_command(raw)
        keep_running = assistant.handle_command(command)
        if not keep_running:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
