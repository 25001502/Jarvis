"""
Jarvis GUI - A cool tech interface for the Jarvis assistant
Inspired by futuristic AI interfaces with a modern dark theme
"""
import argparse
import datetime as dt
import json
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from tkinter import scrolledtext, ttk

try:
    import customtkinter as ctk
    HAS_CUSTOMTKINTER = True
except ImportError:
    HAS_CUSTOMTKINTER = False
    ctk = None

from jarvis import JarvisAssistant, LocalLLMClient


class JarvisGUI:
    def __init__(self, assistant: JarvisAssistant):
        self.assistant = assistant
        self.root = None
        self.running = True
        self.conversation_text = None
        self.input_field = None
        self.status_label = None
        self.llm_status_label = None
        self.reminder_count_label = None
        self.waveform_canvas = None
        self.waveform_animation_id = None
        self.waveform_offset = 0

        # Override assistant's speak method to display in GUI
        self.original_speak = assistant.speak
        assistant.speak = self.gui_speak

    def gui_speak(self, text: str) -> None:
        """Override speak to display in GUI and call original"""
        if self.conversation_text:
            self.conversation_text.config(state=tk.NORMAL)
            self.conversation_text.insert(tk.END, f"JARVIS: {text}\n\n", "jarvis")
            self.conversation_text.see(tk.END)
            self.conversation_text.config(state=tk.DISABLED)
        # Call original speak for voice output
        self.original_speak(text)

    def create_gui(self):
        """Create the main GUI window"""
        if HAS_CUSTOMTKINTER:
            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("blue")
            self.root = ctk.CTk()
        else:
            self.root = tk.Tk()
            self.root.configure(bg="#0a0e27")

        self.root.title("J.A.R.V.I.S - Just A Rather Very Intelligent System")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)

        # Configure styles
        if not HAS_CUSTOMTKINTER:
            style = ttk.Style()
            style.theme_use('clam')
            style.configure(".", background="#0a0e27", foreground="#00d4ff")

        # Create main container
        main_frame = self._create_frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header
        self._create_header(main_frame)

        # Status bar
        self._create_status_bar(main_frame)

        # Waveform visualization
        self._create_waveform(main_frame)

        # Conversation display
        self._create_conversation_display(main_frame)

        # Input area
        self._create_input_area(main_frame)

        # Start update loop
        self.update_status()
        self.animate_waveform()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _create_frame(self, parent):
        """Create a frame with proper styling"""
        if HAS_CUSTOMTKINTER:
            return ctk.CTkFrame(parent, fg_color="#0a0e27")
        else:
            frame = tk.Frame(parent, bg="#0a0e27")
            return frame

    def _create_header(self, parent):
        """Create header with title"""
        header_frame = self._create_frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        if HAS_CUSTOMTKINTER:
            title_label = ctk.CTkLabel(
                header_frame,
                text="⚡ J.A.R.V.I.S ⚡",
                font=ctk.CTkFont(size=32, weight="bold"),
                text_color="#00d4ff"
            )
        else:
            title_label = tk.Label(
                header_frame,
                text="⚡ J.A.R.V.I.S ⚡",
                font=("Arial", 32, "bold"),
                fg="#00d4ff",
                bg="#0a0e27"
            )
        title_label.pack(pady=10)

        if HAS_CUSTOMTKINTER:
            subtitle_label = ctk.CTkLabel(
                header_frame,
                text="Just A Rather Very Intelligent System",
                font=ctk.CTkFont(size=12),
                text_color="#00a8cc"
            )
        else:
            subtitle_label = tk.Label(
                header_frame,
                text="Just A Rather Very Intelligent System",
                font=("Arial", 12),
                fg="#00a8cc",
                bg="#0a0e27"
            )
        subtitle_label.pack()

    def _create_status_bar(self, parent):
        """Create status indicators"""
        status_frame = self._create_frame(parent)
        status_frame.pack(fill=tk.X, pady=(0, 10))

        # Create three columns for status
        for i in range(3):
            status_frame.columnconfigure(i, weight=1)

        # System status
        if HAS_CUSTOMTKINTER:
            self.status_label = ctk.CTkLabel(
                status_frame,
                text="● ONLINE",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="#00ff41"
            )
        else:
            self.status_label = tk.Label(
                status_frame,
                text="● ONLINE",
                font=("Arial", 11, "bold"),
                fg="#00ff41",
                bg="#0a0e27"
            )
        self.status_label.grid(row=0, column=0, padx=5, sticky="w")

        # LLM status
        if HAS_CUSTOMTKINTER:
            self.llm_status_label = ctk.CTkLabel(
                status_frame,
                text="🤖 LLM: Checking...",
                font=ctk.CTkFont(size=11),
                text_color="#00a8cc"
            )
        else:
            self.llm_status_label = tk.Label(
                status_frame,
                text="🤖 LLM: Checking...",
                font=("Arial", 11),
                fg="#00a8cc",
                bg="#0a0e27"
            )
        self.llm_status_label.grid(row=0, column=1, padx=5)

        # Reminders
        if HAS_CUSTOMTKINTER:
            self.reminder_count_label = ctk.CTkLabel(
                status_frame,
                text="📋 Reminders: 0",
                font=ctk.CTkFont(size=11),
                text_color="#00a8cc"
            )
        else:
            self.reminder_count_label = tk.Label(
                status_frame,
                text="📋 Reminders: 0",
                font=("Arial", 11),
                fg="#00a8cc",
                bg="#0a0e27"
            )
        self.reminder_count_label.grid(row=0, column=2, padx=5, sticky="e")

    def _create_waveform(self, parent):
        """Create animated waveform visualization"""
        waveform_frame = self._create_frame(parent)
        waveform_frame.pack(fill=tk.X, pady=(0, 10))

        self.waveform_canvas = tk.Canvas(
            waveform_frame,
            height=60,
            bg="#0a0e27",
            highlightthickness=0
        )
        self.waveform_canvas.pack(fill=tk.X, padx=5)

    def _create_conversation_display(self, parent):
        """Create conversation history display"""
        conv_frame = self._create_frame(parent)
        conv_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        if HAS_CUSTOMTKINTER:
            label = ctk.CTkLabel(
                conv_frame,
                text="CONVERSATION LOG",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="#00d4ff"
            )
        else:
            label = tk.Label(
                conv_frame,
                text="CONVERSATION LOG",
                font=("Arial", 12, "bold"),
                fg="#00d4ff",
                bg="#0a0e27"
            )
        label.pack(anchor="w", padx=5, pady=(0, 5))

        # Create text widget
        self.conversation_text = scrolledtext.ScrolledText(
            conv_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg="#0f1425",
            fg="#00ff41",
            insertbackground="#00d4ff",
            selectbackground="#1a3a52",
            relief=tk.FLAT,
            borderwidth=2
        )
        self.conversation_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Configure tags for styling
        self.conversation_text.tag_config("user", foreground="#00d4ff", font=("Consolas", 10, "bold"))
        self.conversation_text.tag_config("jarvis", foreground="#00ff41", font=("Consolas", 10))
        self.conversation_text.tag_config("system", foreground="#ffa500", font=("Consolas", 9, "italic"))

        self.conversation_text.config(state=tk.DISABLED)

        # Welcome message
        self.display_system_message(f"Welcome, {self.assistant.display_name()}. I am JARVIS, your local AI assistant.")
        self.display_system_message("Type your commands below or say 'help' for available commands.")

    def _create_input_area(self, parent):
        """Create command input area"""
        input_frame = self._create_frame(parent)
        input_frame.pack(fill=tk.X)

        if HAS_CUSTOMTKINTER:
            label = ctk.CTkLabel(
                input_frame,
                text="COMMAND INPUT",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="#00d4ff"
            )
        else:
            label = tk.Label(
                input_frame,
                text="COMMAND INPUT",
                font=("Arial", 12, "bold"),
                fg="#00d4ff",
                bg="#0a0e27"
            )
        label.pack(anchor="w", padx=5, pady=(0, 5))

        # Input field
        if HAS_CUSTOMTKINTER:
            self.input_field = ctk.CTkEntry(
                input_frame,
                placeholder_text="Enter command...",
                font=ctk.CTkFont(size=12),
                height=40
            )
        else:
            self.input_field = tk.Entry(
                input_frame,
                font=("Consolas", 12),
                bg="#0f1425",
                fg="#00d4ff",
                insertbackground="#00d4ff",
                relief=tk.FLAT,
                borderwidth=2
            )
        self.input_field.pack(fill=tk.X, padx=5, pady=5)
        self.input_field.bind("<Return>", self.on_submit)
        self.input_field.focus()

        # Submit button
        if HAS_CUSTOMTKINTER:
            submit_btn = ctk.CTkButton(
                input_frame,
                text="⚡ EXECUTE",
                command=self.on_submit,
                font=ctk.CTkFont(size=12, weight="bold"),
                height=35,
                fg_color="#00d4ff",
                hover_color="#00a8cc"
            )
        else:
            submit_btn = tk.Button(
                input_frame,
                text="⚡ EXECUTE",
                command=self.on_submit,
                font=("Arial", 12, "bold"),
                bg="#00d4ff",
                fg="#0a0e27",
                activebackground="#00a8cc",
                relief=tk.FLAT,
                padx=20,
                pady=8,
                cursor="hand2"
            )
        submit_btn.pack(pady=5)

    def display_system_message(self, message: str):
        """Display a system message in the conversation"""
        if self.conversation_text:
            self.conversation_text.config(state=tk.NORMAL)
            self.conversation_text.insert(tk.END, f"[SYSTEM] {message}\n\n", "system")
            self.conversation_text.see(tk.END)
            self.conversation_text.config(state=tk.DISABLED)

    def on_submit(self, event=None):
        """Handle command submission"""
        command = self.input_field.get().strip()
        if not command:
            return

        # Clear input
        self.input_field.delete(0, tk.END)

        # Display user command
        self.conversation_text.config(state=tk.NORMAL)
        self.conversation_text.insert(tk.END, f"YOU: {command}\n", "user")
        self.conversation_text.see(tk.END)
        self.conversation_text.config(state=tk.DISABLED)

        # Process command in background thread
        threading.Thread(target=self.process_command, args=(command,), daemon=True).start()

    def process_command(self, command: str):
        """Process command through Jarvis assistant"""
        normalized = self.assistant.normalize_command(command)

        if normalized in {"exit", "quit", "shutdown", "goodbye"}:
            self.root.after(0, self.on_closing)
            return

        keep_running = self.assistant.handle_command(normalized)
        if not keep_running:
            self.root.after(0, self.on_closing)

    def update_status(self):
        """Update status indicators"""
        if not self.running:
            return

        # Update LLM status
        if self.assistant.llm_client:
            llm_ready, llm_msg = self.assistant.llm_status()
            status_text = "🤖 LLM: Ready" if llm_ready else "🤖 LLM: Offline"
            color = "#00ff41" if llm_ready else "#ff6b6b"
        else:
            status_text = "🤖 LLM: Disabled"
            color = "#808080"

        if HAS_CUSTOMTKINTER:
            self.llm_status_label.configure(text=status_text, text_color=color)
        else:
            self.llm_status_label.configure(text=status_text, fg=color)

        # Update reminder count
        pending = self.assistant.list_pending_reminders()
        reminder_text = f"📋 Reminders: {len(pending)}"
        if HAS_CUSTOMTKINTER:
            self.reminder_count_label.configure(text=reminder_text)
        else:
            self.reminder_count_label.configure(text=reminder_text)

        # Schedule next update
        self.root.after(2000, self.update_status)

    def animate_waveform(self):
        """Animate the waveform visualization"""
        if not self.running:
            return

        self.waveform_canvas.delete("all")
        width = self.waveform_canvas.winfo_width()
        height = self.waveform_canvas.winfo_height()

        if width <= 1:
            self.root.after(50, self.animate_waveform)
            return

        mid_y = height / 2

        # Draw multiple wave layers
        colors = ["#00d4ff", "#00a8cc", "#007a99"]
        amplitudes = [15, 10, 5]
        frequencies = [0.05, 0.08, 0.12]

        for i, (color, amplitude, frequency) in enumerate(zip(colors, amplitudes, frequencies)):
            points = []
            for x in range(0, width, 2):
                y = mid_y + amplitude * (
                    0.5 * (x / 20 + self.waveform_offset * frequency) % 2 - 1
                )
                points.extend([x, y])

            if len(points) >= 4:
                self.waveform_canvas.create_line(
                    *points,
                    fill=color,
                    width=2,
                    smooth=True
                )

        self.waveform_offset += 1

        # Schedule next frame
        self.waveform_animation_id = self.root.after(50, self.animate_waveform)

    def on_closing(self):
        """Handle window closing"""
        self.running = False
        if self.waveform_animation_id:
            self.root.after_cancel(self.waveform_animation_id)

        self.display_system_message("Shutting down JARVIS...")
        self.assistant.shutdown()
        self.assistant.save_persistent_memory()

        self.root.after(500, self.root.destroy)

    def run(self):
        """Start the GUI"""
        self.create_gui()
        self.root.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Jarvis GUI - Local AI Assistant Interface")
    parser.add_argument("--no-llm", action="store_true", help="Disable local Ollama conversation fallback")
    parser.add_argument("--model", default="llama3.2:1b", help="Ollama model name to use for local chat")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama server URL")
    parser.add_argument("--llm-timeout", type=int, default=60, help="Timeout in seconds for local LLM responses")
    parser.add_argument("--llm-max-tokens", type=int, default=96, help="Max tokens generated by local LLM")
    parser.add_argument("--memory-file", default=".jarvis_memory.json", help="Path to persistent Jarvis memory file")
    parser.add_argument("--history-turns", type=int, default=12, help="Number of recent conversation turns to retain")
    args = parser.parse_args()

    # Create LLM client if enabled
    llm_client = None
    if not args.no_llm:
        llm_client = LocalLLMClient(
            base_url=args.ollama_url,
            model=args.model,
            timeout_seconds=max(args.llm_timeout, 10),
            max_tokens=max(args.llm_max_tokens, 32),
        )

    # Create Jarvis assistant (GUI mode - no voice/microphone by default)
    assistant = JarvisAssistant(
        use_voice=False,
        use_microphone=False,
        llm_client=llm_client,
        memory_file=args.memory_file,
        max_history_turns=max(args.history_turns, 6),
    )

    # Check if customtkinter is available
    if not HAS_CUSTOMTKINTER:
        print("Note: For enhanced visuals, install customtkinter: pip install customtkinter")
        print("Starting with standard tkinter interface...\n")

    # Create and run GUI
    gui = JarvisGUI(assistant)
    gui.run()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
