import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import os
import json

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

load_dotenv()

class LanguageModelProcessor:
    def __init__(self):
        self.api_url = "https://govnamics.onrender.com/api/v1/prediction/55acb087-8f53-4e7d-adf7-3e08463bb88a"
        self.headers = {"Content-Type": "application/json"}

        # Create a session with connection pooling
        self.session = requests.Session()
        
        # Set up retry mechanism
        retry_strategy = Retry(
            total=3,  # Total number of retries
            status_forcelist=[500, 502, 503, 504],  # Retry on these status codes
            backoff_factor=1  # Backoff factor for exponential backoff
        )
        
        # Create an adapter with the retry strategy
        adapter = HTTPAdapter(max_retries=retry_strategy)
        
        # Mount the adapter to the session
        self.session.mount("https://", adapter)

    def process(self, text, dialog_history):
        payload = {
            "question": text,
            "dialog_history": dialog_history
        }
        start_time = time.time()

        # Use the session to send the request
        response = self.session.post(self.api_url, json=payload, headers=self.headers)
        end_time = time.time()

        if response.status_code == 200:
            api_response = response.json()
            print(f"API Response: {api_response}")  # Print the API response

            if isinstance(api_response, dict) and "text" in api_response:
                response_text = api_response["text"]
            else:
                response_text = "Sorry, I couldn't find an answer to your question."
        else:
            response_text = "Sorry, I couldn't process your request at the moment."

        elapsed_time = int((end_time - start_time) * 1000)
        print(f"API ({elapsed_time}ms): {response_text}")
        return response_text

class TextToSpeech:
    # Set your Deepgram API Key and desired voice model
    DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    MODEL_NAME = "aura-stella-en"  # Example model name, change as needed

    @staticmethod
    def is_installed(lib_name: str) -> bool:
        lib = shutil.which(lib_name)
        return lib is not None

    def speak(self, text):
        if not self.is_installed("ffplay"):
            raise ValueError("ffplay not found, necessary to stream audio.")

        DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model={self.MODEL_NAME}&performance=some&encoding=linear16&sample_rate=24000"
        headers = {
            "Authorization": f"Token {self.DG_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "text": text
        }

        player_command = ["ffplay", "-autoexit", "-", "-nodisp"]
        player_process = subprocess.Popen(
            player_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        start_time = time.time()  # Record the time before sending the request
        first_byte_time = None  # Initialize a variable to store the time when the first byte is received

        with requests.post(DEEPGRAM_URL, stream=True, headers=headers, json=payload) as r:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    if first_byte_time is None:  # Check if this is the first chunk received
                        first_byte_time = time.time()  # Record the time when the first byte is received
                        ttfb = int((first_byte_time - start_time)*1000)  # Calculate the time to first byte
                        print(f"TTS Time to First Byte (TTFB): {ttfb}ms\n")
                    player_process.stdin.write(chunk)
                    player_process.stdin.flush()

        if player_process.stdin:
            player_process.stdin.close()
        player_process.wait()

class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)

transcript_collector = TranscriptCollector()

async def get_transcript(callback):
    transcription_complete = asyncio.Event()  # Event to signal transcription completion

    try:
        # example of setting up a client config. logging values: WARNING, VERBOSE, DEBUG, SPAM
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram: DeepgramClient = DeepgramClient("", config)

        dg_connection = deepgram.listen.asynclive.v("1")
        print("Listening...")

        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            
            if not result.speech_final:
                transcript_collector.add_part(sentence)
            else:
                # This is the final part of the current sentence
                transcript_collector.add_part(sentence)
                full_sentence = transcript_collector.get_full_transcript()
                # Check if the full_sentence is not empty before printing
                if len(full_sentence.strip()) > 0:
                    full_sentence = full_sentence.strip()
                    print(f"Human: {full_sentence}")
                    callback(full_sentence)  # Call the callback with the full_sentence
                    transcript_collector.reset()
                    transcription_complete.set()  # Signal to stop transcription and exit

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=300,
            smart_format=True,
        )

        await dg_connection.start(options)

        # Open a microphone stream on the default input device
        microphone = Microphone(dg_connection.send)
        microphone.start()

        await transcription_complete.wait()  # Wait for the transcription to complete instead of looping indefinitely

        # Wait for the microphone to close
        microphone.finish()

        # Indicate that we've finished
        await dg_connection.finish()

    except Exception as e:
        print(f"Could not open socket: {e}")
        return

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()
        self.dialog_history = []  # Initialize an empty list to store dialog history
        self.conversation_ended = False  # Flag to track if the conversation has ended

    def add_to_history(self, speaker, text):
        self.dialog_history.append({"speaker": speaker, "text": text})  # Append a dictionary with speaker and text

    def print_dialog_history(self):
        for dialog in self.dialog_history:
            print(f"{dialog['speaker']}: {dialog['text']}")

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence
            self.add_to_history("Human", full_sentence)  # Add human dialog to history

            # Check for "goodbye" to exit the loop
            if "goodbye" in self.transcription_response.lower():
                self.conversation_ended = True  # Set the flag to indicate the conversation has ended

        # Add the initial system greeting
        initial_greeting = "Thank you for calling the City of Carrollton, my Name is Carrie, can I help you?"
        self.add_to_history("System", initial_greeting)  # Add initial greeting to history

        tts = TextToSpeech()
        tts.speak(initial_greeting)  # Speak the initial greeting

        # Loop until the conversation is ended
        while not self.conversation_ended:
            await get_transcript(handle_full_sentence)
            
            if self.conversation_ended:
                break  # Exit the loop if the conversation has ended
            
            api_response = self.llm.process(self.transcription_response, self.dialog_history)
            self.add_to_history("System", api_response)  # Add system response to history

            tts.speak(api_response)

            # Optionally, print the dialog history after each exchange
            self.print_dialog_history()

            # Reset transcription_response for the next loop iteration
            self.transcription_response = ""

        print("Conversation ended. Goodbye!")

if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())