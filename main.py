import time, os
t = time.process_time()
import warnings
import threading
import numpy as np
import whisper
import sounddevice as sd
from queue import Queue
from rich.console import Console
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from TextToSpeechServiceMMS import TextToSpeechService
print("Time elapsed to load packages: " + str(time.process_time() - t))

# Suppress specific FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.")

console = Console()
t = time.process_time()
stt = whisper.load_model("base")
print("Time elapsed to load Whisper: " + str(time.process_time() - t))
t = time.process_time()
tts = TextToSpeechService()
print("Time elapsed to load TextToSpeechService: " + str(time.process_time() - t))

template = """
You are a friendly AI assistant that helps me learning Portuguese. You are precise in grammar and vocabulary. If I make a mistake, you correct me and provide an explanation and translation if needed. Keep it short and crisp. Always respond in Portuguese! 
Use the following format:
---
<A response correcting my mistakes with a quick explanation in Portuguese (max. 2 sentences).>
<Your quick response to my statement containing a follow-up question in Portuguese (max. 2 sentences).>
---
The conversation transcript is as follows:
{history}
And here is the user's statement in Portuguese: {input}
Your response:
"""

t = time.process_time()
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
chain = ConversationChain(
    prompt=PROMPT,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="Assistant:"),
    llm=Ollama(model="mistral"),
)
print("Time elapsed to define ConversationChain: " + str(time.process_time() - t))


def record_audio(stop_event, data_queue):

    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(samplerate=16000, dtype="int16", channels=1, callback=callback):
        while not stop_event.is_set():
            time.sleep(0.1)


def transcribe(audio_np: np.ndarray) -> str:
    result = stt.transcribe(audio_np, language="pt", fp16=False)
    text = result["text"].strip()
    return text


def get_llm_response(text: str) -> str:
    response = chain.predict(input=text)
    if response.startswith("Assistant:"):
        response = response[len("Assistant:") :].strip()
    return response


def play_audio(sample_rate, audio_array):
    sd.play(audio_array, sample_rate)
    sd.wait()


if __name__ == "__main__":

    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")

    try:
        while True:
            console.input("Press Enter to start recording, then press Enter again to stop recording")

            data_queue = Queue()
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue)
            )
            recording_thread.start()

            input()
            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            if audio_np.size > 0:
                with console.status("Transcribing...", spinner="earth"):
                    text = transcribe(audio_np)
                console.print(f"[yellow]You: {text}")

                with console.status("Generating response...", spinner="earth"):
                    t = time.process_time()
                    response = get_llm_response(text)
                    print("Time elapsed to get response from LLM: " + str(time.process_time() - t))
                    # espeak TTS alternative: espeak_ng(response)
                    t = time.process_time()
                    sample_rate, audio_array = tts.long_form_synthesize(response)
                    print("Time elapsed to synthesize response: " + str(time.process_time() - t))

                console.print(f"[cyan]Assistant: {response}")
                play_audio(sample_rate, audio_array)
            else:
                console.print("[red]No audio recorded. Please check audio input.")

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")
            