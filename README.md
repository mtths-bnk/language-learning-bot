# Language Learning Bot
**The idea:** Learning languages with a voice bot – offline on your machine

Why paying for all these trending language apps that can "talk to you with AI" when you can run it on your own machine leveraging open source AI models? At least that was my initial thought as I built this simple app. :) 

This implementation is teaching European Portuguese.

The results are promising but unfortunately there are still a few limitations. 

### Limitations
- The mistral-7B model and my prompt are still limited in its language teaching capabilities. You can try with a larger model (comes with potentially larger response times) or by improving the prompt.


## Try it out!
### Requirements
- Tested on MacBook Air M2 (16 GB)
- Python (tested with 3.12.3)

### Installation
1. Install Ollama: https://ollama.com/
2. After Ollama installation, run `ollama pull mistral` in the Terminal
3. Clone the repository to your local machine `git clone [...]`
4. Create a virtual environment: `python3 -m venv venv`
5. Activate your virtual environment: `source venv/bin/activate`
6. Install required packages: `pip install -r requirements.txt`
7. Run the app: `python3 main.py`

## Technical Background
The app currently runs in a terminal. It uses [whisper](https://github.com/openai/whisper) as speech recognition model (speech to text) with the base model (multilingual) of 74M parameters. After transcribing the text, a large language model (in this case [mistral-7B](https://ollama.com/library/mistral:7b)) – running locally using Ollama – is interpreting the input and using a prompt to generate an answer. The speech synthesizer (text to speech) uses a [Massively Multilingual Speech (MMS)](https://huggingface.co/facebook/mms-tts) model, in this case the [Portuguese](https://huggingface.co/facebook/mms-tts-por) model. It leverages the [VITS implementation](https://huggingface.co/docs/transformers/model_doc/vits).

## Ideas to Improve
- Experiment with models and prompt for better results
- Build a simple frontend for easy use
- Implement language switch and prompt adjustment in frontend