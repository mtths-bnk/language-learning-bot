from transformers import VitsModel, AutoTokenizer, VitsTokenizer
import torch
import numpy as np
import nltk

nltk.download('punkt_tab')

class TextToSpeechService:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-por")
        self.model = VitsModel.from_pretrained("facebook/mms-tts-por")
        self.model.to(self.device)

    def synthesize(self, text: str):
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate speech waveform
        with torch.no_grad():
            output = self.model(**inputs).waveform

        # Convert the generated waveform to numpy array and extract sample rate
        audio_array = output.squeeze().cpu().numpy()
        sample_rate = self.model.config.sampling_rate
        return sample_rate, audio_array

    def long_form_synthesize(self, text: str):
        pieces = []
        sentences = nltk.sent_tokenize(text)
        silence = np.zeros(int(0.25 * self.model.config.sampling_rate))

        for sent in sentences:
            sample_rate, audio_array = self.synthesize(sent)
            pieces += [audio_array, silence.copy()]

        return self.model.config.sampling_rate, np.concatenate(pieces)
