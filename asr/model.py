import os
import argparse
import fleep
import librosa
import subprocess
import torch

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM

class ASR:
    def __init__(self, sample_rate, model_path, processor_path, use_lm, output_path, output_file_name):
        self.sample_rate = sample_rate
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        self.use_lm = use_lm
        if self.use_lm:
            self.processor = Wav2Vec2ProcessorWithLM.from_pretrained(processor_path)
        else:
            self.processor = Wav2Vec2Processor.from_pretrained(processor_path)
        self.output_path = output_path
        self.output_file_name = output_file_name

    def decode(self, logits):
        if self.use_lm:
            transcription = self.processor.batch_decode(logits.detach().cpu().numpy()).text[0].lower()
        else:
            predicted_ids = torch.argmax(logits.cpu(), dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0].lower()
        return transcription

    def get_input_file_info(self, input_path):
        with open(input_path, "rb") as file:
            info = fleep.get(file.read(128))
        assert info.type[0] == 'audio', f"Expected input file to be audio, got: {info.type[0]}"
        return info.extension[0]

    def load_audio(self, audio_path):
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        return audio

    def save_transcripts(self, transcript):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        with open(self.output_path+'/'+self.output_file_name, 'w') as fp:
            fp.write(transcript)
