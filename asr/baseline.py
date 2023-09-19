import os
import argparse
import glob
import time
import torch
import librosa

from jiwer import wer
from datasets import Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM

class ASRBaseline:
    def __init__(self, data_path, split, sample_rate, model_path, processor_path, use_lm, use_gpu):
        self.eval_path = data_path
        self.split = split
        self.sr = sample_rate
        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.use_lm = use_lm
        if self.use_lm:
            self.processor = Wav2Vec2ProcessorWithLM.from_pretrained(processor_path)
        else:
            self.processor = Wav2Vec2Processor.from_pretrained(processor_path)

    def read_txt_file(self, txt_f):
        with open(txt_f, "r") as fp:
            samples = fp.read().split("\n")
            samples = {s.split()[0]: " ".join(s.split()[1:]) for s in samples if len(s.split()) > 2}
        return samples

    def read_audio_file(self, audio_path):
        audio, sr = librosa.load(audio_path, sr=self.sr)
        file_name = os.path.basename(audio_path).split('.')[0]
        return {file_name: audio}

    def fetch_audio_text_mapping(self):
        audio_files = glob.glob(f"{self.eval_path}/{self.split}/*/*/*.flac")
        txt_files = glob.glob(f"{self.eval_path}/{self.split}/*/*/*.txt")

        txt_samples = {}
        for txt_f in txt_files:
            txt_samples.update(self.read_txt_file(txt_f))

        audio_samples = {}
        for audio_f in audio_files:
            audio_samples.update(self.read_audio_file(audio_f))

        file_ids = set(audio_samples.keys()) & set(txt_samples.keys())
        print(f"{len(file_ids)} files are found in LibriSpeech/{self.split}")
        audio_filtered_samples = [audio_samples[file_id] for file_id in file_ids]
        txt_filtered_samples = [txt_samples[file_id] for file_id in file_ids]
        assert len(txt_filtered_samples) == len(audio_filtered_samples), f"Number of transcripts and audio " \
                                                                         f"files should be equal."
        samples = {"audio_array": audio_filtered_samples, "text": txt_filtered_samples}
        return samples

    def map_to_pred(self, batch):
        input_values = self.processor(batch["audio_array"], return_tensors="pt", sampling_rate=self.sr).input_values
        input_values = input_values.to(self.device)
        with torch.no_grad():
            logits = self.model(input_values).logits
        if self.use_lm:
            transcription = self.processor.batch_decode(logits.detach().cpu().numpy()).text[0]
        else:
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
        batch["transcription"] = transcription
        return batch

def main(args):
    baseline = ASRBaseline(args.input_dir, args.split, args.sr, args.model, args.processor, args.use_lm, args.use_gpu)
    samples = baseline.fetch_audio_text_mapping()
    data_eval = Dataset.from_dict(samples)
    result = data_eval.map(baseline.map_to_pred)
    print("WER: ", wer(result["text"], result["transcription"]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, help="Expects LibriSpeech dataset")
    parser.add_argument('--split', type=str, help="Eg: test-clean, test-other")
    parser.add_argument('--model', type=str)
    parser.add_argument('--processor', type=str, help="Make sure to give the "
                                                      "processor path based on LM being used or not.")
    parser.add_argument('--sr', type=int, default=16000, help="Set the sample rate (in Hz) of the audio to the "
                                                              "value on which the model was trained on")
    parser.add_argument('--use_lm', type=bool, default=False, help="Set to True if you want to use the language model.")
    parser.add_argument('--use_gpu', type=bool, default=False)
    args = parser.parse_args()
    main(args)
