import os
import argparse
import subprocess
import torch

from model import ASR


def main(args):
    device = torch.device("cuda" if args.use_gpu else "cpu")
    output_file_name = os.path.basename(args.input_audio).split('.')[0] + '.txt'
    asr = ASR(args.sr, args.model, args.processor, args.use_lm, args.output_path, output_file_name)
    audio_format = asr.get_input_file_info(args.input_audio)
    # Convert audio format to .wav if not already in .wav
    if audio_format != 'wav':
        wav_audio_path = args.input_audio.split('.')[0]+'.wav'
        subprocess.call(['ffmpeg', '-i', args.input_audio, wav_audio_path])
        audio = asr.load_audio(wav_audio_path)
    else:
        audio = asr.load_audio(args.input_audio)

    input_values = asr.processor(audio, return_tensors="pt", sampling_rate=args.sr).input_values
    input_values = input_values.to(device)
    asr.model.to(device)
    asr.model.eval()
    with torch.no_grad():
        logits = asr.model(input_values).logits

    transcription = asr.decode(logits)
    output_file_name = os.path.basename(args.input_audio).split('.')[0]+'.txt'
    asr.save_transcripts(transcription)
    print(f'\nTranscript {output_file_name} saved at {args.output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_audio', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--processor', type=str, help="Make sure to give the "
                                                      "processor path based on LM being used or not.")
    parser.add_argument('--sr', type=int, default=16000, help="Set the sample rate (in Hz) of the audio to the "
                                                              "value on which the model was trained on")
    parser.add_argument('--use_lm', type=bool, default=True, help="Set to True if you want to use the language model.")
    parser.add_argument('--use_gpu', type=bool, default=False)
    args = parser.parse_args()
    main(args)