# Input

.m4a, .mp3, .wav file (if not .wav, automatically convert first)

# Output

.txt file (raw output of the corresponding speech file)

# Example

```
python main.py ../data/example.m4a /tmp/output.txt --model ../pretrained/models/ --processor ../processors/processor_with_lm --sr 16000 --use_lm True --use_gpu True
```
# Instructions for baseline

Download the test data (test-clean or test-other) from https://www.openslr.org/12 
Untar the file
Audio in flaac format along with gold transcripts will be found inside the split name under LibriSpeech/

```
python baseline.py ../data/LibriSpeech --split test-clean --model ../pretrained/models/ --processor ../processors/processor_with_lm --sr 16000 --use_gpu True
```
For test-clean and test-other dataset splits, the WER obtained should be 0.0338 and 0.0858 respectively on [wav2vec2-base-960h model](https://huggingface.co/facebook/wav2vec2-base-960h) similar to the scores listed for BASE LS-960 under Table 10 of [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477).
