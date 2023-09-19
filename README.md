# Setup

```
conda create --name AD python=3.8
conda activate AD
pip install -r requirements.txt
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html # Need cu111 for RTX A4000  
conda deactivate
```

# Input

.m4a, .mp3, .wav file (if not .wav, automatically convert first)


# Output

.csv (e.g., "Mean Yngve score,	Mean Frazier Score,	SDL length,	Type to token ratio, lemmatized	Type to token ratio,	Honoroe's stats,	Automatic Readability Index,	Brunet's index and Coleman Liau's index")


# Example

```
python main.py data/example.m4a /tmp/output.csv
```
