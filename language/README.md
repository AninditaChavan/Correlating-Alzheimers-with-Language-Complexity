# Input

.txt (raw output of a speech file)


# Output

.csv (e.g., "Mean Yngve score,	Mean Frazier Score,	SDL length,	Type to token ratio lemmatized,	Type to token ratio,	Honoroe's stats,	Automatic Readability Index,	Brunet's index,	Coleman Liau's index")


# Example

```
python main.py transcript.txt output.csv --normalize  # The code will do sentence splitting, casing, spell correction, etc. as preprocessing
```
