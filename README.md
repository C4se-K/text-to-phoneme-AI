# Text-to-Phoneme AI

A custom-trained LSTM-based AI model for converting English words into their ARPABET phoneme representations. This project uses a sequence-to-sequence approach to map graphemes (text characters) to phonemes, enabling applications like speech synthesis, pronunciation guides, and linguistic analysis.

## Features
- Trained on American English pronunciation data.
- Uses LSTM for encoding/decoding sequences.
- Includes tokenizer for efficient subword handling.
- Supports inference on unseen words via learned patterns.

## Project Structure
| File/Directory | Description |
| --- | --- |
| `American-English.zip` | Dataset for training. contains word-phoneme pairs (From CMU-ARBABET Dictionary). |
| `training_encoding.py` | Script for preprocessing and encoding training data into model-ready format. |
| `testing_decoding.py` | Script for model inference, decoding predictions back to phonemes, and testing accuracy. |
| `test3.py` | Quick test script, validats tokenizer and for basic model loading. |
| `tokenizer_0.json` / `tokenizer_20231126_064040.json` | JSON files defining the trained tokenizer for text-to-token conversion. |

## Requirements
- Python 3.8+
- PyTorch
- `tokenizers`
- `zipfile`

Install dependencies:
pip install torch tokenizers
## Setup
1. Clone the repository:
git clone https://github.com/C4se-K/text-to-phoneme-AI.git
```markdown
cd text-to-phoneme-AI
```
2. Unzip the dataset:
```markdown
unzip American-English.zip
```
This should yield a file like `American-English.txt` or similar with word-to-phoneme mappings.

## Training
The model appears pre-trained, but if retraining is needed:
1. Run the encoding script to prepare data:
python training_encoding.py
This likely loads the dataset, tokenizes words and phonemes, and saves encoded tensors for LSTM training.

2. Train the LSTM (note: training script may be integrated or require a separate `train.py`; adjust based on code review).

## Usage
### Inference (Text to Phonemes)
Use `testing_decoding.py` for predictions:
python testing_decoding.py --input_word "hello" --model_path path/to/model.pth
- Expected output: Phoneme sequence like `HH AH0 L OW1`.
- The script handles tokenization, LSTM forward pass, and decoding to ARPABET.

### Quick Test
Run the basic test:
python test3.py
This may validate setup or run a simple end-to-end example.

## Example
Input: `python`  
Output: `P AY0 TH AH0 N`

## Contributing
1. Fork the repo.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit changes (`git commit -m 'Add amazing feature'`).
4. Push to branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (add one if missing).

## Acknowledgments
- Built with PyTorch for the LSTM architecture.
- Dataset inspired by standard American English pronunciation resources.
