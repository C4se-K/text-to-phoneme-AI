# Text-to-Phoneme LSTM AI

A custom-trained LSTM-based AI model for converting English words into their ARPABET phoneme representations. This project uses a sequence-to-sequence approach to map graphemes (text characters) to phonemes, enabling applications like speech synthesis, pronunciation guides, and linguistic analysis.

## Features
- Trained on American English pronunciation data.
- Uses LSTM for encoding/decoding sequences.
- Includes tokenizer for efficient subword handling.
- Supports inference on unseen words via learned patterns.

## Requirements
- Python 3.8+
- PyTorch==2.13.0 (no longer supported)
- keras==2.13.1 (depricated)
- transformers
- `tokenizers`

installation
```markdown
pip install -r requirements.txt
```

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

## Training
Train the LSTM
run `training_encoding.py`. it outputs a trained model.
```markdown
python training_encoding.py
```

## Usage
### Inference (Text to Phonemes)
Use `testing_decoding.py` 
for predictions:
1. run `testing_decoding.py`.
- loops until 'end' is typed.
- prints out the Phoneme sequence of words typed.
2. enter any word that you want to convert. 
- expected output: Phoneme sequence like `HH AH0 L OW1`.
- the script handles tokenization, LSTM forward pass, and decoding to ARPABET.

## Example
Input: `python`  
Output: `P AY TH AH N`

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
