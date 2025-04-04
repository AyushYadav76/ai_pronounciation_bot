import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from phonemizer import phonemize
import difflib

# Load pretrained model & processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
model.eval()

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    return waveform.squeeze()

def transcribe(audio_tensor):
    inputs = processor(audio_tensor, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription.lower()

def to_phonemes(text):
    return phonemize(text, language='en-us', backend='espeak', strip=True).upper()

def compare_phonemes(expected, spoken):
    seq = difflib.SequenceMatcher(None, expected.split(), spoken.split())
    feedback = []
    for opcode in seq.get_opcodes():
        tag, i1, i2, j1, j2 = opcode
        if tag != 'equal':
            feedback.append(f"Expected {' '.join(expected.split()[i1:i2])} but got {' '.join(spoken.split()[j1:j2])}")
    return feedback

def evaluate_pronunciation(audio_path, expected_text):
    audio = load_audio(audio_path)
    spoken_text = transcribe(audio)
    expected_phonemes = to_phonemes(expected_text)
    spoken_phonemes = to_phonemes(spoken_text)
    feedback = compare_phonemes(expected_phonemes, spoken_phonemes)
    return {
        "expected": expected_phonemes,
        "spoken": spoken_phonemes,
        "spoken_text": spoken_text,
        "feedback": feedback
    }

# Example usage:
if __name__ == "__main__":
    path = "sample.wav"
    expected = "through"
    result = evaluate_pronunciation(path, expected)
    print(result)
