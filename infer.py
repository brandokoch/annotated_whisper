import torch 
import json
import numpy as np
import random
import os
import argparse

from whisper import Whisper
from whisper import load_audio
from whisper import Configuration

# Seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

CONFIG = Configuration()

def main():

    parser = argparse.ArgumentParser(description="Whisper")
    parser.add_argument("--model-type", type=str, default = "tiny", help="Model type")
    parser.add_argument("--audio-pth", type=str, help="Path to audio file")
    args = parser.parse_args()
    model_type = args.model_type
    audio_pth = args.audio_pth

    model = Whisper(model_type, CONFIG)

    assert os.path.exists(audio_pth), f"Audio file not found: {audio_pth}"
    audio = load_audio(audio_pth, sr = 16000)

    result = model.transcribe(audio)

    # Save result to json
    with open("result.json", "w") as f:
        json.dump(result, f, indent = 4)
        print("Result saved to result.json")

    print("Transcription: ", result["text"])

if __name__ == "__main__":
    main()