# Annotated Whisper 

This is my refactor of the whisper ASR model by OpenAI. When released the whisper repository had a very complicated function call stack and a lot of boilerplate which was unecessary. This repo is the result of my exploration of the original whisper codebase during which I was heavily refactoring and removing parts I found redundant. I also added additional comments for clarity (including the dimensions of tensors at each step).  
- Original Paper [Radford et al. "Robust Speech Recognition via Large-Scale Weak Supervision" 2022.](https://arxiv.org/pdf/2212.04356.pdf)
- Original code [OpenAI/whisper](https://github.com/openai/whisper)

  
<a href=https://arxiv.org/pdf/2212.04356.pdf>
  <p align="center">
    <img width="540" height="700" src="https://github.com/brandokoch/annotated_whisper/assets/57716666/b148b51d-18ab-46bb-b9f4-0d3863cef001">
  </p>
</a>

## Ubuntu Installation 

```bash
sudo apt update && sudo apt install ffmpeg
git clone https://github.com/brandokoch/annotated_whisper
conda create -n annotated_whisper python=3.10 
conda activate annotated_whisper
pip install -r requirements.txt 
```

## Usage
Inference is ran using the `infer.py` script and providing the model type and audio pth. To adjust the inference configuration you can edit the default configuration in `whisper/config.py`. 

```bash
cd repo_dir
python infer.py --model-type medium --audio-pth data/jfk.flac
```


## License
This repository is under an MIT License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/brandokoch/annotated_whisper/blob/master/LICENSE)
