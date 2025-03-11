# Audiocraft Service

This service provides a FastAPI wrapper around Facebook's AudioCraft models for music generation. It exposes endpoints for generating music from text prompts and melodies.

## Features

- Text-to-music generation using MusicGen models
- Support for melody-conditioned generation 
- Multiple model size options (small, medium, large)
- Stereo and mono output options
- Optional MultiBand Diffusion decoder
- Azure Blob Storage integration for output files
- Configurable generation parameters (temperature, top-k, top-p, etc.)

## API Endpoints

### POST /generate
Generate music from a text prompt and optional melody.

Parameters:
- `prompt` (str): Text description of the desired music
- `model` (str): MusicGen model to use
- `decoder` (str, optional): "MultiBand_Diffusion" or default. Default: "MultiBand_Diffusion"
- `melody` (str, optional): Path to melody audio file
- `duration` (int, optional): Duration in seconds. Default: 10
- `topk` (int, optional): Top-k sampling parameter. Default: 250
- `topp` (float, optional): Top-p sampling parameter. Default: 0
- `temperature` (float, optional): Sampling temperature. Default: 1.0
- `cfg_coef` (float, optional): Classifier-free guidance coefficient. Default: 3.0
- `user_id` (str, optional): User identifier for output path

Returns paths to generated audio files in Azure Blob Storage.

### GET /models
List available MusicGen models.

Returns list of model IDs that can be used with the generate endpoint.

## Running the Service

### Prerequisites
- Python 3.9+
- PyTorch 2.1.0 (MUST)
- CUDA-capable GPU (recommended), be sure to follow [this link](https://pytorch.org/get-started/previous-versions/#linux-and-windows-22) for installation
- FastAPI stuffs (tried my best to put everything under [fastapi_requirements.txt](./fastapi_requirements.txt))
- Azure Blob Storage account and credentials

### Run
```shell
cd service && python main.py
```

---

# Below contains original AudioCraft Readme

# AudioCraft
![docs badge](https://github.com/facebookresearch/audiocraft/workflows/audiocraft_docs/badge.svg)
![linter badge](https://github.com/facebookresearch/audiocraft/workflows/audiocraft_linter/badge.svg)
![tests badge](https://github.com/facebookresearch/audiocraft/workflows/audiocraft_tests/badge.svg)

AudioCraft is a PyTorch library for deep learning research on audio generation. AudioCraft contains inference and training code
for two state-of-the-art AI generative models producing high-quality audio: AudioGen and MusicGen.


## Installation
AudioCraft requires Python 3.9, PyTorch 2.1.0. To install AudioCraft, you can run the following:

```shell
# Best to make sure you have torch installed first, in particular before installing xformers.
# Don't run this if you already have PyTorch installed.
python -m pip install 'torch==2.1.0'
# You might need the following before trying to install the packages
python -m pip install setuptools wheel
# Then proceed to one of the following
python -m pip install -U audiocraft  # stable release
python -m pip install -U git+https://git@github.com/facebookresearch/audiocraft#egg=audiocraft  # bleeding edge
python -m pip install -e .  # or if you cloned the repo locally (mandatory if you want to train).
python -m pip install -e '.[wm]'  # if you want to train a watermarking model
```

We also recommend having `ffmpeg` installed, either through your system or Anaconda:
```bash
sudo apt-get install ffmpeg
# Or if you are using Anaconda or Miniconda
conda install "ffmpeg<5" -c conda-forge
```

## Models

At the moment, AudioCraft contains the training code and inference code for:
* [MusicGen](./docs/MUSICGEN.md): A state-of-the-art controllable text-to-music model.
* [AudioGen](./docs/AUDIOGEN.md): A state-of-the-art text-to-sound model.
* [EnCodec](./docs/ENCODEC.md): A state-of-the-art high fidelity neural audio codec.
* [Multi Band Diffusion](./docs/MBD.md): An EnCodec compatible decoder using diffusion.
* [MAGNeT](./docs/MAGNET.md): A state-of-the-art non-autoregressive model for text-to-music and text-to-sound.
* [AudioSeal](./docs/WATERMARKING.md): A state-of-the-art audio watermarking.
* [MusicGen Style](./docs/MUSICGEN_STYLE.md): A state-of-the-art text-and-style-to-music model.
* [JASCO](./docs/JASCO.md): "High quality text-to-music model conditioned on chords, melodies and drum tracks"


## Training code

AudioCraft contains PyTorch components for deep learning research in audio and training pipelines for the developed models.
For a general introduction of AudioCraft design principles and instructions to develop your own training pipeline, refer to
the [AudioCraft training documentation](./docs/TRAINING.md).

For reproducing existing work and using the developed training pipelines, refer to the instructions for each specific model
that provides pointers to configuration, example grids and model/task-specific information and FAQ.


## API documentation

We provide some [API documentation](https://facebookresearch.github.io/audiocraft/api_docs/audiocraft/index.html) for AudioCraft.


## FAQ

#### Is the training code available?

Yes! We provide the training code for [EnCodec](./docs/ENCODEC.md), [MusicGen](./docs/MUSICGEN.md),[Multi Band Diffusion](./docs/MBD.md) and [JASCO](./docs/JASCO.md).

#### Where are the models stored?

Hugging Face stored the model in a specific location, which can be overridden by setting the `AUDIOCRAFT_CACHE_DIR` environment variable for the AudioCraft models.
In order to change the cache location of the other Hugging Face models, please check out the [Hugging Face Transformers documentation for the cache setup](https://huggingface.co/docs/transformers/installation#cache-setup).
Finally, if you use a model that relies on Demucs (e.g. `musicgen-melody`) and want to change the download location for Demucs, refer to the [Torch Hub documentation](https://pytorch.org/docs/stable/hub.html#where-are-my-downloaded-models-saved).


## License
* The code in this repository is released under the MIT license as found in the [LICENSE file](LICENSE).
* The models weights in this repository are released under the CC-BY-NC 4.0 license as found in the [LICENSE_weights file](LICENSE_weights).


## Citation

For the general framework of AudioCraft, please cite the following.
```
@inproceedings{copet2023simple,
    title={Simple and Controllable Music Generation},
    author={Jade Copet and Felix Kreuk and Itai Gat and Tal Remez and David Kant and Gabriel Synnaeve and Yossi Adi and Alexandre Défossez},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
}
```

When referring to a specific model, please cite as mentioned in the model specific README, e.g
[./docs/MUSICGEN.md](./docs/MUSICGEN.md), [./docs/AUDIOGEN.md](./docs/AUDIOGEN.md), etc.
