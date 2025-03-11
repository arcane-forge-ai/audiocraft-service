from fastapi import APIRouter, HTTPException
from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen, MultiBandDiffusion
from audiocraft.models.encodec import InterleaveStereoCompressionModel
from einops import rearrange
import torch
import logging
import time
from tempfile import NamedTemporaryFile
from dependencies.file_cleaner import FileCleaner
from dependencies.filestore.blob_storage import upload_to_azure_blob
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

USE_DIFFUSION = False
DEFAULT_MODEL = MusicGen.get_pretrained(
    'facebook/musicgen-stereo-melody-large')
MODEL = None
MBD = None

AVAILABLE_MODELS = [
    "facebook/musicgen-melody", "facebook/musicgen-medium",
    "facebook/musicgen-small", "facebook/musicgen-large",
    "facebook/musicgen-melody-large", "facebook/musicgen-stereo-small",
    "facebook/musicgen-stereo-medium", "facebook/musicgen-stereo-melody",
    "facebook/musicgen-stereo-large", "facebook/musicgen-stereo-melody-large"
]

# # We have to wrap subprocess call to clean a bit the log when using gr.make_waveform
# _old_call = subprocess.call

# def _call_nostderr(*args, **kwargs):
#     # Avoid ffmpeg vomiting on the logs.
#     kwargs['stderr'] = subprocess.DEVNULL
#     kwargs['stdout'] = subprocess.DEVNULL
#     _old_call(*args, **kwargs)

# subprocess.call = _call_nostderr
# # Preallocating the pool of processes.
# pool = ProcessPoolExecutor(4)
# pool.__enter__()

file_cleaner = FileCleaner()


@router.post("/generate")
async def generate(prompt: str,
                   model: str,
                   decoder: str = "MultiBand_Diffusion",
                   melody: str = None,
                   duration: int = 10,
                   topk: int = 250,
                   topp: float = 0,
                   temperature: float = 1.0,
                   cfg_coef: float = 3.0,
                   user_id: str = None):
    if model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400,
                            detail=f"Model {model} not found.")

    if temperature < 0:
        raise HTTPException(status_code=400,
                            detail="Temperature must be >= 0.")
    if topk < 0:
        raise HTTPException(status_code=400,
                            detail="Topk must be non-negative.")
    if topp < 0:
        raise HTTPException(status_code=400,
                            detail="Topp must be non-negative.")

    if user_id is None:
        user_id = "anonymous"

    logger.info(
        f"Starting music generation with parameters: text='{prompt}', model='{model}', duration={duration}\n"
        f"decoder={decoder}, melody={melody}, duration={duration}, topk={topk}, topp={topp}, temperature={temperature}, cfg_coef={cfg_coef}, user_id={user_id}"
    )

    topk = int(topk)
    if decoder == "MultiBand_Diffusion":
        USE_DIFFUSION = True
        logger.info("Loading diffusion model...")
        load_diffusion()
    else:
        USE_DIFFUSION = False
    load_model(model)

    max_generated = 0

    videos, wavs = _do_predictions(
        [prompt],
        [melody],
        duration,
        progress=True,
        use_diffusion=USE_DIFFUSION,
        top_k=topk,
        top_p=topp,
        temperature=temperature,
        cfg_coef=cfg_coef,
    )

    # Generate a unique prefix for the blob path based on timestamp in YYYY/MM/DD/HH/mm/SS format
    now = datetime.now()
    timestamp_path = now.strftime("%Y/%m/%d/%H/%M/%S")
    unique_id = str(uuid.uuid4())[:16]
    blob_prefix = f"{user_id}/music_gen/{timestamp_path}/{unique_id}"

    # Upload original wav and video files (for both diffusion and non-diffusion)
    wav_blob_path = upload_file(blob_prefix, wavs[0], "wav")

    # TODO: temporary disable video generation
    # video_blob_path = upload_file(blob_prefix, videos[0], "mp4")

    # If using diffusion, upload the additional diffusion files
    if USE_DIFFUSION:
        diff_wav_blob_path = upload_file(blob_prefix, wavs[1], "wav",
                                         "diffusion")

        # TODO: temporary disable video generation
        # diff_video_blob_path = upload_file(blob_prefix, videos[1], "mp4",
        #                                    "diffusion")

        return wav_blob_path, diff_wav_blob_path, None, None
    else:
        return wav_blob_path, None, None, None


@router.get("/models")
async def get_available_models():
    return {"models": AVAILABLE_MODELS}


def load_diffusion():
    global MBD
    if MBD is None:
        logger.info("loading MBD")
        MBD = MultiBandDiffusion.get_mbd_musicgen()


def load_model(version='facebook/musicgen-stereo-melody-large'):
    global MODEL
    logger.info(f"Loading model {version}")
    if MODEL is None or MODEL.name != version:
        # Clear PyTorch CUDA cache and delete model
        del MODEL
        torch.cuda.empty_cache()
        MODEL = None  # in case loading would crash
        MODEL = MusicGen.get_pretrained(version)


def _do_predictions(texts, melodies, duration, progress=False, use_diffusion=False, **gen_kwargs):
    MODEL.set_generation_params(duration=duration, **gen_kwargs)
    logger.info(
        f"new batch {len(texts)} {texts} {[None if m is None else (m[0], m[1].shape) for m in melodies]}"
    )
    beginning_time = time.time()
    processed_melodies = []
    target_sample_rate = 32000
    target_ac = 1
    for melody in melodies:
        if melody is None:
            processed_melodies.append(None)
        else:
            sr, melody = melody[0], torch.from_numpy(melody[1]).to(
                MODEL.device).float().t()
            if melody.dim() == 1:
                melody = melody[None]
            melody = melody[..., :int(sr * duration)]
            melody = convert_audio(melody, sr, target_sample_rate, target_ac)
            processed_melodies.append(melody)

    try:
        if any(m is not None for m in processed_melodies):
            outputs = MODEL.generate_with_chroma(
                descriptions=texts,
                melody_wavs=processed_melodies,
                melody_sample_rate=target_sample_rate,
                progress=progress,
                return_tokens=use_diffusion)
        else:
            outputs = MODEL.generate(texts,
                                     progress=progress,
                                     return_tokens=use_diffusion)
    except RuntimeError as e:
        raise HTTPException(status_code=500,
                            detail="Error while generating " + e.args[0])
    if use_diffusion:
        tokens = outputs[1]
        if isinstance(MODEL.compression_model,
                      InterleaveStereoCompressionModel):
            left, right = MODEL.compression_model.get_left_right_codes(tokens)
            tokens = torch.cat([left, right])
        outputs_diffusion = MBD.tokens_to_wav(tokens)
        if isinstance(MODEL.compression_model,
                      InterleaveStereoCompressionModel):
            assert outputs_diffusion.shape[1] == 1  # output is mono
            outputs_diffusion = rearrange(outputs_diffusion,
                                          '(s b) c t -> b (s c) t',
                                          s=2)
        outputs = torch.cat([outputs[0], outputs_diffusion], dim=0)
    outputs = outputs.detach().cpu().float()
    # pending_videos = []
    out_wavs = []
    for output in outputs:
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(file.name,
                        output,
                        MODEL.sample_rate,
                        strategy="loudness",
                        loudness_headroom_db=16,
                        loudness_compressor=True,
                        add_suffix=False)
            out_wavs.append(file.name)
            file_cleaner.add(file.name)
    # out_videos = [pending_video.result() for pending_video in pending_videos]
    # for video in out_videos:
    #     file_cleaner.add(video)
    logger.info(f"batch finished {len(texts)} {time.time() - beginning_time}")
    logger.info(f"Tempfiles currently stored: {len(file_cleaner.files)}")
    # return [out_videos], out_wavs
    return [], out_wavs


# Helper function to upload a file and return its blob path
def upload_file(blob_prefix, file_path, file_type, variant="original"):
    if file_path:
        blob_path = f"{blob_prefix}_{variant}.{file_type}"
        upload_to_azure_blob(blob_path, file_path)
        return blob_path
    return None
