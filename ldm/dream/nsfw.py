import logging
import os
import numpy as np

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
# get rid of the annoying black image message lol
logging.getLogger(StableDiffusionSafetyChecker.__module__).addFilter(lambda record: 0)

from diffusers.utils import DIFFUSERS_CACHE
from transformers import CLIPFeatureExtractor
from PIL import Image
from huggingface_hub import snapshot_download

# to get the safety checker, unfortunately we need to use
# more disk space to download the other version of this model
# if it isnt already downloaded you can set this to true
DOWNLOAD = False

cache = snapshot_download(
    "CompVis/stable-diffusion-v1-4",
    cache_dir=DIFFUSERS_CACHE,
    resume_download=False,
    proxies=None,
    local_files_only=not DOWNLOAD,
    use_auth_token=DOWNLOAD,
    revision=None
)

safety_checker = StableDiffusionSafetyChecker.from_pretrained(os.path.join(cache, "safety_checker"))
feature_extractor = CLIPFeatureExtractor.from_pretrained(os.path.join(cache, "feature_extractor"))

def check_nsfw(images: [Image]):
    imnp = [np.array(im) / 255 for im in images]

    features = feature_extractor(images, return_tensors="pt")
    _, nsfw = safety_checker(images=imnp, clip_input=features.pixel_values)

    return nsfw
