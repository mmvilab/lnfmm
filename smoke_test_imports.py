import torch
print(torch.__version__)

import torch.nn as nn
try:
    nll = nn.NLLLoss(size_average=False, ignore_index=0)
    print("NLLLoss(size_average=...) OK — you're on an old-enough PyTorch")
except TypeError as e:
    print("FAILS on this PyTorch version:", e)

try:
    from pycocotools.coco import COCO
    print("pycocotools OK")
except ImportError as e:
    print("pycocotools missing/broken:", e)

import nltk
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    print("nltk punkt + punkt_tab present")
except LookupError:
    print("Run nltk.download('punkt') and nltk.download('punkt_tab') NOW")

from modules.text_modules import TextEncoder, TextDecoder
from modules.latent_align_modules import FlowLatent, GaussianDiag
from modules.SaGAN import Generator, Discriminator
from modules.image_modules import ImageEncoder
from utils.build_vocab_coco import Vocabulary
from utils.custom_cococaptions import CocoCaptions
print("all repo modules import cleanly")
