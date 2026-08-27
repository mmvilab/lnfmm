# STATUS: written but not yet run/validated. Smoke-test with --num_images 63
# before trusting output. rev_x/rev_z unpacking order verified against
# FlowLatent.reverse_flow() source, but full pipeline untested end-to-end.

import torch, json, pickle, os, argparse
import numpy as np
import nltk
import torchvision.transforms as transforms
from pycocotools.coco import COCO

from modules.text_modules import TextEncoder
from modules.latent_align_modules import FlowLatent
from modules.SaGAN import Generator
from utils.build_vocab_coco import Vocabulary

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--config', default='params_t2i')
parser.add_argument('--outdir', required=True)
parser.add_argument('--num_images', type=int, default=1000)
args = parser.parse_args()

config = json.loads(open('params.json').read())[args.config]
img_dim = int(config['img_dim'])
noise_im = int(config['noise_im'])
embed_size = int(config['embed_size'])
word_dim = int(config['word_dim'])
max_length = int(config['max_length'])
num_decoder_tokens = int(config['num_decoder_tokens'])
batch_size = int(config['batch_size'])
vocab_path = config['vocab_path']

device = torch.device('cuda')

vocab = pickle.load(open(vocab_path, 'rb'))
vocab_w2i = vocab.word2idx

image_decoder = Generator(z_dim=noise_im, g_conv_dim=32, t2i_dim=img_dim).to(device)
txtEncoder = TextEncoder(batch_size, word_dim, embed_size, num_decoder_tokens).to(device)
flow_latent_align = FlowLatent(batch_size, img_dim, hidden_channels=1024, K=12,
                                gaussian_dims=0, gaussian_var=0, coupling='linear').to(device)
flow_latent_image = FlowLatent(batch_size, noise_im, hidden_channels=512,
                                gaussian_dims=noise_im, gaussian_var=0.25, cond_dim=img_dim,
                                coupling='full').to(device)

ckpt = torch.load(args.checkpoint, map_location=device)
image_decoder.load_state_dict(ckpt['image_decoder_sd'])
txtEncoder.load_state_dict(ckpt['txtEncoder_sd'])
flow_latent_align.load_state_dict(ckpt['flow_latent_align_sd'])
flow_latent_image.load_state_dict(ckpt['flow_latent_image_sd'])

image_decoder.eval(); txtEncoder.eval()
flow_latent_align.eval(); flow_latent_image.eval()

coco = COCO('annotations/captions_val2014.json')
ann_ids = list(coco.anns.keys())[:args.num_images]

def tokenize_caption(cap):
    tokens = nltk.tokenize.word_tokenize(str(cap).lower())
    seq = [vocab_w2i['<start>']]
    seq.extend([vocab_w2i[t] for t in tokens if t in vocab_w2i])
    seq = seq[:max_length-1]
    seq.append(vocab_w2i['<end>'])
    return seq

os.makedirs(args.outdir, exist_ok=True)

with torch.no_grad():
    for batch_start in range(0, len(ann_ids), batch_size):
        batch_ann_ids = ann_ids[batch_start:batch_start+batch_size]
        if len(batch_ann_ids) < batch_size:
            break

        anns = coco.loadAnns(batch_ann_ids)
        captions = [a['caption'] for a in anns]
        img_ids = [a['image_id'] for a in anns]

        seqs = [tokenize_caption(c) for c in captions]
        lengths = np.array([len(s) for s in seqs])
        sort_order = np.argsort(lengths)[::-1]

        seq_arr = np.zeros((batch_size, max_length))
        for idx, a in enumerate(sort_order):
            seq_arr[idx, :lengths[a]] = seqs[a]
        seq_tensor = torch.LongTensor(seq_arr.astype(np.int64)).to(device)
        len_tensor = torch.LongTensor(lengths[sort_order].astype(np.int64)).to(device)

        txtencoded_hidden = txtEncoder(seq_tensor, len_tensor)
        z, nll, _ = flow_latent_align(x=txtencoded_hidden[:, :img_dim], z_im=None, z=None,
                                       cond=None, eps_std=None, reverse=False)
        z_im_text2img = z[:, :img_dim]

        rev_x, rev_z = flow_latent_image(x=None, z=None, cond=z_im_text2img, reverse=True, eps_std=1.0)
        generated = image_decoder(t2i=z_im_text2img, z=rev_x)

        generated = (generated.clamp(-1, 1) + 1) / 2

        for idx, a in enumerate(sort_order):
            img_pil = transforms.ToPILImage()(generated[idx].detach().cpu())
            img_pil.save(os.path.join(args.outdir, f'{img_ids[a]}.png'))

        print(f'Generated {batch_start+batch_size}/{len(ann_ids)}')
