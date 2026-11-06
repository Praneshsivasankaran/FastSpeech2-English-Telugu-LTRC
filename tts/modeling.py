"""Small, explicit adapter around the unmodified reference FastSpeech2 modules."""
from __future__ import annotations
import hashlib
import json
import random
import sys
import zipfile
from pathlib import Path
import numpy as np
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
VENDOR = PROJECT / 'vendor' / 'FastSpeech2'
sys.path.insert(0, str(VENDOR))
from model import FastSpeech2
from model.loss import FastSpeech2Loss
import model.modules as modules
import utils.tools as upstream_tools

LOSS_NAMES = ('total', 'mel', 'postnet_mel', 'pitch', 'energy', 'duration')

def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else PROJECT / p

def read_config(path):
    config = yaml.safe_load(Path(path).read_text())
    config['preprocess']['path']['preprocessed_path'] = str(resolve(config['preprocess']['path']['preprocessed_path']))
    return config

def digest(config):
    # Excludes execution-only settings; records the exact architecture/front-end contract.
    contract = {'language': config['language'], 'preprocessing': config['preprocess']['preprocessing'], 'model': config['model']}
    stats = Path(config['preprocess']['path']['preprocessed_path']) / 'stats.json'
    contract['stats'] = json.loads(stats.read_text())
    speakers = stats.parent / 'speakers.json'
    contract['speakers'] = json.loads(speakers.read_text()) if speakers.exists() else None
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def device_for(name='auto'):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device(name)
    if device.type == 'cuda':
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        if device.index is None: device=torch.device('cuda',torch.cuda.current_device())
    if name == 'mps' and not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable')
    # Reference helper globals otherwise allocate masks and length tensors on CPU.
    modules.device = device; upstream_tools.device = device
    return device

def read_vocab(config, path=None):
    p = resolve(path or config['paths']['vocab'])
    vocab = json.loads(p.read_text())
    if isinstance(vocab, list): vocab = {p: i for i,p in enumerate(vocab)}
    if 'phone_to_id' in vocab: vocab = vocab['phone_to_id']
    if 'symbols' in vocab: vocab = {p:i for i,p in enumerate(vocab['symbols'])}
    if not isinstance(vocab, dict) or not all(isinstance(v, int) for v in vocab.values()):
        raise ValueError('vocab must be a list of phones or a phone-to-ID dictionary')
    if len(set(vocab.values())) != len(vocab): raise ValueError('Duplicate vocabulary IDs')
    return vocab

def build_model(config, vocab, device):
    model = FastSpeech2(config['preprocess'], config['model'])
    # Zero is padding; preprocessing must never silently map unknown phones to zero.
    model.encoder.src_word_emb = torch.nn.Embedding(max(vocab.values()) + 1,
        config['model']['transformer']['encoder_hidden'], padding_idx=0)
    return model.to(device)

def load_checkpoint(path, config, vocab, model, device):
    state = torch.load(resolve(path), map_location=device, weights_only=False)
    if state.get('language') != config['language']: raise ValueError('Wrong-language checkpoint')
    if state.get('contract_sha256') != digest(config): raise ValueError('Checkpoint/config mismatch')
    if state.get('vocab') != vocab: raise ValueError('Checkpoint/vocabulary mismatch')
    if state.get('step',0) < 1: raise ValueError('Checkpoint has no completed training steps')
    model.load_state_dict(state['model'], strict=True)
    return state

def load_vocoder(device):
    import hifigan
    path = VENDOR / 'hifigan' / 'generator_universal.pth.tar'
    if not path.exists():
        archive = path.with_suffix(path.suffix + '.zip')
        with zipfile.ZipFile(archive) as z:
            names = [n for n in z.namelist() if Path(n).name == path.name]
            if len(names) != 1: raise ValueError('Unexpected vocoder archive')
            path.write_bytes(z.read(names[0]))
    config = hifigan.AttrDict(json.loads((VENDOR / 'hifigan' / 'config.json').read_text()))
    model = hifigan.Generator(config)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['generator'], strict=True)
    model.eval(); model.remove_weight_norm(); model.to(device)
    return model
