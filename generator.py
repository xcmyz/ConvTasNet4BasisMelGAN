import torch
import torch.nn as nn

import os
import audio
import argparse
import numpy as np
import hparams as hp

from tqdm import tqdm
from analysis import get_DNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def autoencoder(model, wav):
    wav = torch.stack([wav]).to(device)
    with torch.no_grad():
        est_source, weight, weight_ = model.module.autoencode(wav)

    est_wav = est_source[0].cpu().numpy()
    weight = weight[0].cpu().numpy()
    weight_ = weight_[0].cpu().numpy().T
    return est_wav, weight, weight_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--get_test', type=bool, default=False)
    args = parser.parse_args()

    model = get_DNN(args.step)
    weight = model.module.decoder.basis_signals.weight
    weight = weight.detach().numpy().astype(np.float32)
    np.save("basis_signal_weight.npy", weight, allow_pickle=False)
    os.makedirs("weight", exist_ok=True)
    list_filename = list()
    with open(os.path.join("LJSpeech-1.1", 'metadata.csv'), encoding='utf-8') as f:
        for line in f.readlines():
            parts = line.strip().split('|')
            wav_path = os.path.join(
                "LJSpeech-1.1", 'wavs', '%s.wav' % parts[0])
            list_filename.append(wav_path)

    if not args.get_test:
        if hp.gen_size != 0:
            list_filename = list_filename[:hp.gen_size]
    else:
        list_filename = [list_filename[0]] + list_filename[-6:]

    for i in tqdm(range(len(list_filename))):
        wav = audio.load_wav(list_filename[i])
        wav = torch.Tensor(wav)
        wav_, weight, weight_ = autoencoder(model, wav)
        np.save(os.path.join("weight", str(i)+".npy"), weight_)
