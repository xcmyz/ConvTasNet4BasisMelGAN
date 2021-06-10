import torch
import torch.nn as nn

import os
import audio
import argparse
import numpy as np
import hparams as hp

from tqdm import tqdm
from analysis import get_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def autoencoder(model, wav):
    wav = torch.stack([wav]).to(device)
    with torch.no_grad():
        est_source, weight, weight_ = model.module.autoencode(wav)
        if False:
            test_wav = model.module.vocoder(weight_)
            audio.save_wav(test_wav[0].numpy(), "test_.wav", hp.sample_rate)

    est_wav = est_source[0].cpu().numpy()
    weight = weight[0].cpu().numpy()
    weight_ = weight_[0].cpu().numpy().T
    return est_wav, weight, weight_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=0)
    args = parser.parse_args()

    model = get_model(args.step)
    weight = model.module.decoder.basis_signals.weight
    weight = weight.detach().numpy().astype(np.float32)
    np.save("basis_signal_weight.npy", weight, allow_pickle=False)
    os.makedirs("generated", exist_ok=True)
    os.makedirs("weight", exist_ok=True)
    list_filename = list()
    with open("BZNSYP.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        len_lines = len(lines)
        for i in tqdm(range(len_lines)):
            line = lines[i]
            line = line[:-1]
            wav = audio.load_wav(line, sample_rate=hp.sample_rate, encode=False)
            wav = torch.Tensor(wav)
            wav_, weight, weight_ = autoencoder(model, wav)
            filename = line.split("/")[-1]
            audio.save_wav(wav_, os.path.join("generated", filename), hp.sample_rate)
            np.save(os.path.join("weight", f"{filename}.npy"), weight_)
