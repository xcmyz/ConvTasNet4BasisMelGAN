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
DATASET = "aishell3"  # aishell3 or biaobei


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
    args = parser.parse_args()

    model = get_model(args.step)
    weight = model.module.decoder.basis_signals.weight
    weight = weight.detach().cpu().numpy().astype(np.float32)
    os.makedirs("Basis-MelGAN-dataset")
    np.save(os.path.join("Basis-MelGAN-dataset", "basis_signal_weight.npy"), weight, allow_pickle=False)
    os.makedirs(os.path.join("Basis-MelGAN-dataset", "generated"), exist_ok=True)
    os.makedirs(os.path.join("Basis-MelGAN-dataset", "weight"), exist_ok=True)
    list_filename = list()
    if DATASET == "biaobei":
        path_filename = "BZNSYP.txt"
    elif DATASET == "aishell3":
        path_filename = "aishell3.txt"
    with open(path_filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
        len_lines = len(lines)
        for i in tqdm(range(len_lines)):
            line = lines[i]
            line = line[:-1]
            wav = audio.load_wav(line, sample_rate=hp.sample_rate, encode=False)
            wav = torch.Tensor(wav)
            wav_, weight, weight_ = autoencoder(model, wav)
            filename = line.split("/")[-1]
            audio.save_wav(wav_, os.path.join("Basis-MelGAN-dataset", "generated", filename), hp.sample_rate)
            np.save(os.path.join("Basis-MelGAN-dataset", "weight", f"{filename}.npy"), weight_)
