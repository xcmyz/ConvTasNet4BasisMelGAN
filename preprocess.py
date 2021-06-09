import torch

import os
import audio
import numpy as np
import hparams as hp

from tqdm import tqdm
from functools import partial
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor


def _process_utterance(in_path, out_path, index):
    wav = torch.Tensor(audio.load_wav(in_path, hp.sample_rate, encode=False))
    noi = audio.add_noise(wav, quantization_channel=hp.quantization_channel)
    mix = wav.float() + noi

    wav_name = f"{index}.wav.npy"
    noi_name = f"{index}.noi.npy"
    mix_name = f"{index}.mix.npy"

    np.save(os.path.join(out_path, wav_name), wav.numpy(), allow_pickle=False)
    np.save(os.path.join(out_path, noi_name), noi.numpy(), allow_pickle=False)
    np.save(os.path.join(out_path, mix_name), mix.numpy(), allow_pickle=False)


def get_pathfile():
    with open("BZNSYP.txt", "w", encoding="utf-8") as f:
        for filename in os.listdir(os.path.join("BZNSYP", "Wave")):
            f.write(os.path.abspath(os.path.join("BZNSYP", "Wave", filename)) + "\n")


if __name__ == "__main__":
    # Get path in a file
    get_pathfile()
    os.makedirs(hp.dataset_path, exist_ok=True)
    with open("BZNSYP.txt", "r", encoding="utf-8") as f:
        paths = f.readlines()
        length = len(paths)
        for i in tqdm(range(length)):
            path = paths[i]
            path = path[:-1]
            index = path.split("/")[-1]
            _process_utterance(os.path.join(path), hp.dataset_path, index)
