import torch

import os
import audio
import random
import numpy as np
import hparams as hp

from tqdm import tqdm
from functools import partial
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

DATASET = "aishell3"  # aishell3 or biaobei


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
    if DATASET == "biaobei":
        with open("BZNSYP.txt", "w", encoding="utf-8") as f:
            for filename in os.listdir(os.path.join("BZNSYP", "Wave")):
                if filename[0] != ".":
                    f.write(os.path.abspath(os.path.join("BZNSYP", "Wave", filename)) + "\n")
    elif DATASET == "aishell3":
        cnt = 0
        with open("aishell3.txt", "w", encoding="utf-8") as f:
            files = []
            wav_path = os.path.join("data_aishell3", "train", "wav")
            for speaker_name in os.listdir(wav_path):
                path = os.path.join(wav_path, speaker_name)
                for wav_name in os.listdir(path):
                    cnt += 1
                    files.append(os.path.abspath(os.path.join(path, wav_name)))
            wav_path = os.path.join("data_aishell3", "test", "wav")
            for speaker_name in os.listdir(wav_path):
                path = os.path.join(wav_path, speaker_name)
                for wav_name in os.listdir(path):
                    cnt += 1
                    files.append(os.path.abspath(os.path.join(path, wav_name)))
            print(f"load {cnt} files.")
            files = random.sample(files, hp.dataset_size)
            for file in files:
                f.write(f"{file}\n")


if __name__ == "__main__":
    # Get path in a file
    get_pathfile()
    os.makedirs(hp.dataset_path, exist_ok=True)
    filename = "BZNSYP.txt" if DATASET == "biaobei" else "aishell3.txt"
    with open(filename, "r", encoding="utf-8") as f:
        paths = f.readlines()
        length = len(paths)
        for i in tqdm(range(length)):
            path = paths[i]
            path = path[:-1]
            index = path.split("/")[-1]
            _process_utterance(os.path.join(path), hp.dataset_path, index)
