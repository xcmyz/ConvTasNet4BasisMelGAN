import torch

from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import os
import math
import time
import audio

import random
import numpy as np
import hparams as hp

from tqdm import tqdm
from utils import process_text, pad_1D, pad_2D
from utils import pad_1D_tensor, pad_2D_tensor

random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_data_to_buffer():
    buffer = []
    file_list = []
    with open("BZNSYP.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            file_list.append(line.split("/")[-1][:-1])
    file_list_len = len(file_list)
    start = time.perf_counter()
    min_length = 1e9
    for i in tqdm(range(file_list_len)):
        filename = file_list[i]
        mix_filename = f"{filename}.mix.npy"
        noi_filename = f"{filename}.noi.npy"
        wav_filename = f"{filename}.wav.npy"

        mix_filename = os.path.join(hp.dataset_path, mix_filename)
        noi_filename = os.path.join(hp.dataset_path, noi_filename)
        wav_filename = os.path.join(hp.dataset_path, wav_filename)

        mix = np.load(mix_filename)
        noi = np.load(noi_filename)
        wav = np.load(wav_filename)

        mix = torch.from_numpy(mix).float()
        noi = torch.from_numpy(noi).float()
        wav = torch.from_numpy(wav).float()
        target = torch.stack([wav, noi]).float()

        if mix.size(0) < min_length:
            min_length = mix.size(0)

        buffer.append({"mix": mix, "target": target})

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))
    print("min length is {:d}".format(min_length))

    return buffer


class BufferDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        data = self.buffer[idx]
        len_data = data["mix"].size(0)
        start_index = random.randint(0, len_data - hp.fixed_length - 1)
        end_index = start_index + hp.fixed_length
        buffer_cut = {"mix": data["mix"][start_index:end_index], "target": data["target"][:, start_index:end_index]}
        return buffer_cut


def reprocess_tensor(batch, cut_list):
    mixs = [batch[ind]["mix"] for ind in cut_list]
    lengths = [mix.size(0) for mix in mixs]
    mixs = torch.stack(mixs)
    lengths = torch.Tensor(lengths)
    targets = [batch[ind]["target"] for ind in cut_list]
    targets = torch.stack(targets)
    return {"mix": mixs, "target": targets, "length": lengths}


def collate_fn_tensor(batch):
    len_arr = np.array([d["mix"].size(0) for d in batch])
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)
    real_batchsize = batchsize // hp.batch_expand_size
    cut_list = list()
    for i in range(hp.batch_expand_size):
        cut_list.append(index_arr[i * real_batchsize:(i + 1) * real_batchsize])
    output = list()
    for i in range(hp.batch_expand_size):
        output.append(reprocess_tensor(batch, cut_list[i]))

    return output


if __name__ == "__main__":
    # TEST
    get_data_to_buffer()
