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

random.seed(str(time.time()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_data_to_buffer():
    buffer = list()
    file_length = len(os.listdir(hp.dataset_path)) // 3
    if hp.test_size != 0:
        file_length = hp.test_size
    start = time.perf_counter()

    min_length = 1e9
    # print(min_length)
    for i in tqdm(range(file_length)):
        mix_filename = "mix-%05d.npy" % (i + 1)
        noi_filename = "noi-%05d.npy" % (i + 1)
        wav_filename = "wav-%05d.npy" % (i + 1)

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

    # min length of waveform is 8881 (sample rate: 8000)
    # min length of waveform is 24477 (sample rate: 22050)

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
        buffer_cut = {"mix": data["mix"][start_index:end_index],
                      "target": data["target"][:, start_index:end_index]}
        # buffer_cut = {"mix": data["mix"], "target": data["target"]}

        # buffer_cut = {"mix": data["mix"], "target": data["target"]}
        # audio.save_wav(buffer_cut["mix"].float().numpy(), os.path.join("test", "mix.wav"))
        # audio.save_wav(buffer_cut["target"][0].float().numpy(), os.path.join("test", "wav.wav"))
        # audio.save_wav(buffer_cut["target"][1].float().numpy(), os.path.join("test", "noi.wav"))

        if random.random() <= hp.clean_p:
            buffer_cut["mix"] = buffer_cut["target"][0]
            buffer_cut["target"] = torch.cat(
                [buffer_cut["target"][:1, :], torch.zeros(1, buffer_cut["target"].size(1))])

            # audio.save_wav(buffer_cut["mix"].float().numpy(), os.path.join("test", "mix_.wav"))
            # audio.save_wav(buffer_cut["target"][0].float().numpy(), os.path.join("test", "wav_.wav"))
            # audio.save_wav(buffer_cut["target"][1].float().numpy(), os.path.join("test", "noi_.wav"))

        return buffer_cut


def reprocess_tensor(batch, cut_list):
    mixs = [batch[ind]["mix"] for ind in cut_list]
    lengths = [mix.size(0) for mix in mixs]

    # mixs = pad_1D_tensor(mixs)
    mixs = torch.stack(mixs)
    lengths = torch.Tensor(lengths)

    # targets = [batch[ind]["target"].transpose(0, 1) for ind in cut_list]
    targets = [batch[ind]["target"] for ind in cut_list]
    # targets = pad_2D_tensor(targets)
    # targets = targets.transpose(1, 2)
    targets = torch.stack(targets)

    return {"mix": mixs, "target": targets, "length": lengths}


def collate_fn_tensor(batch):
    len_arr = np.array([d["mix"].size(0) for d in batch])
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)
    real_batchsize = batchsize // hp.batch_expand_size

    cut_list = list()
    for i in range(hp.batch_expand_size):
        cut_list.append(
            index_arr[i * real_batchsize:(i + 1) * real_batchsize])

    output = list()
    for i in range(hp.batch_expand_size):
        output.append(reprocess_tensor(batch, cut_list[i]))

    return output


if __name__ == "__main__":
    # TEST
    get_data_to_buffer()
