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
    wav = torch.Tensor(audio.load_wav(in_path))
    noi = audio.add_noise(wav, quantization_channel=hp.quantization_channel)
    mix = wav.float() + noi

    wav_name = 'wav-%05d.npy' % index
    noi_name = 'noi-%05d.npy' % index
    mix_name = "mix-%05d.npy" % index

    np.save(os.path.join(out_path, wav_name), wav.numpy(), allow_pickle=False)
    np.save(os.path.join(out_path, noi_name), noi.numpy(), allow_pickle=False)
    np.save(os.path.join(out_path, mix_name), mix.numpy(), allow_pickle=False)


if __name__ == "__main__":
    futures = list()
    os.makedirs(hp.dataset_path, exist_ok=True)
    executor = ProcessPoolExecutor(max_workers=cpu_count())

    file_list = os.listdir(os.path.join("BZNSYP", "Wave"))
    if hp.test_size != 0:
        file_list = file_list[:hp.test_size]
    # for index, filename in enumerate(file_list):
    for index in tqdm(range(len(file_list))):
        filename = file_list[index]
        in_path = os.path.join("BZNSYP", "Wave", filename)
        # futures.append(executor.submit(
        #     partial(_process_utterance, in_path, hp.dataset_path, index+1)))
        _process_utterance(in_path, hp.dataset_path, index + 1)

        # if (index + 1) % 100 == 0:
        #     print("Done", index + 1)

    # [future.result() for future in tqdm(futures)]
