import os
import torch
import torch.nn as nn
import argparse
import numpy as np
import random
import time
import shutil
import librosa
import scipy.io.wavfile

import matplotlib
import matplotlib.pyplot as plt
import hparams as hp
import utils
import dataset
import tasnet
import audio

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

random.seed(str(time.time()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_data(data, filename, figsize=(24, 4)):
    _, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto',
                       origin='upper', interpolation='none')

    os.makedirs("image", exist_ok=True)
    plt.savefig(os.path.join("image", filename))


def get_DNN(num):
    checkpoint_path = "checkpoint_" + str(num) + ".pth.tar"
    model = nn.DataParallel(
        tasnet.ConvTasNet(N=hp.N, L=hp.L,
                          B=hp.B, H=hp.H,
                          P=hp.P, X=hp.X,
                          R=hp.R, C=hp.C,
                          norm_type=hp.norm_type,
                          causal=hp.causal,
                          mask_nonlinear=hp.mask_nonlinear)).to(device)
    model.load_state_dict(
        torch.load(os.path.join(hp.checkpoint_path, checkpoint_path),
                   map_location=torch.device(device))['model'])
    # torch.save({'model': model.state_dict()}, "checkpoint_270000.pth.tar", _use_new_zipfile_serialization=False)
    model.eval()
    return model


def get_file_list():
    ljspeech_path = os.path.join("LJSpeech-1.1")
    wavs_path = os.path.join(ljspeech_path, "wavs")
    file_list = os.listdir(wavs_path)
    out_file_list = random.sample(file_list, 3)
    # print(out_file_list)
    return out_file_list, wavs_path


# def get_file_list():
#     wavs_path = os.path.join("for_test")
#     out_file_list = ["1.wav", "2.wav", "3.wav"]
#     # print(out_file_list)
#     return out_file_list, wavs_path


def test(model, mix):
    mix = torch.stack([mix]).to(device)
    with torch.no_grad():
        est_source, weight = model.module.test(mix)

    est_wav = est_source[0][1].cpu().numpy()
    est_noi = est_source[0][0].cpu().numpy()
    weight = weight[0].cpu().numpy()
    return est_noi, est_wav, weight


if __name__ == "__main__":
    # Test
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=0)
    args = parser.parse_args()

    model = get_DNN(args.step)
    file_list, wavs_path = get_file_list()
    os.makedirs("result", exist_ok=True)
    weights = list()

    for i, filename in enumerate(file_list):
        wav = audio.load_wav(os.path.join(wavs_path, filename))
        audio.save_wav(wav, os.path.join(
            "result", str(args.step) + "_" + str(i) + "_original.wav"))
        wav = torch.Tensor(wav)
        noi = audio.add_noise(
            wav, quantization_channel=hp.quantization_channel)
        audio.save_wav(noi.numpy(), os.path.join(
            "result", str(args.step) + "_" + str(i) + "_noi.wav"))
        mix = wav.float() + noi
        audio.save_wav(mix.numpy(), os.path.join(
            "result", str(args.step) + "_" + str(i) + "_mix.wav"))

        est_noi, est_wav, _ = test(model, mix)
        audio.save_wav(est_noi, os.path.join(
            "result", str(args.step) + "_" + str(i) + "_est_noi.wav"))
        audio.save_wav(est_wav, os.path.join(
            "result", str(args.step) + "_" + str(i) + "_est_wav.wav"))

        mix_ = wav.float()
        audio.save_wav(mix_.numpy(), os.path.join(
            "result", str(args.step) + "_" + str(i) + "_mix_non_noi.wav"))
        est_noi_, est_wav_, weight = test(model, mix_)
        weights.append(weight)
        audio.save_wav(est_noi_, os.path.join(
            "result", str(args.step) + "_" + str(i) + "_est_noi_non_noi.wav"))
        audio.save_wav(est_wav_, os.path.join(
            "result", str(args.step) + "_" + str(i) + "_est_wav_non_noi.wav"))

        # wav_autoencoder, weight_autoencoder = autoencoder(model, wav.float())
        # weights.append(weight_autoencoder)
        # audio.save_wav(wav_autoencoder, os.path.join(
        #     "result", str(args.step) + "_" + str(i) + "_autoencoder.wav"))

        print("Done", i)
    plot_data(weights, str(args.step) + ".jpg", figsize=(30, 4))
