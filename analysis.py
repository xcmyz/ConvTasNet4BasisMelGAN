import os
import torch
import torch.nn as nn
import argparse
import numpy as np
import random
import time
import tasnet
import audio
import scipy
import pandas as pd
import hparams as hp
import seaborn as sns
import hparams as hp

from scipy import signal
from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

random.seed(str(time.time()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET = "aishell3"  # aishell3 or biaobei


def plot_data(data, filename, figsize=(24, 4)):
    _, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto',
                       origin='upper', interpolation='none')

    os.makedirs("image", exist_ok=True)
    plt.savefig(os.path.join("image", filename))


def get_model(num):
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
    model.eval()
    return model


def get_file_list():
    if DATASET == "biaobei":
        ljspeech_path = os.path.join("BZNSYP")
        wavs_path = os.path.join(ljspeech_path, "Wave")
        file_list = os.listdir(wavs_path)
    elif DATASET == "aishell3":
        wavs_path = os.path.join("data_aishell3", "train", "wav")
        file_list = []
        for speaker in os.listdir(wavs_path):
            path = os.path.join(wavs_path, speaker)
            for filename in os.listdir(path):
                file_list.append(os.path.join(speaker, filename))
    file_list_ = []
    for filename in file_list:
        if filename[0] != ".":
            file_list_.append(filename)
    out_file_list = random.sample(file_list_, 3)
    return out_file_list, wavs_path


def test(model, mix):
    mix = torch.stack([mix]).to(device)
    with torch.no_grad():
        est_source, weight = model.module.test(mix)

    est_wav = est_source[0][0].cpu().numpy()
    est_noi = est_source[0][1].cpu().numpy()
    weight = weight[0].cpu().numpy()
    return est_noi, est_wav, weight


if __name__ == "__main__":
    # Test
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=0)
    args = parser.parse_args()

    model = get_model(args.step)
    file_list, wavs_path = get_file_list()
    os.makedirs("result", exist_ok=True)
    weights = list()

    for i, filename in enumerate(file_list):
        wav = audio.load_wav(os.path.join(wavs_path, filename), encode=False)
        audio.save_wav(wav, os.path.join("result", str(args.step) + "_" + str(i) + "_original.wav"), hp.sample_rate)
        wav = torch.Tensor(wav)
        noi = audio.add_noise(wav, quantization_channel=hp.quantization_channel)
        audio.save_wav(noi.numpy(), os.path.join("result", str(args.step) + "_" + str(i) + "_noi.wav"), hp.sample_rate)
        mix = wav.float() + noi
        audio.save_wav(mix.numpy(), os.path.join("result", str(args.step) + "_" + str(i) + "_mix.wav"), hp.sample_rate)

        est_noi, est_wav, _ = test(model, mix)
        audio.save_wav(est_noi, os.path.join("result", str(args.step) + "_" + str(i) + "_est_noi.wav"), hp.sample_rate)
        audio.save_wav(est_wav, os.path.join("result", str(args.step) + "_" + str(i) + "_est_wav.wav"), hp.sample_rate)

        mix_ = wav.float()
        audio.save_wav(mix_.numpy(), os.path.join("result", str(args.step) + "_" + str(i) + "_mix_non_noi.wav"), hp.sample_rate)
        est_noi_, est_wav_, weight = test(model, mix_)
        weights.append(weight)
        audio.save_wav(est_noi_, os.path.join("result", str(args.step) + "_" + str(i) + "_est_noi_non_noi.wav"), hp.sample_rate)
        audio.save_wav(est_wav_, os.path.join("result", str(args.step) + "_" + str(i) + "_est_wav_non_noi.wav"), hp.sample_rate)

        print("Done", i)
    plot_data(weights, str(args.step) + ".jpg", figsize=(30, 4))

    magnitude = []
    sorted_magnitude = []
    peak_magnitude = []
    w = 0
    basis_signal = model.module.decoder.basis_signals.weight.detach().numpy()
    for i in range(basis_signal.shape[1]):
        one = basis_signal[:, i]
        fft_result = np.fft.fft(one)
        fft_result = abs(fft_result)[len(abs(fft_result)) // 2:]
        magnitude.append(fft_result)
        peak_magnitude.append(np.argmax(fft_result))
    index = np.argsort(-np.array(peak_magnitude))
    for i in index:
        sorted_magnitude.append(magnitude[i])
    sorted_magnitude = torch.Tensor(sorted_magnitude).numpy().T
    hz = int(hp.sample_rate / (2 * sorted_magnitude.shape[0]))
    data = pd.DataFrame(sorted_magnitude,
                        index=list(reversed([(i * hz) for i in range(sorted_magnitude.shape[0])])),
                        columns=[i for i in range(sorted_magnitude.shape[1])])
    plt.figure(figsize=(16, 5))
    sns.heatmap(data=data)
    plt.title('FFT')
    plt.xlabel("Filter index")
    plt.ylabel("Frequency (Hz)")
    plt.show()
