import os
import audio
import torch
import librosa
import audio_tool

import numpy as np
import hparams as hp

import matplotlib
import matplotlib.pyplot as plt

from tasnet import Encoder
from tasnet import Decoder
from tasnet import ConvTasNet

from analysis import get_DNN
from generator import autoencoder
from utils import get_param_num, overlap_and_add, draw_picture


if __name__ == "__main__":
    print("TEST")

    # torch.manual_seed(123)
    # B, K, L, N, C = 2, 3, 4, 3, 2
    # hidden_size, num_layers = 4, 2
    # mixture = torch.randint(3, (B, K, L))
    # lengths = torch.LongTensor([K for i in range(B)])
    # # test Encoder
    # encoder = Encoder(L, N)
    # encoder.conv1d_U.weight.data = torch.randint(
    #     2, encoder.conv1d_U.weight.size())
    # encoder.conv1d_V.weight.data = torch.randint(
    #     2, encoder.conv1d_V.weight.size())
    # mixture_w, norm_coef = encoder(mixture)
    # print('mixture', mixture)
    # print('U', encoder.conv1d_U.weight)
    # print('V', encoder.conv1d_V.weight)
    # print('mixture_w', mixture_w)
    # print('norm_coef', norm_coef)

    # # test Separator
    # separator = Separator(N, hidden_size, num_layers)
    # est_mask = separator(mixture_w, lengths)
    # print('est_mask', est_mask)

    # # test Decoder
    # decoder = Decoder(N, L)
    # est_mask = torch.randint(2, (B, K, C, N))
    # est_source = decoder(mixture_w, est_mask, norm_coef)
    # print('est_source', est_source)

    # # test TasNet
    # tasnet = TasNet(L, N, hidden_size, num_layers)
    # est_source = tasnet(mixture, lengths)
    # print('est_source', est_source)

    # print("TEST")
    # test_encoder = Encoder(wav_length=256, encoder_dim=512)
    # test_decoder = Decoder(encoder_dim=512,
    #                        component_num=1024,
    #                        wav_length=256,
    #                        dim_attention=1024,
    #                        num_head=4)

    # mixture = torch.randn(2, 1234, 256)
    # mixture_w, norm_coef = test_encoder(mixture)
    # print(mixture_w.size())
    # print(norm_coef.size())

    # est_source = test_decoder(mixture_w, norm_coef)
    # print(est_source.size())

    # test_tasnet = TasNet(num_head=hp.tasnet_num_head,
    #                      encoder_dim=hp.tasnet_encoder_dim,
    #                      attention_dim=hp.tasnet_attention_dim,
    #                      wav_length=hp.tasnet_wav_length,
    #                      component_num=hp.tasnet_component_num)
    # mixture = torch.randn(2, 1234, 256)
    # print(test_tasnet(mixture).size())
    # print(get_param_num(test_tasnet))

    # print("TEST")
    # tsa = SelfAttention(dim_key=256, dim_query=512, dim_value=256, dim=1024)
    # key = torch.randn(1024, 256)
    # value = torch.randn(1024, 256)
    # query = torch.randn(2, 1234, 512)
    # result, score = tsa(key, value, query)
    # print(result.size())
    # print(score.size())

    # wav, _ = librosa.load("data/LJSpeech-1.1/wavs/LJ001-0001.wav", sr=22050)
    # peak = np.abs(wav).max()
    # wav /= peak
    # wav = audio.preemphasis(wav, 0.85)
    # wav = audio.inv_preemphasis(wav, 0.85)
    # audio.save_wav(wav, "test.wav", 22050)

    # source = torch.randn(2, 100, 256)
    # wav = overlap_and_add(source, 10)
    # print(wav.size())

    # wav = audio.load_wav(os.path.join(
    #     "data", "LJSpeech-1.1", "wavs", "LJ001-0001.wav"))
    # mel = audio.melspectrogram(wav)
    # wav_ = audio.inv_mel_spectrogram(mel)
    # audio.save_wav(wav_, "test.wav")

    wav = audio.load_wav(os.path.join(
        "LJSpeech-1.1", "wavs", "LJ001-0002.wav"))
    wav = torch.Tensor(wav)
    noi = audio.add_noise(wav, hp.quantization_channel)
    mix = (wav + noi).numpy()
    audio.save_wav(mix, os.path.join("test", "test.wav"))

    N = 256
    L = 20
    B = 256
    H = 512
    P = 3
    X = 8
    R = 4
    C = 2
    norm_type = "gLN"
    causal = False
    mask_nonlinear = "relu"

    test_model = ConvTasNet(N, L, B, H, P, X, R, C,
                            norm_type=norm_type,
                            causal=causal,
                            mask_nonlinear=mask_nonlinear)
    print(get_param_num(test_model))

    # mixture = torch.randn(3, 300)
    # est_source = test_model(mixture)
    # print(est_source.size())
    os.makedirs("test", exist_ok=True)

    mix = np.load(os.path.join("dataset", "mix-00300.npy"))
    wav = np.load(os.path.join("dataset", "wav-00300.npy"))
    noi = np.load(os.path.join("dataset", "noi-00300.npy"))

    x = np.array([i for i in range(mix.shape[0])])
    draw_picture(x, mix, wav, noi, ["mix", "wav", "noi"], "test")

    audio.save_wav(mix, os.path.join("test", "test_0.wav"))
    audio.save_wav(wav, os.path.join("test", "test_1.wav"))
    audio.save_wav(noi, os.path.join("test", "test_2.wav"))

    print(audio_tool.tools.get_mel("LJSpeech-1.1/wavs/LJ029-0012.wav").shape)
    print(audio_tool.tools.get_mel("LJSpeech-1.1/wavs/LJ023-0039.wav").shape)
    print(audio_tool.tools.get_mel("LJSpeech-1.1/wavs/LJ004-0136.wav").shape)

    model = get_DNN("270000")
    wav = audio.load_wav("LJSpeech-1.1/wavs/LJ029-0012.wav")
    print(wav)
    wav = torch.Tensor(wav)
    wav_, weight, weight_ = autoencoder(model, wav)
    print(wav_)

    print(wav.shape, wav_.shape)
    x = [i for i in range(wav.shape[0])]
    plt.plot(x, wav, 'r--', label='type1')
    plt.legend()
    plt.savefig(os.path.join("test", "wav_view_wav.jpg"))
    plt.clf()

    plt.plot(x, wav_, 'g--', label='type2')
    plt.legend()
    plt.savefig(os.path.join("test", "wav_view_wav_.jpg"))
    plt.clf()

    wav_direct_ = wav_
    wav_direct_ *= 1. / max(0.01, np.max(np.abs(wav_direct_)))
    plt.plot(x, wav_direct_, 'g--', label='type3')
    plt.legend()
    plt.savefig(os.path.join("test", "wav_view_wav_direct_.jpg"))
    plt.clf()

    audio.save_wav(wav_, "test_temp_.wav")
    wav_save_ = audio.load_wav("test_temp_.wav")
    plt.plot(x, wav_save_, 'g--', label='type3')
    plt.legend()
    plt.savefig(os.path.join("test", "wav_view_wav_save_.jpg"))
    plt.clf()

    # audio.save_wav(wav, "test_temp.wav")
    # wav_save = audio.load_wav("test_temp.wav")
    # plt.plot(x, wav_save, 'g--', label='type4')
    # plt.legend()
    # plt.savefig(os.path.join("test", "wav_view_wav_save.jpg"))
    # plt.clf()
