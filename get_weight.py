import torch

import parser
import argparse
import numpy as np

from analysis import get_DNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=0)
    args = parser.parse_args()

    model = get_DNN(args.step)
    weight = model.module.decoder.basis_signals.weight
    weight = weight.detach().numpy().astype(np.float32)
    np.save("basis_signal_weight.npy", weight, allow_pickle=False)
