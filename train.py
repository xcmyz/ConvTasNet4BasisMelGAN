import torch
import utils
import argparse
import os
import time
import math
import audio
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hparams as hp

from loss import cal_loss
from tasnet import ConvTasNet
from dataset import BufferDataset, DataLoader
from dataset import get_data_to_buffer, collate_fn_tensor
from radam import RAdam
from scheduler import ScheduledOptim


def main(args):
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define model
    print("Use TasNet")
    model = nn.DataParallel(ConvTasNet(N=hp.N, L=hp.L,
                                       B=hp.B, H=hp.H,
                                       P=hp.P, X=hp.X,
                                       R=hp.R, C=hp.C,
                                       norm_type=hp.norm_type,
                                       causal=hp.causal,
                                       mask_nonlinear=hp.mask_nonlinear)).to(device)
    print("Model Has Been Defined")
    num_param = utils.get_param_num(model)
    print('Number of TTS Parameters:', num_param)

    # Get buffer
    print("Load data to buffer")
    buffer = get_data_to_buffer()

    # Optimizer and loss
    optimizer = RAdam(model.parameters())
    scheduled_optim = ScheduledOptim(optimizer, hp.N, hp.n_warm_up_step, args.restore_step)
    print("Defined Optimizer and Loss Function.")

    # Load checkpoint if exists
    try:
        checkpoint = torch.load(os.path.join(
            hp.checkpoint_path, 'checkpoint_%d.pth.tar' % args.restore_step))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n---Model Restored at Step %d---\n" % args.restore_step)
    except:
        os.makedirs(hp.checkpoint_path, exist_ok=True)
        print("\n---Start New Training---\n")

    # Init logger
    os.makedirs(hp.logger_path, exist_ok=True)

    # Get dataset
    dataset = BufferDataset(buffer)

    # Get Training Loader
    training_loader = DataLoader(dataset,
                                 batch_size=hp.batch_expand_size * hp.batch_size,
                                 shuffle=True,
                                 collate_fn=collate_fn_tensor,
                                 drop_last=True,
                                 num_workers=0)
    total_step = hp.epochs * len(training_loader) * hp.batch_expand_size

    # Define Some Information
    Time = np.array([])
    Start = time.perf_counter()

    # Training
    model = model.train()

    for epoch in range(hp.epochs):
        for i, batchs in enumerate(training_loader):
            # real batch start here
            for j, db in enumerate(batchs):
                start_time = time.perf_counter()

                current_step = i * hp.batch_expand_size + j + args.restore_step + epoch * len(training_loader) * hp.batch_expand_size + 1

                # Init
                scheduled_optim.zero_grad()

                # Get Data
                mix = db["mix"].float().to(device)
                target = db["target"].float().to(device)
                length = db["length"].int().to(device)

                # Forward
                est_source, mixture_w, source_w = model(mix)

                weight_average = mixture_w.sum() / (mixture_w.size(0) * mixture_w.size(1) * mixture_w.size(2))
                weight_average_ = source_w.sum() / (source_w.size(0) * source_w.size(1) * source_w.size(2) * source_w.size(3))
                str0 = "weight average value: {:.6f}, weight_ average value: {:.6f}.".format(weight_average, weight_average_)
                print(str0)

                # Cal Loss
                # Only calculate human voice snr
                # !!! different from paper
                loss, max_snr, \
                    estimate_source, reorder_estimate_source \
                    = cal_loss(target[:, :1, :], est_source[:, :1, :], length)
                if False:
                    audio.save_wav(target[0][0].cpu().numpy(), "test.wav", hp.sample_rate)

                # Logger
                l = loss.item()
                m_s = torch.mean(max_snr).item()
                with open(os.path.join("logger", "loss.txt"), "a") as f_loss:
                    f_loss.write(str(l)+"\n")

                # Backward
                loss.backward()

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_thresh)

                # Update weights
                if args.frozen_learning_rate:
                    scheduled_optim.step_and_update_lr_frozen(args.learning_rate_frozen)
                else:
                    scheduled_optim.step_and_update_lr()

                # Print
                if current_step % hp.log_step == 0:
                    Now = time.perf_counter()

                    str1 = "Epoch [{}/{}], Step [{}/{}]:".format(epoch + 1, hp.epochs, current_step, total_step)
                    str2 = "Loss: {:.6f}, SNR: {:.6f}".format(l, m_s)
                    str3 = "Current Learning Rate is {:.6f}.".format(scheduled_optim.get_learning_rate())
                    str4 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format((Now-Start), (total_step-current_step)*np.mean(Time))

                    print("\n" + str1)
                    print(str2)
                    print(str3)
                    print(str4)

                    with open(os.path.join("logger", "logger.txt"), "a") as f_logger:
                        f_logger.write(str0 + "\n")
                        f_logger.write(str1 + "\n")
                        f_logger.write(str2 + "\n")
                        f_logger.write(str3 + "\n")
                        f_logger.write(str4 + "\n")
                        f_logger.write("\n")

                if current_step % hp.save_step == 0:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                               os.path.join(hp.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                    print("save model at step %d ..." % current_step)

                end_time = time.perf_counter()
                Time = np.append(Time, end_time - start_time)
                if len(Time) == hp.clear_Time:
                    temp_value = np.mean(Time)
                    Time = np.delete(Time, [i for i in range(len(Time))], axis=None)
                    Time = np.append(Time, temp_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    parser.add_argument('--frozen_learning_rate', type=bool, default=False)
    parser.add_argument("--learning_rate_frozen", type=float, default=2e-4)
    args = parser.parse_args()
    main(args)
