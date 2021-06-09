import os
import numpy as np


# Wav
num_mels = 80
num_freq = 1025
frame_length_ms = 50
frame_shift_ms = 10
fmin = 40
hop_size = 240
sample_rate = 24000
min_level_db = -100
ref_level_db = 20
preemphasize = True
preemphasis = 0.97
rescale_out = 0.4
signal_normalization = True
griffin_lim_iters = 60
power = 1.5


# Model
quantization_channel = int(np.sqrt(2 ** 8))


N = 256
L = 30
B = 256
H = 512
P = 3
X = 8
R = 4
C = 2
norm_type = "gLN"
causal = False
mask_nonlinear = "relu"


# Train
fixed_length = 20000
gen_size = 10000

logger_path = os.path.join("logger")
dataset_path = os.path.join("dataset")
checkpoint_path = os.path.join("model_new")
vocoder_test_path = os.path.join("vocoder_test")

batch_size = 8
epochs = 2000
n_warm_up_step = 4000

learning_rate = 1e-3
weight_decay = 1e-6
grad_clip_thresh = 1.0

save_step = 3000
log_step = 5
clear_Time = 20

batch_expand_size = 8
