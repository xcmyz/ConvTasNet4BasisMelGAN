import os
import numpy as np


# Wav
num_mels = 80
sample_rate = 24000
# sample_rate = 8000
num_freq = 1025
frame_length_ms = 13.5
frame_shift_ms = 2.50
preemphasis_enable = False
preemphasis = 0.97
fmin = 40
min_level_db = -100
ref_level_db = 20
signal_normalization = True
griffin_lim_iters = 60
power = 1.5


# Model
# quantization_channel = int(np.sqrt(2 ** 16))
# quantization_channel = int(np.sqrt(2 ** 12))
quantization_channel = int(np.sqrt(2 ** 8))


N = 256
# N = 192
L = 40
# L = 16
# L = 20
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
clean_p = 0.0

gen_size = 9800
test_size = 0

logger_path = os.path.join("logger")
dataset_path = os.path.join("dataset")
checkpoint_path = os.path.join("model_new")
vocoder_test_path = os.path.join("vocoder_test")

# batch_size = 3
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
