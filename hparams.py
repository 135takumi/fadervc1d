from pathlib import Path

import torch


# data_root
#data_root = Path('/mnt/d/brood/M1/projects/fadervc1d/sessions')
data_root = Path('/home/isako/M1/projects/fadervc1d/sessions')
# data_root = Path('/Users/takumiisako/M1/projects/fadervc1d/sessions')

# exp_name
exp_name = 'default'

# feature config
sampling_rate = 24000
mcep_channels = 24
seq_len = 128
speaker_num = 4

# train config
batch_size = 128
lr = 1e-3
epochs = 6000
beta = 0.05
lat_dis_lambda = 0.05

# test config
valid_file_num = 32
test_file_num = 20

# save config
save_interval = 250

# debug mode
debug = False

# cuda device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# lambda_schedule
lambda_schedule = 4000

