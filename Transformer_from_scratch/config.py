import torch


# GPU device setting
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# model parameter setting
batch_size = 128
max_seq_len = 256
d_model = 512
ffn_hidden = 1024
n_heads = 8
n_layers = 6
dropout = 0.1
epochs = 60
PAD_ID = 0

# optimizer parameter setting
init_lr = 1e-5
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100
epoch = 1000
clip = 1.0
weight_decay = 5e-4
inf = float('inf')
