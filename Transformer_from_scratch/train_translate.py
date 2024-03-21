import torch
import torch.nn as nn
import os
import time
import config
from transformer import Transformer
from translation_data_loader import (
    get_batch_indices,
    load_cn_vocab,
    load_en_vocab,
    load_train_data
)
from utils.epoch_timer import epoch_time


def main():
    device = config.device
    cn2idx, idx2cn = load_cn_vocab()
    en2idx, idx2en = load_en_vocab()
    # X: cn Y: en
    X, Y = load_train_data()

    model = Transformer(
        src_pad_idx=config.PAD_ID,
        trg_pad_idx=config.PAD_ID,
        src_vocab_size=len(cn2idx),
        trg_vocab_size=len(en2idx),
        max_seq_len=config.max_seq_len,
        d_model=config.d_model,
        ffn_hidden=config.ffn_hidden,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
        device=device
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), config.init_lr)
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_ID)

    print_interval = 100
    counter = 0

    for epoch in range(config.epochs):
        start_time = time.time()

        for index, _ in get_batch_indices(len(X), config.batch_size):
            x_batch = torch.LongTensor(X[index]).to(device)
            y_batch = torch.LongTensor(Y[index]).to(device)
            y_input = y_batch[:, :-1]
            y_label = y_batch[:, 1:]
            y_hat = model(x_batch, y_input)

            y_label_mask = y_label != config.PAD_ID
            preds = torch.argmax(y_hat, -1)
            correct = preds == y_label
            acc = torch.sum(y_label_mask * correct) / torch.sum(y_label_mask)

            n, seq_len = y_label.shape
            y_hat = torch.reshape(y_hat, (n * seq_len, -1))
            y_label = torch.reshape(y_label, (n * seq_len, ))
            loss = criterion(y_hat, y_label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            if counter % print_interval == 0:
                end_time = time.time()
                elapsed_mins, elapsed_secs = epoch_time(start_time, end_time)
                print(f'{counter:08d} {elapsed_mins:02d}:{elapsed_secs:02d}'
                      f' loss: {loss.item()} acc: {acc.item()}')

                start_time = time.time()
            counter += 1

    model_path = './model/model.pth'
    if not os.path.exists('model'):
        os.makedirs('model')
    torch.save(model.state_dict(), model_path)

    print(f'Model saved to {model_path}')


if __name__ == '__main__':
    main()