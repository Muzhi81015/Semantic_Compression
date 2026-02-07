import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import SNR_to_noise, initNetParams, train_step, val_step, train_mi
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
# from models.LSTM import DeepSC
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser()
# parser.add_argument('--data-dir', default='data/data.pkl', type=str)
parser.add_argument('--vocab-file', default='/vocab.json', type=str)
# parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-AWGN', type=str)
# parser.add_argument('--channel', default='Rayleigh', type=str, help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--channel', default='AWGN', type=str, help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=256, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--epochs', default=20, type=int)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def validate(epoch, args, net):
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)

    net.eval()
    pbar = tqdm(test_iterator)
    total_loss = 0
    total_sentences = 0
    total_time = 0

    with torch.no_grad():
        for sents in pbar:
            start_time = time.time()
            sents = sents.to(device)
            loss = val_step(net, sents, sents, 0.1, pad_idx,
                            criterion, args.channel)
            total_time += (time.time() - start_time)

            total_loss += loss
            total_sentences += sents.size(0)

            pbar.set_description(
                'Epoch: {}; Type: VAL; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )

    avg_time_per_sentence = total_time / total_sentences
    print(f"平均验证时间每句子: {avg_time_per_sentence:.5f} 秒")

    return total_loss / len(test_iterator)


def train(epoch, args, net, mi_net=None):
    train_eur = EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)

    noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))

    total_time = 0
    total_sentences = 0

    for sents in pbar:
        start_time = time.time()
        sents = sents.to(device)

        if mi_net is not None:
            mi = train_mi(net, mi_net, sents, 0.1, pad_idx, mi_opt, args.channel)
            loss = train_step(net, sents, sents, 0.1, pad_idx,
                              optimizer, criterion, args.channel, mi_net)
        else:
            loss = train_step(net, sents, sents, noise_std[0], pad_idx,
                              optimizer, criterion, args.channel)

        total_time += (time.time() - start_time)
        total_sentences += sents.size(0)

        if mi_net is not None:
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}; MI {:.5f}'.format(
                    epoch + 1, loss, mi
                )
            )
        else:
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )

    avg_time_per_sentence = total_time / total_sentences
    print(f"平均训练时间每句子: {avg_time_per_sentence:.5f} 秒")


# import time  # 新增的导入
if __name__ == '__main__':
    args = parser.parse_args()
    args.vocab_file = './data/' + args.vocab_file

    """ preparing the dataset """
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    """ define optimizer and loss function """
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                    num_vocab, num_vocab, args.d_model, args.num_heads,
                    args.dff, 0.1).to(device)

    mi_net = Mine().to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(deepsc.parameters(),
                                 lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=1e-3)

    initNetParams(deepsc)

    # ✅ 正确的变量初始化位置（在循环外）
    total_train_time = 0
    total_val_time = 0
    record_acc = 10  # 越小越好，初始设高

    for epoch in range(args.epochs):
        print(f"\n======== Epoch {epoch + 1} ========")

        # --- Training ---
        train_start = time.time()
        train(epoch, args, deepsc)
        train_end = time.time()
        epoch_train_time = train_end - train_start
        total_train_time += epoch_train_time
        print(f"训练总耗时: {epoch_train_time:.2f} 秒")

        # --- Validation ---
        val_start = time.time()
        avg_acc = validate(epoch, args, deepsc)
        val_end = time.time()
        epoch_val_time = val_end - val_start
        total_val_time += epoch_val_time
        print(f"验证总耗时: {epoch_val_time:.2f} 秒")

        # --- Save model if improved ---
        if avg_acc < record_acc:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            ckpt_path = os.path.join(args.checkpoint_path, f'checkpoint_{str(epoch + 1).zfill(2)}.pth')
            torch.save(deepsc.state_dict(), ckpt_path)
            record_acc = avg_acc













