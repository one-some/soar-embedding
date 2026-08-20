import os
import gc
import re
import csv
import glob
import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import tqdm

from torch.utils.data import DataLoader
from bert_pytorch.model import BERT
from bert_pytorch.trainer import BERTTrainer
from bert_pytorch.dataset import LogDataset, WordVocab
from bert_pytorch.dataset.sample import generate_train_valid
from bert_pytorch.dataset.utils import save_parameters


class Trainer():
    def __init__(self, options):
        self.device = options["device"]
        self.model_dir = options["model_dir"]
        self.model_path = options["model_path"]
        self.vocab_path = options["vocab_path"]
        self.output_path = options["output_dir"]
        self.window_size = options["window_size"]
        self.adaptive_window = options["adaptive_window"]
        self.sample_ratio = options["train_ratio"]
        self.valid_ratio = options["valid_ratio"]
        self.seq_len = options["seq_len"]
        self.max_len = options["max_len"]
        self.corpus_lines = options["corpus_lines"]
        self.on_memory = options["on_memory"]
        self.batch_size = options["batch_size"]
        self.num_workers = options["num_workers"]
        self.lr = options["lr"]
        self.adam_beta1 = options["adam_beta1"]
        self.adam_beta2 = options["adam_beta2"]
        self.adam_weight_decay = options["adam_weight_decay"]
        self.with_cuda = options["with_cuda"]
        self.cuda_devices = options["cuda_devices"]
        self.log_freq = options["log_freq"]
        self.epochs = options["epochs"]
        self.hidden = options["hidden"]
        self.layers = options["layers"]
        self.attn_heads = options["attn_heads"]
        self.is_logkey = options["is_logkey"]
        self.is_time = options["is_time"]
        self.scale = options["scale"]
        self.scale_path = options["scale_path"]
        self.n_epochs_stop = options["n_epochs_stop"]
        self.hypersphere_loss = options["hypersphere_loss"]
        self.mask_ratio = options["mask_ratio"]
        self.min_len = options['min_len']
        self.resume = options.get("resume", False)

        print("Save options parameters")
        save_parameters(options, self.model_dir + "parameters.txt")

    def train(self):

        print("Loading vocab", self.vocab_path)
        vocab = WordVocab.load_vocab(self.vocab_path)
        print("vocab Size: ", len(vocab))

        print("\nLoading Train Dataset")
        logkey_train, logkey_valid, time_train, time_valid = generate_train_valid(self.output_path + "train", window_size=self.window_size,
                                     adaptive_window=self.adaptive_window,
                                     valid_size=self.valid_ratio,
                                     sample_ratio=self.sample_ratio,
                                     scale=self.scale,
                                     scale_path=self.scale_path,
                                     seq_len=self.seq_len,
                                     min_len=self.min_len
                                    )

        train_dataset = LogDataset(logkey_train, time_train, vocab, seq_len=self.seq_len,
                                   corpus_lines=self.corpus_lines, on_memory=self.on_memory, mask_ratio=self.mask_ratio)

        print("\nLoading valid Dataset")
        valid_dataset = LogDataset(logkey_valid, time_valid, vocab, seq_len=self.seq_len,
                                   on_memory=self.on_memory, mask_ratio=self.mask_ratio)

        print("Creating Dataloader")
        self.train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                            collate_fn=train_dataset.collate_fn, drop_last=True)
        self.valid_data_loader = DataLoader(valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                            collate_fn=train_dataset.collate_fn, drop_last=True)
        del train_dataset
        del valid_dataset
        del logkey_train
        del logkey_valid
        del time_train
        del time_valid
        gc.collect()

        resume_epoch, latest = 0, None
        if self.resume:
            snapshots = glob.glob(os.path.join(self.model_dir, "bert_epoch_*.pth"))
            if snapshots:
                def _epoch_num(p):
                    m = re.search(r"bert_epoch_(\d+)\.pth", p)
                    return int(m.group(1)) if m else 0
                latest = max(snapshots, key=_epoch_num)
                resume_epoch = _epoch_num(latest)
                print(f"Resuming from {latest} (epoch {resume_epoch})")

        if resume_epoch > 0:
            # Snapshot is the full BERTLog wrapper (BERT + LM head); BERTTrainer
            # expects raw BERT and re-wraps. Unwrap before passing.
            checkpoint = torch.load(latest, weights_only=False, map_location=self.device)
            bert_model = checkpoint.bert if hasattr(checkpoint, 'bert') else checkpoint
        else:
            print("Building BERT model")
            bert_model = BERT(len(vocab), max_len=self.max_len, hidden=self.hidden,
                              n_layers=self.layers, attn_heads=self.attn_heads,
                              is_logkey=self.is_logkey, is_time=self.is_time)

        print("Creating BERT Trainer")
        self.trainer = BERTTrainer(bert_model, len(vocab), train_dataloader=self.train_data_loader,
                                   valid_dataloader=self.valid_data_loader,
                                   lr=self.lr, betas=(self.adam_beta1, self.adam_beta2),
                                   weight_decay=self.adam_weight_decay,
                                   with_cuda=self.with_cuda, cuda_devices=self.cuda_devices,
                                   log_freq=self.log_freq,
                                   is_logkey=self.is_logkey, is_time=self.is_time,
                                   hypersphere_loss=self.hypersphere_loss)

        self.start_iteration(surfix_log="log2", start_epoch=resume_epoch)
        self.plot_train_valid_loss("_log2")

    def start_iteration(self, surfix_log, start_epoch=0):
        print("Training Start")
        csv_file = os.path.join(self.model_dir, "training_metrics.csv")

        best_loss = float('inf')
        epochs_no_improve = 0

        # Recover best_loss and the no-improvement counter from the CSV log.
        if start_epoch > 0 and os.path.exists(csv_file):
            try:
                with open(csv_file) as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    best_loss = min(float(r['Valid_Loss']) for r in rows)
                    for r in reversed(rows):
                        if float(r['Valid_Loss']) <= float(r['Best_Loss']) + 1e-9:
                            break
                        epochs_no_improve += 1
            except Exception as e:
                print(f"could not recover training state: {e}")

        csv_mode = 'a' if start_epoch > 0 else 'w'
        with open(csv_file, csv_mode, newline='') as f:
            writer = csv.writer(f)
            if csv_mode == 'w':
                writer.writerow(['Epoch', 'Time', 'Train_Loss', 'Valid_Loss', 'Best_Loss'])

        for epoch in range(start_epoch, self.epochs):
            print("\n")
            if self.hypersphere_loss:
                center = self.calculate_center([self.train_data_loader, self.valid_data_loader])
                self.trainer.hyper_center = center

            train_loss, train_dist = self.trainer.train(epoch)
            avg_loss, valid_dist = self.trainer.valid(epoch)
            self.trainer.save_log(self.model_dir, surfix_log)

            self.trainer.save(self.model_dir + f"bert_epoch_{epoch+1}.pth")

            if self.hypersphere_loss:
                self.trainer.radius = self.trainer.get_radius(train_dist + valid_dist, self.trainer.nu)

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.trainer.save(self.model_path)
                epochs_no_improve = 0

                if epoch > 10 and self.hypersphere_loss:
                    best_center = self.trainer.hyper_center
                    best_radius = self.trainer.radius
                    total_dist = train_dist + valid_dist

                    if best_center is None:
                        raise TypeError("center is None")

                    torch.save({"center": best_center, "radius": best_radius},
                               self.model_dir + "best_center.pt")
                    torch.save(total_dist, self.model_dir + "best_total_dist.pt")
            else:
                epochs_no_improve += 1

            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    train_loss, avg_loss, best_loss,
                ])

            if epochs_no_improve == self.n_epochs_stop:
                print("Early stopping")
                break

    def calculate_center(self, data_loader_list):
        print("start calculate center")
        with torch.no_grad():
            outputs = 0
            total_samples = 0
            for data_loader in data_loader_list:
                data_iter = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))
                for i, data in data_iter:
                    data = {key: value.to(self.device) for key, value in data.items()}

                    result = self.trainer.model.forward(data["bert_input"], data["time_input"])
                    cls_output = result["cls_output"]

                    outputs += torch.sum(cls_output.detach().clone(), dim=0)
                    total_samples += cls_output.size(0)

        return outputs / total_samples

    def plot_train_valid_loss(self, surfix_log):
        try:
            train_loss = pd.read_csv(self.model_dir + f"train{surfix_log}.csv")
            valid_loss = pd.read_csv(self.model_dir + f"valid{surfix_log}.csv")
            sns.lineplot(x="epoch", y="loss", data=train_loss, label="train loss")
            sns.lineplot(x="epoch", y="loss", data=valid_loss, label="valid loss")
            plt.title("epoch vs train loss vs valid loss")
            plt.legend()
            plt.savefig(self.model_dir + "train_valid_loss.png")
        except Exception as e:
            print(f"could not generate plot: {e}")
