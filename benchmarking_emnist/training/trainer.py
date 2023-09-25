import sys
import torch
from tqdm import tqdm
from colorama import Fore
from itertools import groupby
from torchmetrics.text import CharErrorRate
import numpy as np


class Trainer:
    def __init__(
        self, model, criterion, optimizer, train_loader, val_loader, epochs, args
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.args = args
        self.metrics = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        self.cer = CharErrorRate()
        self.all_train_cers = []
        self.all_val_cers = []

    def train(self, epoch):
        # Training
        train_correct = 0
        train_total = 0
        self.model.train()
        for x_train, y_train in tqdm(
            self.train_loader,
            position=0,
            leave=True,
            file=sys.stdout,
            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET),
        ):
            batch_size = x_train.shape[0]
            x_train = x_train.view(batch_size, 1, x_train.shape[1], x_train.shape[2])
            self.optimizer.zero_grad()
            y_pred = self.model(x_train)
            y_pred = y_pred.permute(1, 0, 2)
            input_lengths = torch.IntTensor(batch_size).fill_(
                self.args["cnn_output_width"]
            )
            target_lengths = torch.IntTensor([len(t) for t in y_train])
            loss = self.criterion(y_pred, y_train, input_lengths, target_lengths)
            loss.backward()
            self.optimizer.step()
            _, max_index = torch.max(y_pred, dim=2)
            for i in range(batch_size):
                raw_prediction = list(max_index[:, i].detach().cpu().numpy())
                prediction = torch.IntTensor(
                    [
                        c
                        for c, _ in groupby(raw_prediction)
                        if c != self.args["blank_label"]
                    ]
                )

                # check if predictions are empty and generate str list
                if prediction.numel() == 0:
                    prediction_for_cer = [""]
                else:
                    prediction_for_cer = [str(pred.item()) for pred in prediction]
                y_train_for_cer = [str(y.item()) for y in y_train[i]]

                self.all_train_cers.append(
                    self.cer(
                        prediction_for_cer,
                        y_train_for_cer,
                    ).item()
                )

                if len(prediction) == len(y_train[i]) and torch.all(
                    prediction.eq(y_train[i])
                ):
                    train_correct += 1
                train_total += 1

        train_accuracy = train_correct / train_total
        self.metrics["train_loss"].append(loss.item())
        self.metrics["train_accuracy"].append(train_accuracy)
        print(
            f"EPOCH {epoch + 1}/{self.epochs} - TRAINING. Correct: {train_correct}/{train_total} = {train_accuracy:.4f} - Average CER Score: {round(np.mean(np.array(self.all_train_cers)), 3)}"
        )

    def validate(self, epoch):
        # Validation
        val_correct = 0
        val_total = 0
        self.model.eval()
        with torch.no_grad():
            for x_val, y_val in tqdm(
                self.val_loader,
                position=0,
                leave=True,
                file=sys.stdout,
                bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET),
            ):
                batch_size = x_val.shape[0]
                x_val = x_val.view(batch_size, 1, x_val.shape[1], x_val.shape[2])
                y_pred = self.model(x_val)
                y_pred = y_pred.permute(1, 0, 2)
                input_lengths = torch.IntTensor(batch_size).fill_(
                    self.args["cnn_output_width"]
                )
                target_lengths = torch.IntTensor([len(t) for t in y_val])
                self.criterion(y_pred, y_val, input_lengths, target_lengths)
                _, max_index = torch.max(y_pred, dim=2)
                for i in range(batch_size):
                    raw_prediction = list(max_index[:, i].detach().cpu().numpy())
                    prediction = torch.IntTensor(
                        [
                            c
                            for c, _ in groupby(raw_prediction)
                            if c != self.args["blank_label"]
                        ]
                    )

                    # check if predictions are empty and generate str list
                    if prediction.numel() == 0:
                        prediction_for_cer = [""]
                    else:
                        prediction_for_cer = [str(pred.item()) for pred in prediction]
                    y_train_for_cer = [str(y.item()) for y in y_val[i]]

                    self.all_val_cers.append(
                        self.cer(
                            prediction_for_cer,
                            y_train_for_cer,
                        ).item()
                    )

                    if len(prediction) == len(y_val[i]) and torch.all(
                        prediction.eq(y_val[i])
                    ):
                        val_correct += 1
                    val_total += 1

        val_accuracy = val_correct / val_total
        self.metrics["val_accuracy"].append(val_accuracy)
        print(
            f"EPOCH {epoch + 1}/{self.epochs} - TESTING. Correct: {val_correct}/{val_total} = {val_accuracy:.4f} - Average CER Score: {round(np.mean(np.array(self.all_val_cers)), 3)}"
        )

    def train_validate_test(self):
        for epoch in range(self.epochs):
            self.train(epoch)
            self.validate(epoch)
