import sys
import torch
from tqdm import tqdm
from colorama import Fore
from itertools import groupby
from torchmetrics.text import CharErrorRate
import numpy as np
import hyperparameter as hp
import matplotlib as mpl
import matplotlib.pyplot as plt

class Trainer:
    def __init__(
        self, model, criterion, optimizer, train_loader, val_loader, test_loader, epochs, args, model_path=None
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.args = args
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))

        self.metrics = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        self.cer = CharErrorRate()
        self.all_train_cers = []
        self.all_val_cers = []
        self.epoch_train_cers = []
        self.epoch_val_cers = []

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

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
            train_correct_incr, train_total_incr = self.get_batch_cers(y_pred, y_train, batch_size)
            train_correct += train_correct_incr
            train_total += train_total_incr
        train_accuracy = train_correct / train_total
        # Add CER for epoch
        self.epoch_train_cers.append(round(np.mean(np.array(self.all_train_cers)), 3))
        self.metrics["train_loss"].append(loss.item())
        self.metrics["train_accuracy"].append(train_accuracy)
        print(
            f"EPOCH {epoch + 1}/{self.epochs} - TRAINING. Correct: {train_correct}/{train_total} = {train_accuracy:.4f} - Average CER Score: {round(np.mean(np.array(self.all_train_cers)), 3)}"
        )

    def get_batch_cers(self, y_pred, y_train, batch_size):
        train_correct = 0
        train_total = 0
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
        return train_correct, train_total

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
                val_correct_incr, val_total_incr = self.get_batch_cers(y_pred, y_val, batch_size)
                val_correct += val_correct_incr
                val_total += val_total_incr

        val_accuracy = val_correct / val_total
        # Add CER for epoch
        self.epoch_val_cers.append(round(np.mean(np.array(self.all_val_cers)), 3))
        self.metrics["val_accuracy"].append(val_accuracy)
        print(
            f"EPOCH {epoch + 1}/{self.epochs} - TESTING. Correct: {val_correct}/{val_total} = {val_accuracy:.4f} - Average CER Score: {round(np.mean(np.array(self.all_val_cers)), 3)}"
        )
        
    def test(self, plot_n = 1):
        # Validation
        test_correct = 0
        test_total = 0
        self.model.eval()
        with torch.no_grad():
            for x_test, y_test in tqdm(
                self.test_loader,
                position=0,
                leave=True,
                file=sys.stdout,
                bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET),
            ):
                batch_size = x_test.shape[0]
                x_test = x_test.view(batch_size, 1, x_test.shape[1], x_test.shape[2])
                y_pred = self.model(x_test)
                y_pred = y_pred.permute(1, 0, 2)
                input_lengths = torch.IntTensor(batch_size).fill_(
                    self.args["cnn_output_width"]
                )
                target_lengths = torch.IntTensor([len(t) for t in y_test])
                self.criterion(y_pred, y_test, input_lengths, target_lengths)
                test_correct_incr, test_total_incr = self.get_batch_cers(y_pred, y_test, batch_size)
                test_correct += test_correct_incr
                test_total += test_total_incr

        test_accuracy = test_correct / test_total
        # Add CER for epoch
        self.epoch_test_cers.append(round(np.mean(np.array(self.all_test_cers)), 3))
        self.metrics["test_accuracy"].append(test_accuracy)
        print(
            f"TESTING. Correct: {test_correct}/{test_total} = {test_accuracy:.4f} - Average CER Score: {round(np.mean(np.array(self.all_val_cers)), 3)}"
        )
        


        test_preds = []
        (x_test, y_test) = next(iter(self.train_loader))
        y_pred = self.model(x_test.view(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2]))

        # Prepare plotting results
        y_pred = y_pred.permute(1, 0, 2)
        _, max_index = torch.max(y_pred, dim=2)
        for i in range(x_test.shape[0]):
            raw_prediction = list(max_index[:, i].detach().cpu().numpy())
            prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != hp.BLANK_LABEL])
            test_preds.append(prediction)

        for j in range(len(x_test)):
            if j == plot_n:
                break

            mpl.rcParams["font.size"] = 8
            plt.imshow(x_test[j], cmap='gray')
            mpl.rcParams["font.size"] = 18
            plt.gcf().text(x=0.1, y=0.1, s="Actual: " + str(y_test[j].numpy()))
            plt.gcf().text(x=0.1, y=0.2, s="Predicted: " + str(test_preds[j].numpy()))
            plt.show()


    def train_validate_test(self):
        for epoch in range(self.epochs):
            self.train(epoch)
            self.validate(epoch)
