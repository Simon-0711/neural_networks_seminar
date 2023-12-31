import sys
import torch
from tqdm import tqdm
from itertools import groupby
from torchmetrics.text import CharErrorRate
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class Trainer:
    def __init__(
        self,
        model,
        model_name,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        epochs,
        args,
        scheduler=None,
        model_path=None,
        device="cpu",
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.scheduler = scheduler
        self.device = device

        self.args = args
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))

        # After epoch
        self.metrics = {
            "train_loss": [],
            "train_accuracy": [],
            "train_cer": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_cer": [],
            "test_loss": [],
            "test_accuracy": [],
            "test_cer": [],
        }
        self.cer = CharErrorRate()
        self.all_train_cers = []
        self.all_val_cers = []
        self.all_test_cers = []
        # self.epoch_train_cers = []
        # self.epoch_val_cers = []
        # self.epoch_test_cers = []

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def train(self, epoch):
        # Training
        self.model.eval()
        train_correct = 0
        train_total = 0
        self.model.train()
        for x_train, y_train in tqdm(
            self.train_loader
        ):
            x_train.to(self.device)
            y_train.to(self.device)
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
            train_correct_incr, train_total_incr = self.get_batch_cers(
                y_pred, y_train, batch_size, self.all_train_cers
            )
            train_correct += train_correct_incr
            train_total += train_total_incr

        if self.scheduler != None:
            self.scheduler.step()
        train_accuracy = train_correct / train_total

        # Add CER for epoch
        self.metrics["train_cer"].append(
            round(np.mean(np.array(self.all_train_cers)), 3)
        )
        self.metrics["train_loss"].append(loss.item())
        self.metrics["train_accuracy"].append(train_accuracy)
        print(
            f"Training Epoch: {epoch + 1}/{self.epochs}; Loss: {loss.item()}; Correct: {train_correct}/{train_total} = {train_accuracy:.4f}; Average CER Score: {round(np.mean(np.array(self.all_train_cers)), 3)}"
        )

    def get_batch_cers(self, y_pred, y_gold, batch_size, all_cers):
        train_correct = 0
        train_total = 0
        _, max_index = torch.max(y_pred, dim=2)
        for i in range(batch_size):
            raw_prediction = list(max_index[:, i].detach().cpu().numpy())
            prediction = torch.IntTensor(
                [c for c, _ in groupby(raw_prediction) if c != self.args["blank_label"]]
            )

            # check if predictions are empty and generate str list
            if prediction.numel() == 0:
                prediction_for_cer = [""]
            else:
                prediction_for_cer = [str(pred.item()) for pred in prediction]
            y_gold_for_cer = [str(y.item()) for y in y_gold[i]]

            all_cers.append(
                self.cer(
                    prediction_for_cer,
                    y_gold_for_cer,
                ).item()
            )

            if len(prediction) == len(y_gold[i]) and torch.all(
                prediction.eq(y_gold[i])
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
                self.val_loader
            ):
                x_val.to(self.device)
                y_val.to(self.device)
                batch_size = x_val.shape[0]
                x_val = x_val.view(batch_size, 1, x_val.shape[1], x_val.shape[2])
                y_pred = self.model(x_val)
                y_pred = y_pred.permute(1, 0, 2)
                input_lengths = torch.IntTensor(batch_size).fill_(
                    self.args["cnn_output_width"]
                )
                target_lengths = torch.IntTensor([len(t) for t in y_val])
                loss = self.criterion(y_pred, y_val, input_lengths, target_lengths)
                val_correct_incr, val_total_incr = self.get_batch_cers(
                    y_pred, y_val, batch_size, self.all_val_cers
                )
                val_correct += val_correct_incr
                val_total += val_total_incr
        val_accuracy = val_correct / val_total
        # Add CER for epoch
        self.metrics["val_cer"].append(round(np.mean(np.array(self.all_val_cers)), 3))
        self.metrics["val_accuracy"].append(val_accuracy)
        self.metrics["val_loss"].append(loss.item())
        print(
            f"Validate Epoch: {epoch + 1}/{self.epochs}; Loss: {loss.item()}; Correct: {val_correct}/{val_total} = {val_accuracy:.4f}; Average CER Score: {round(np.mean(np.array(self.all_val_cers)), 3)}"
        )

    def test(self, plot_n=1, plot=False):
        # Testing
        test_correct = 0
        test_total = 0
        self.model.eval()
        with torch.no_grad():
            for x_test, y_test in tqdm(
                self.test_loader
            ):
                x_test.to(self.device)
                y_test.to(self.device)
                batch_size = x_test.shape[0]
                x_test = x_test.view(batch_size, 1, x_test.shape[1], x_test.shape[2])
                y_pred = self.model(x_test)
                y_pred = y_pred.permute(1, 0, 2)
                input_lengths = torch.IntTensor(batch_size).fill_(
                    self.args["cnn_output_width"]
                )
                target_lengths = torch.IntTensor([len(t) for t in y_test])
                loss = self.criterion(y_pred, y_test, input_lengths, target_lengths)
                test_correct_incr, test_total_incr = self.get_batch_cers(
                    y_pred, y_test, batch_size, self.all_test_cers
                )
                test_correct += test_correct_incr
                test_total += test_total_incr

        test_accuracy = test_correct / test_total
        # Add CER for epoch
        self.metrics["test_cer"].append(round(np.mean(np.array(self.all_test_cers)), 3))
        self.metrics["test_accuracy"].append(test_accuracy)
        self.metrics["test_loss"].append(loss.item())

        print(
            f"\033[91mTest Predictions --> Loss: {loss.item()}; Correct: {test_correct}/{test_total} = {test_accuracy:.4f}; Average CER Score: {round(np.mean(np.array(self.all_test_cers)), 3)}\033[0m"
        )

        if plot:
            test_preds = []
            (x_test, y_test) = next(iter(self.train_loader))
            y_pred = self.model(
                x_test.view(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2])
            )

            # Prepare plotting results
            y_pred = y_pred.permute(1, 0, 2)
            _, max_index = torch.max(y_pred, dim=2)
            for i in range(x_test.shape[0]):
                raw_prediction = list(max_index[:, i].detach().cpu().numpy())
                prediction = torch.IntTensor(
                    [c for c, _ in groupby(raw_prediction) if c != self.args["blank_label"]]
                )
                test_preds.append(prediction)

            for j in range(len(x_test)):
                if j == plot_n:
                    break
                print("Test Sample: " + str(j + 1))
                print("Gold Label: " + str(y_test[j].numpy()))
                print("Model Output: " + str(test_preds[j].numpy()))
                mpl.rcParams["font.size"] = 6
                mpl.rcParams["font.size"] = 10
                plt.imshow(x_test[j], cmap="gray")
                plt.show()

                # plt.gcf().text(x=0.1, y=0.1, s="Gold Label: " + str(y_test[j].numpy()) + ";  Model Output: " + str(test_preds[j].numpy()))

    def predictions_to_string(predictions, label_map):
        return ''.join([label_map[c] for c in predictions])

    def train_validate_test(self):
        for epoch in range(self.epochs):
            self.train(epoch)
            self.validate(epoch)
        self.test()
        # self.metrics["train_cer"] = self.epoch_train_cers
        # self.metrics["val_cer"] = self.epoch_val_cers
        # self.metrics["test_cer"] = self.epoch_test_cers
        return self.metrics

    def to_device(self, device):
        self.model.to(device)
