import torch

from image import ImageData

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from os import listdir
from json import dump, load
from torch import Tensor, flatten, save as t_save, load as t_load

PROBABILITY = 0.98
LOST = 2

LR = 1 * 10 ** -5
STEP = 5


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.relu = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.dropout = nn.Dropout(0.2)

        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.layer3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.fc = nn.Linear(5 * 5 * 128, 54)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.relu(x)

        x = self.layer3(x)
        x = self.relu(x)

        x = self.dropout(x)

        x = flatten(x)

        x = self.fc(x)

        return x


class Net:
    def __init__(self, set_title: str = "original_set"):
        self.dataset = []
        self.set_title = set_title

        self.create_dataset()

        self.net = SimpleCNN()
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=LR
        )
        self.criterion = nn.CrossEntropyLoss()

        self.lost_data = []
        self.percent_data = []

        print("The network has been initialized")

    def load_settings(self):
        self.net.load_state_dict(
            t_load(f'checkpoints/{self.set_title}.pt', weights_only=True)
        )

    def save_settings(self):
        t_save(self.net.state_dict(), f'checkpoints/{self.set_title}.pt')

    def load_data(self):
        with open(
                f"output/{self.set_title}_lost_list.json", "r", encoding="utf8"
        ) as file:
            self.lost_data += load(file)

        with open(
                f"output/{self.set_title}_perc_list.json", "r", encoding="utf8"
        ) as file:
            self.percent_data += load(file)

    def save_data(self):
        with open(
                f"output/{self.set_title}_lost_list.json", "w", encoding="utf8"
        ) as file:
            dump(self.lost_data, file, indent=4)

        with open(
                f"output/{self.set_title}_perc_list.json", "w", encoding="utf8"
        ) as file:
            dump(self.percent_data, file, indent=4)

    def create_dataset(self) -> list[ImageData]:
        """
        Training image dataset creation function.

        :return: list[ImageData] - List of objects of the
        ImageData class for training the model
        """
        dataset = []

        for file in listdir(f"materials/{self.set_title}"):
            self.dataset.append(
                ImageData(f"materials/{self.set_title}", file)
            )

        return dataset

    def training_network(self, generations=100):
        percent = self.testing_network(testing_data=False)
        print(f"Training: {percent * 100:.2f}%")

        percent = self.testing_network()
        self.percent_data.append(percent)
        print(f"Test:     {percent * 100:.2f}%\n")

        index = 0

        if generations == -1:
            while (len(self.percent_data) == 0 or
                   self.percent_data[-1] < PROBABILITY):
                index += 1
                self.__one_generation__(index)
        elif generations == -2:
            while (len(self.lost_data) == 0 or
                   self.lost_data[-1] > LOST):
                index += 1
                self.__one_generation__(index)
        else:
            while index != generations:
                index += 1
                self.__one_generation__(index)

    def __one_generation__(self, index: int):
        error_list = []

        for data in self.dataset:
            for rotate in range(0, 8):
                loss = self.train_once(
                    data.training_data[rotate],
                    data.result
                )
                if loss is not None:
                    error_list.append(loss)

        print(str(index), sum(error_list) / len(error_list))

        self.lost_data.append(sum(error_list) / len(error_list))

        if index % STEP == 0:
            percent = self.testing_network(testing_data=False)
            print(f"Training: {percent * 100:.2f}%")

            percent = self.testing_network()
            self.percent_data.append(percent)
            print(f"Test:     {percent * 100:.2f}%\n")

    def train_once(self, data: Tensor, result: Tensor) -> float:
        if data is None:
            return None

        self.optimizer.zero_grad()

        net_out = self.net(data)

        loss = self.criterion(net_out, result)
        loss.backward()
        self.optimizer.step()

        return float(loss)

    def testing_network(self, with_output=False, testing_data=True) -> float:
        self.net.eval()

        with torch.no_grad():
            pass_count = 0
            for data in self.dataset:
                for degrees in range(8):
                    if testing_data:
                        net_out: Tensor = self.net(data.testing_data[degrees])
                    else:
                        net_out: Tensor = self.net(data.training_data[degrees])

                    if with_output:
                        data.print_result(net_out)
                    pass_count += data.result_equals(net_out)

        self.net.train()

        return pass_count / (len(self.dataset) * 8)

    def show_graphic(self) -> None:
        plt.plot(
            self.lost_data,
            label="The amount of cross entropy"
        )
        plt.plot(
            range(0, len(self.percent_data) * STEP, STEP),
            self.percent_data,
            label="Percentage of match"
        )

        plt.grid(visible=True)

        plt.legend(loc="best")
        plt.title("Neural network training schedule")

        plt.show()


if __name__ == "__main__":
    net = Net()

    net.load_settings()
    net.load_data()

    net.training_network(-1)

    net.save_settings()
    net.save_data()

    print(net.testing_network(with_output=True))
    # print(net.testing_network(with_output=True, testing_data=False))

    net.show_graphic()
