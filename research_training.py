import os.path
import time

import numpy as np

from igng import IncrementalGrowingNeuralGas
from pprint import pprint
from prettytable import PrettyTable
from sklearn.preprocessing import Normalizer
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class TrainingResearch:
    def __init__(
            self,
            sigma: float = 0.05,
            epsilon_b: float = 0.1,
            epsilon_n: float = 0.05,
            a_max: int = 10,
            a_mature: int = 4,
    ):
        self.sigma = sigma
        self.epsilon_b = epsilon_b
        self.epsilon_n = epsilon_n
        self.a_max = a_max
        self.a_mature = a_mature

        self.vectorCount_trainingTime_table: PrettyTable = PrettyTable()
        self.vectorCount_trainingTime_table.field_names = [
            "Количество параметров",
            "Количество векторов",
            "Время обучения нейронной сети",
        ]
        self.vectorCounts = [
            100,
            200,
            400,
            600,
            1000,
            2000,
            3000,
            4000,
            5000,
            6000,
            7000,
            8000,
            9000,
            10000,
        ]
        self.default_n_features = 2
        self.vectorCount_trainingTime_results = []
        self.vectorCount_trainingTime_results_dst = "vectorCount_trainingTime_results.txt"

        self.vectorSize_trainingTime_table: PrettyTable = PrettyTable()
        self.vectorCount_trainingTime_table.field_names = [
            "Количество параметров",
            "Количество векторов",
            "Время обучения нейронной сети",
        ]
        self.default_n_samples = 3000
        self.vectorSize_trainingTime_results = []
        self.vectorSize_trainingTime_results_dst = "vectorSize_trainingTime_results.txt"
        self.vectorSizes = [
            1,
            2,
            4,
            8,
            12,
            20,
            28,
            40,
            60,
            80,
            100,
            150,
            200,
            300,
        ]

    def generate_data(self, n_samples: int, n_features: int) -> np.ndarray:
        data = datasets.make_multilabel_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=5,
        )
        data = StandardScaler().fit_transform(data[0])
        return data

    def build_igng(self):
        return IncrementalGrowingNeuralGas(
            sigma=self.sigma,
            epsilon_b=self.epsilon_b,
            epsilon_n=self.epsilon_n,
            a_max=self.a_max,
            a_mature=self.a_mature,
        )

    def perform_single_time_research(self, n_samples: int, n_features: int) -> float:
        igng = self.build_igng()
        data = self.generate_data(n_samples, n_features)

        t0 = time.time()
        igng.fit(data)
        t1 = time.time()

        dt = round(t1 - t0, 2)

        return dt

    def perform_vectorCount_trainingTime_research(self):
        for vector_count in self.vectorCounts:
            dt = self.perform_single_time_research(vector_count, self.default_n_features)
            row = [self.default_n_features, vector_count, dt]

            self.vectorCount_trainingTime_table.add_row(row)
            self.vectorCount_trainingTime_results.append(row)
            pprint(row)

    def perform_vectorSize_trainingTime_research(self):
        for vector_size in self.vectorSizes:
            dt = self.perform_single_time_research(self.default_n_samples, vector_size)
            row = [vector_size, self.default_n_samples, dt]

            self.vectorSize_trainingTime_table.add_row(row)
            self.vectorSize_trainingTime_results.append(row)
            pprint(row)

    def print_vectorCount_trainingTime_research_results(self):
        print(self.vectorCount_trainingTime_table)

    def print_vectorSize_trainingTime_research_results(self):
        print(self.vectorSize_trainingTime_table)

    def show_vectorCount_trainingTime_research_results(self):
        self.prepare_plt_to_show_vectorCount_trainingTime_research_results(self.vectorCount_trainingTime_results)
        plt.show()

    def prepare_plt_to_show_vectorCount_trainingTime_research_results(self, data):
        times = [r[2] for r in data]
        vector_counts = [r[1] for r in data]
        vector_size = data[0][0]

        plt.plot(
            vector_counts,
            times,
            color="black",
            marker=".",
            label=f"Количество параметров {vector_size}",
        )
        plt.legend()

        plt.title("Зависимость времени обучения от количества векторов")
        plt.xlabel("Количество векторов")
        plt.ylabel("Время обучения нейронной сети (в секундах)")

    def show_vectorSize_trainingTime_research_results(self):
        self.prepare_plt_to_show_vectorSize_trainingTime_research_results(self.vectorSize_trainingTime_results)
        plt.show()

    def prepare_plt_to_show_vectorSize_trainingTime_research_results(self, data):
        times = [r[2] for r in data]
        vector_sizes = [r[0] for r in data]
        vector_count = data[0][1]

        plt.plot(
            vector_sizes,
            times,
            color="black",
            marker=".",
            label=f"Количество векторов {vector_count}",
        )
        plt.legend()

        plt.title("Зависимость времени обучения от количества параметров")
        plt.xlabel("Количество параметров")
        plt.ylabel("Время обучения нейронной сети (в секундах)")

    def save_vectorCount_trainingTime_research_results(self):
        self.save_data_to_file(self.vectorCount_trainingTime_results, self.vectorCount_trainingTime_results_dst)

    def save_vectorSize_trainingTime_research_results(self):
        self.save_data_to_file(self.vectorSize_trainingTime_results, self.vectorSize_trainingTime_results_dst)

    def save_data_to_file(self, data, path):
        with open(path, "wt") as f:
            for d in data:
                row = ",".join(list(map(str, d)))
                f.write(f"{row}\n")
    def show_vectorCount_trainingTime_research_results_from_backup(self):
        if not os.path.exists(self.vectorCount_trainingTime_results_dst):
            print("vectorCount_trainingTime_research results backup not found")
            return

        data = []
        with open(self.vectorCount_trainingTime_results_dst, "rt") as f:
            for row in f.readlines():
                row_split = row.split(",")
                data.append(
                    [
                        int(row_split[0]),
                        int(row_split[1]),
                        float(row_split[2]),
                    ]
                )

        self.prepare_plt_to_show_vectorCount_trainingTime_research_results(data)
        plt.show()

    def show_vectorSize_trainingTime_research_results_from_backup(self):
        if not os.path.exists(self.vectorSize_trainingTime_results_dst):
            print("vectorSize_trainingTime_research results backup not found")
            return

        data = []
        with open(self.vectorSize_trainingTime_results_dst, "rt") as f:
            for row in f.readlines():
                row_split = row.split(",")
                data.append(
                    [
                        int(row_split[0]),
                        int(row_split[1]),
                        float(row_split[2]),
                    ]
                )

        self.prepare_plt_to_show_vectorSize_trainingTime_research_results(data)
        plt.show()


if __name__ == "__main__":
    research = TrainingResearch()

    research.show_vectorCount_trainingTime_research_results_from_backup()
    research.show_vectorSize_trainingTime_research_results_from_backup()

    # research.perform_vectorCount_trainingTime_research()
    # research.print_vectorCount_trainingTime_research_results()
    # research.show_vectorCount_trainingTime_research_results()
    # research.save_vectorCount_trainingTime_research_results()
    #
    # research.perform_vectorSize_trainingTime_research()
    # research.print_vectorSize_trainingTime_research_results()
    # research.show_vectorSize_trainingTime_research_results()
    # research.save_vectorSize_trainingTime_research_results()
