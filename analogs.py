import os
import json
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from pprint import pprint

import report_analyzer


class MfccDBSCAN:
    def __init__(self):
        self.features = []
        self.labels = []

        self.features_dir = "/Users/malyshevin/bmstu/DIPLOMA/program/app/app_data/features"

        self.files_count = 0
        self.load_features()

        self._eps_from = 0.020
        self._eps_to = 0.250
        self._eps_delta = 0.002
        self.round_digits = 3

        self.research_results = []
        self.research_results_dst = "MfccDBSCAN_research_results.json"

    def load_features(self):
        self.files_count = 0
        for file in os.listdir(self.features_dir):
            self.files_count += 1
            reader_id = int(file.split("-")[0])
            path = os.path.join(self.features_dir, file)
            features = []
            with open(path, "rt") as f:
                for line in f.readlines():
                    features.append(list(map(float, line.split(","))))
            labels = [reader_id for _ in features]
            self.features.extend(features)
            self.labels.extend(labels)

    def get_labels(self, eps: float, min_samples: int):
        return DBSCAN(eps=eps, min_samples=min_samples).fit(self.features).labels_

    def get_cluster_readers(self, eps: float, min_samples: int):
        fact_labels = self.get_labels(eps, min_samples)
        cluster_readers = {}
        for reader_id, cluster_id in zip(self.labels, fact_labels):
            if cluster_id not in cluster_readers:
                cluster_readers[cluster_id] = {}
            if reader_id not in cluster_readers[cluster_id]:
                cluster_readers[cluster_id][reader_id] = 0
            cluster_readers[cluster_id][reader_id] += 1
        return cluster_readers

    def make_analyze(self, eps: float, min_samples: int):
        cluster_readers = self.get_cluster_readers(eps, min_samples)
        properties = report_analyzer.calculate_properties(cluster_readers)
        metrics = report_analyzer.calculate_metrics(properties)

        return metrics

    def make_research(self):
        min_samples = 2
        eps = self._eps_from
        while eps - self._eps_to < 0:
            metrics = self.make_analyze(eps, min_samples)
            data = {
                "metrics": metrics,
                "min_samples": min_samples,
                "eps": round(eps, self.round_digits),
                "files_count": self.files_count,
            }
            self.research_results.append(data)
            pprint(data)
            eps += self._eps_delta

        self.save_research_results()

    def save_research_results(self):
        with open(self.research_results_dst, "wt") as f:
            for result in self.research_results:
                f.write(f"{json.dumps(result)}\n")

    def visualize_research_results(self):
        research_results = []
        with open(self.research_results_dst, "rt") as f:
            for line in f.readlines():
                research_result = json.loads(line)
                research_results.append(research_result)

        eps_list = [rr["eps"] for rr in research_results]

        positive_rate_list = [rr["metrics"]["positive_rate"] for rr in research_results]
        plt1 = plt.subplot(111)
        plt1.plot(
            eps_list,
            positive_rate_list,
            color="magenta",
            label="Доля положительных результатов",
        )
        plt1.legend()
        plt1.set_xlabel('Значение параметра eps')
        plt1.set_ylabel('Значение метрики')
        plt1.set_title("Зависимость доли положительных результатов от параметра eps")
        plt.show()

        negative_rate_list = [rr["metrics"]["negative_rate"] for rr in research_results]
        plt2 = plt.subplot(111)
        plt2.plot(
            eps_list,
            negative_rate_list,
            color="black",
            label="Доля негативных результатов",
        )
        plt2.legend()
        plt2.set_xlabel('Значение параметра eps')
        plt2.set_ylabel('Значение метрики')
        plt2.set_title("Зависимость доли негативных результатов от параметра eps")
        plt.show()

        unrecognized_rate_list = [rr["metrics"]["unrecognized_rate"] for rr in research_results]
        plt3 = plt.subplot(111)
        plt3.plot(
            eps_list,
            unrecognized_rate_list,
            color="blue",
            label="Доля неопределенных результатов",
        )
        plt3.legend()
        plt3.set_xlabel('Значение параметра eps')
        plt3.set_ylabel('Значение метрики')
        plt3.set_title("Зависимость доли неопределенных результатов от параметра eps")
        plt.show()


if __name__ == "__main__":
    research = MfccDBSCAN()
    # research.make_research()
    research.visualize_research_results()
