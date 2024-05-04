import logging
import math
import os.path

import networkx as nx
import numpy as np
from typing import Optional, List
from pathlib import Path

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler


class IncrementalGrowingNeuralGas:
    def __init__(
        self,
        # Максимальное расстояние между нейронами
        sigma: float = 0.05,
        # Коэффициент корректировки ближайшего нейрона
        epsilon_b: float = 0.1,
        # Коэффициент корректировки связанных с ближайшим нейронов
        epsilon_n: float = 0.05,
        # Максимальный возраст связи
        a_max: int = 10,
        # Возраст значащего нейрона
        a_mature: int = 4,
    ):
        self.distance_metric: str = "euclidean_distance"
        self.sigma: float = sigma
        self.epsilon_b: float = epsilon_b
        self.epsilon_n: float = epsilon_n
        self.a_max = a_max
        self.a_mature = a_mature

        self.network: Optional[nx.Graph] = nx.Graph()
        self.effective_network: Optional[nx.Graph] = None
        self._new_node_id: int = 0

    def build_effective_network(self):
        self.effective_network = nx.Graph()
        for node, attrs in self.network.nodes(data=True):
            if attrs["type"] == "mature":
                self.effective_network.add_node(node, **attrs)
        for a, b, attrs in self.network.edges(data=True):
            if self.network.nodes[a]["type"] == "mature" and self.network.nodes[b]["type"] == "mature":
                self.effective_network.add_edge(a, b, **attrs)

    def get_effective_network_status(self):
        if self.effective_network is None:
            return "Нейронная сеть не была обучена."

        node_count = len(self.effective_network.nodes)
        link_count = len(self.effective_network.edges)
        mature_node_count = 0
        for node, attr in self.effective_network.nodes(data=True):
            if attr["type"] == "mature":
                mature_node_count += 1
        mature_link_count = 0
        for link in self.effective_network.edges:
            if self.effective_network.nodes[link[0]]["type"] == "mature" and \
                    self.effective_network.nodes[link[1]]["type"] == "mature":
                mature_link_count += 1
        clusters_count = len(list(nx.connected_components(self.effective_network)))

        return f"Состояние нейронной сети: " \
               f"кластеров - {clusters_count}, " \
               f"узлов - {node_count}, " \
               f"связей - {link_count}, " \
               f"зрелых узлов - {mature_node_count}, " \
               f"связей между зрелыми узлами - {mature_node_count}."

    def dump_network_gml(self, path: Path = "network.gml"):
        nx.write_gml(self.effective_network, path)

    def load_network_gml(self, path: Path = "network.gml"):
        self.network = nx.read_gml(path)
        if len(self.network.nodes) > 0:
            self.effective_network = nx.read_gml(path)
        max_node = -1
        for node in self.network.nodes:
            max_node = max(max_node, int(node))
        self._new_node_id = max_node + 1

    def fit(self, data: np.ndarray):
        print(f"---> fit {len(data)} signals start (sigma {self.sigma})")

        for i, signal in enumerate(data):
            if i % 500 == 0 and i > 0:
                print(f"---> fit {len(data)} signals in progress processed {i}")

            winner_1_ind = self._get_closest_neurons(signal, position_number=0)

            if winner_1_ind is None or not self._is_vigilance_test_passed(signal, winner_1_ind):
                ind = self._get_new_node_id()
                attrs = {"type": "embryo", "vector": list(signal), "age": 0}
                self.network.add_node(ind, **attrs)
                continue

            winner_2_ind = self._get_closest_neurons(signal, position_number=1)

            if winner_2_ind is None or not self._is_vigilance_test_passed(signal, winner_2_ind):
                ind = self._get_new_node_id()
                attrs = {"type": "embryo", "vector": list(signal), "age": 0}
                self.network.add_node(ind, **attrs)

                attrs = {"age": 0}
                self.network.add_edge(ind, winner_1_ind, **attrs)
                continue

            for node_ind in self.network.nodes:
                epsilon: float = 0
                if node_ind in (winner_1_ind, winner_2_ind):
                    epsilon = self.epsilon_b
                elif self.network.has_edge(node_ind, winner_1_ind) or \
                        self.network.has_edge(node_ind, winner_2_ind):
                    epsilon = self.epsilon_n
                else:
                    continue
                vector = np.array(self.network.nodes[node_ind]["vector"])
                vector += epsilon * (np.array(signal) - vector)
                self.network.nodes[node_ind]["vector"] = list(vector)

            for neighbor_ind in self.network.neighbors(winner_1_ind):
                self.network[winner_1_ind][neighbor_ind]["age"] += 1

            if self.network.has_edge(winner_1_ind, winner_2_ind):
                self.network[winner_1_ind][winner_2_ind]["age"] = 0
            else:
                attrs = {"age": 0}
                self.network.add_edge(winner_1_ind, winner_2_ind, **attrs)

            edges_to_remove = []
            for neighbor_ind in self.network.neighbors(winner_1_ind):
                if self.network[winner_1_ind][neighbor_ind]["age"] > self.a_max:
                    edges_to_remove.append([winner_1_ind, neighbor_ind])

            for edge in edges_to_remove:
                self.network.remove_edge(*edge)

            nodes_to_remove = []
            for node_ind in nx.isolates(self.network):
                if self.network.nodes[node_ind]["type"] == "mature":
                    nodes_to_remove.append(node_ind)

            for node_ind in nodes_to_remove:
                self.network.remove_node(node_ind)

            for node_ind in self.network.nodes:
                if self.network.has_edge(node_ind, winner_1_ind):
                    self.network.nodes[node_ind]["age"] += 1
                    if self.network.nodes[node_ind]["age"] > self.a_mature:
                        self.network.nodes[node_ind]["type"] = "mature"

        self.build_effective_network()

    def deep_fit(self, data: np.ndarray):
        print(len(data))
        self.fit(data)
        old = 0
        calin = self.calculate_chi(data)
        while old - calin < 0:
            self.sigma *= 0.95
            self.fit(data)
            old = calin
            calin = self.calculate_chi(data)

        clusters = list(nx.connected_components(self.effective_network))
        clusters.sort(key=lambda x: len(x), reverse=True)
        for node in clusters[0]:
            self.network.remove_node(node)
        self.build_effective_network()

    def perform_clustering(self, data: np.ndarray):
        if self.effective_network is None:
            raise Exception("Нейронная сеть не была обучена")

        node_ind_2_cluster_id = {}
        for cluster_id, connected_component in enumerate(nx.connected_components(self.effective_network)):
            for node_ind in connected_component:
                node_ind_2_cluster_id[node_ind] = cluster_id

        result = []
        for signal in data:
            closest_ind = -1
            min_dist = float("inf")
            for node_ind in node_ind_2_cluster_id:
                dist = self._dist(signal, self.effective_network.nodes[node_ind]["vector"])
                if dist < min_dist:
                    min_dist = dist
                    closest_ind = node_ind
            result.append(node_ind_2_cluster_id[closest_ind])

        return np.array(result)

    def calculate_chi(self, data: np.ndarray) -> float:
        avg_data = None
        for signal in data:
            avg_data = signal if avg_data is None else avg_data + signal
        avg_data /= len(data)

        B = 0.0
        for signal in data:
            dist = self._dist(signal, avg_data)
            B += dist * dist

        W = 0.0
        for signal in data:
            closest_neuron_ind = self._get_closest_mature_neuron(signal)
            vector = self.network.nodes[closest_neuron_ind]["vector"]
            dist = self._dist(signal, vector)
            W += dist * dist

        c = self._get_mature_neuron_count()
        n = len(data)

        return (B / (c - 1)) / (W / (n - c))

    def get_clusters_amount(self):
        if self.effective_network is None:
            raise Exception("Нейронная сеть не была обучена")
        return len(list(nx.connected_components(self.effective_network)))

    def _get_closest_neurons(self, n: np.ndarray, position_number: int = 0) -> Optional[int]:
        distance = []
        for ind, attrs in self.network.nodes(data=True):
            vector = attrs["vector"]
            dist = self._dist(n, vector)
            distance.append((ind, dist))
        distance.sort(key=lambda x: x[1])
        ranking = [u for u, dist in distance]
        if len(ranking) < position_number + 1:
            return None
        return ranking[position_number]

    def _get_closest_mature_neuron(self, signal: np.ndarray) -> int:
        closest_ind = -1
        min_dist = float("inf")
        for neuron_ind, attrs in self.network.nodes(data=True):
            if attrs["type"] == "mature":
                dist = self._dist(signal, attrs["vector"])
                if dist < min_dist:
                    min_dist = dist
                    closest_ind = neuron_ind
        return closest_ind

    def _get_mature_neuron_count(self) -> int:
        cnt = 0
        for neuron_ind, attrs in self.network.nodes(data=True):
            if attrs["type"] == "mature":
                cnt += 1
        return cnt

    def _get_new_node_id(self) -> int:
        res = self._new_node_id
        self._new_node_id += 1
        return res

    def _is_vigilance_test_passed(self, signal: np.ndarray, node_ind: int) -> bool:
        return self._dist(signal, self.network.nodes[node_ind]["vector"]) <= self.sigma

    def _dist(self, a: np.ndarray, b: np.ndarray, *argc, **kwargs) -> float:
        if self.distance_metric == "euclidean_distance":
            return IncrementalGrowingNeuralGas.euclidean_distance(a, b, *argc, **kwargs)
        raise Exception(f"Invalid distance_metric: {self.distance_metric}")

    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray, *argc, **kwargs) -> float:
        s: float = 0.0
        for val_a, val_b in zip(a, b):
            s += (val_a - val_b) * (val_a - val_b)
        return math.sqrt(s)


def train_igng_from_files(igng: IncrementalGrowingNeuralGas, files: List[Path]):
    data = []
    for file in files:
        with open(file, "rt") as f:
            for line in f.readlines():
                data.append(
                    np.array(
                        list(map(float, line.split(",")))
                    )
                )
    data = np.array(data)
    igng.deep_fit(data)


def calculate_vectors_from_files(igng: IncrementalGrowingNeuralGas, files: List[Path], dsts: List[Path]):
    file_signals = {}

    for file in files:
        signals = []
        with open(file, "rt") as f:
            for line in f.readlines():
                signals.append(
                    np.array(
                        list(map(float, line.split(",")))
                    )
                )
        file_signals[file] = np.array(signals)

    file_vectors = {}

    for file in file_signals:
        file_vectors[file] = [0 for _ in range(igng.get_clusters_amount())]

        results = igng.perform_clustering(file_signals[file])
        for result in results:
            file_vectors[file][result] += 1

        total_signals = len(file_signals[file])
        for i in range(len(file_vectors[file])):
            file_vectors[file][i] /= total_signals

    for src, dst in zip(files, dsts):
        with open(dst, "wt") as f:
            vector = file_vectors[src]
            vector = Normalizer().fit_transform(np.array([vector]))[0]
            f.write(",".join(list(map(str, vector))))


def perform_clustering_from_files(vector_files: List[Path], eps: float = 2.0, min_samples: int = 2):
    vectors = []
    for file in vector_files:
        with open(file, "rt") as f:
            vec = np.array(list(map(float, f.readline().split(","))))
            vectors.append(vec)

    vectors = StandardScaler().fit_transform(vectors)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(vectors)
    result = dbscan.labels_

    return result


def generate_clustering_report(vector_files: List[Path], labels: np.ndarray, report_dst: Path = Path("report.txt")):
    ret = {}
    for file, cluster_id in zip(vector_files, labels):
        if cluster_id not in ret:
            ret[cluster_id] = []
        ret[cluster_id].append(file)

    with open(report_dst, "wt") as f:
        for cluster_id in ret:
            f.write(f"[ {cluster_id} ]\n")
            for file in ret[cluster_id]:
                f.write(f"{file}\n")
            f.write("\n")


# def visualize_clustering_report():
    # from sklearn.decomposition import PCA
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import pandas as pd
    #
    # pca2D = PCA(n_components=2)
    # # dimensions
    # pca_2D = pca2D.fit_transform(vectors)
    # pca2D_df = pd.DataFrame(data=pca_2D, columns=['x', 'y'])
    #
    # description = []
    # for file in vector_files:
    #     name = os.path.split(file)[-1].split("-")[0]
    #     description.append(name)
    # pca2D_df['cluster'] = result
    #
    # sns.scatterplot(x='x', y='y', hue='cluster', data=pca2D_df)
    # plt.title("PCA")
    # plt.show()


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler

    # data = datasets.make_moons(n_samples=1000, noise=.05)
    # data = datasets.make_blobs(n_samples=2000)
    data = datasets.make_circles(n_samples=2000, noise=.05)
    data = StandardScaler().fit_transform(data[0])

    gng = IncrementalGrowingNeuralGas()
    # gng.fit(data)
    gng.deep_fit(data)

    gng.dump_network_gml()

    import matplotlib.pyplot as plt

    # subax1 = plt.subplot(221)
    # nx.draw(gng.network)
    #
    # subax2 = plt.subplot(222)
    # nx.draw(gng.network)

    subplt_1 = plt.subplot(111)
    for signal in data:
        subplt_1.plot(signal[0], signal[1], marker=".", color='black')

    for node_ind, attrs in gng.network.nodes(data=True):
        if attrs["type"] == "embryo":
            subplt_1.plot(*attrs["vector"], marker=".", color='red')
        else:
            subplt_1.plot(*attrs["vector"], marker="o", color='blue')

    for a, b in gng.network.edges:
        anode = gng.network.nodes[a]
        avec = anode["vector"]
        bnode = gng.network.nodes[b]
        bvec = bnode["vector"]

        if anode["type"] != "mature" or bnode["type"] != "mature":
            continue

        plt.plot(
            [avec[0], bvec[0]],
            [avec[1], bvec[1]],
            'o-',
            color='blue'
        )

    plt.axis('scaled')
    plt.xlabel("Ось X")
    plt.ylabel("Ось Y")

    plt.show()
