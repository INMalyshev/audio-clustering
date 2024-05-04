import matplotlib.pyplot as plt
from typing import Optional, List, Dict
from igng import IncrementalGrowingNeuralGas
from utils import convert_to_2d
import numpy as np


def visualize_2d_vectors(vectors: np.ndarray, kwargs_list: Optional[List[Dict]] = None):
    if kwargs_list is None:
        kwargs_list = np.array([{} for _ in range(len(vectors))])

    for vector, kwargs in zip(vectors, kwargs_list):
        if "marker" not in kwargs:
            kwargs["marker"] = "."
        plt.plot(
            vector[0],
            vector[1],
            **kwargs,
        )

    plt.title("Двумерная визуализация данных")
    # plt.axis('scaled')
    plt.xlabel("Ось X")
    plt.ylabel("Ось Y")

    plt.show()


def visualize_igng(igng: IncrementalGrowingNeuralGas, features: np.ndarray):
    nodes = []
    vectors = []
    all = []

    for node, attrs in igng.effective_network.nodes(data=True):
        nodes.append(node)
        vectors.append(attrs["vector"])

    cnt = len(vectors)

    all.extend(vectors)
    all.extend(features)

    all_2d = convert_to_2d(np.array(all))
    vectors = all_2d[:cnt]
    features = all_2d[cnt:]

    node_vector = {}
    for node, vector in zip(nodes, vectors):
        node_vector[node] = vector

    for feature in features:
        plt.plot(
            feature[0],
            feature[1],
            '.',
            color='grey'
        )

    for a, b in igng.effective_network.edges:
        if a in node_vector and b in node_vector:
            plt.plot(
                [node_vector[a][0], node_vector[b][0]],
                [node_vector[a][1], node_vector[b][1]],
                'o-',
                color='black'
            )

    plt.show()




