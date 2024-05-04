import os
import json
from pathlib import Path

from pprint import pprint
import matplotlib.pyplot as plt

from igng import IncrementalGrowingNeuralGas, calculate_vectors_from_files, perform_clustering_from_files, generate_clustering_report
from report_analyzer import analyze_report


features_dir = "/Users/malyshevin/bmstu/DIPLOMA/program/app/app_data/features"
vectors_dir = "/Users/malyshevin/bmstu/DIPLOMA/program/app/tmp/vectors"
reports_dir = "/Users/malyshevin/bmstu/DIPLOMA/program/app/tmp/reports"
networks_dir = "/Users/malyshevin/bmstu/DIPLOMA/program/app/networks"
cmp_result_path = "/Users/malyshevin/bmstu/DIPLOMA/program/app/tmp/result.json"


def list_files_from_dir(dir_path):
    files = os.listdir(dir_path)
    res = [Path(os.path.abspath(os.path.join(dir_path, file))) for file in files]
    return res


def get_vector_files(feature_files, dst_dir=vectors_dir):
    res = []
    for path in feature_files:
        file = os.path.split(path)[-1].split(".")[0] + ".vector"
        new_path = Path(os.path.abspath(os.path.join(dst_dir, file)))
        res.append(new_path)
    return res


def build_igng(network_path):
    igng = IncrementalGrowingNeuralGas()
    igng.load_network_gml(network_path)
    return igng


def get_report_path(report_dir, network_path, eps):
    name = os.path.split(network_path)[-1].split(".")[0]
    file = f"{name}-{eps}.report"
    res = os.path.abspath(os.path.join(report_dir, file))
    return Path(res)


def process_network(network_path):
    print(f"---> processing network {network_path} started <---")
    igng = build_igng(network_path)
    feature_paths = list_files_from_dir(features_dir)
    vector_paths = get_vector_files(feature_paths)
    calculate_vectors_from_files(igng, feature_paths, vector_paths)

    eps_from = 0.1
    eps_to = 6.0
    eps_delta = 0.1
    round_digits = 1
    default_min_samples = 2

    result = []

    eps = eps_from
    while eps - eps_to < 0:
        report_path = get_report_path(reports_dir, network_path, round(eps, round_digits))
        labels = perform_clustering_from_files(vector_paths, eps, default_min_samples)
        generate_clustering_report(vector_paths, labels, report_path)
        metrics = analyze_report(report_path)
        gng_sigma = float(str(network_path).split("(")[1].split(",")[0])
        res = {
            "metrics": metrics,
            "dbscan_eps": round(eps, round_digits),
            "dbscan_min_samples": default_min_samples,
            # "network_path": str(network_path),
            "gng_sigma": gng_sigma,
        }
        result.append(res)
        pprint(res)
        eps += eps_delta

    return result


def process_all_networks():
    network_paths = list_files_from_dir(networks_dir)
    results = []

    for path in network_paths:
        result = process_network(path)
        results.extend(result)

    return results


def save_data_to_file(data, path):
    with open(path, "wt") as f:
        for d in data:
            f.write(f"{json.dumps(d)}\n")


def load_data_from_file(path):
    data = []
    with open(path, "rt") as f:
        for line in f.readlines():
            d = json.loads(line)
            data.append(d)
    return data


def get_best_metrics_results(results):
    gng_sigma_2_best_metrics = {}

    for result in results:
        gng_sigma = result["gng_sigma"]
        if gng_sigma not in gng_sigma_2_best_metrics:
            gng_sigma_2_best_metrics[gng_sigma] = {}
        metrics = result["metrics"]
        minimize_rate_metrics = ["duplicate_clusters_rate", "negative_rate", "undefined_clusters_rate", "unrecognized_rate"]
        for metric in minimize_rate_metrics:
            actual_value = metrics[metric]
            if actual_value is None:
                actual_value = 1.0
            best_value = gng_sigma_2_best_metrics[gng_sigma].get(metric, 1.0)
            if best_value >= actual_value:
                gng_sigma_2_best_metrics[gng_sigma][metric] = actual_value
        maximize_rate_metrics = ["positive_rate"]
        for metric in maximize_rate_metrics:
            actual_value = metrics[metric]
            if actual_value is None:
                actual_value = 0.0
            best_value = gng_sigma_2_best_metrics[gng_sigma].get(metric, 0.0)
            if best_value <= actual_value:
                gng_sigma_2_best_metrics[gng_sigma][metric] = actual_value

    return gng_sigma_2_best_metrics


def visualize_results(results):
    sigma_metrics = get_best_metrics_results(results)

    sigmas = list(map(float, list(sigma_metrics.keys())))
    sigmas.sort()
    positive_rates = []
    negative_rates = []
    unrecognized_rates = []
    for sigma in sigmas:
        positive_rates.append(sigma_metrics[sigma]["positive_rate"])
        negative_rates.append(sigma_metrics[sigma]["negative_rate"])
        unrecognized_rates.append(sigma_metrics[sigma]["unrecognized_rate"])

    plt1 = plt.subplot(111)
    plt1.plot(
        sigmas,
        positive_rates,
    )
    plt1.set_xlabel("Значение параметра sigma")
    plt1.set_ylabel("Значение наблюдаемого парметра")
    plt1.set_title("Доля положительных результатов")
    plt.show()

    plt1 = plt.subplot(111)
    plt1.plot(
        sigmas,
        negative_rates,
    )
    plt1.set_xlabel("Значение параметра sigma")
    plt1.set_ylabel("Значение наблюдаемого парметра")
    plt1.set_title("Доля негаливных результатов")
    plt.show()

    plt1 = plt.subplot(111)
    plt1.plot(
        sigmas,
        unrecognized_rates,
    )
    plt1.set_xlabel("Значение параметра sigma")
    plt1.set_ylabel("Значение наблюдаемого парметра")
    plt1.set_title("Доля неопределенных результатов")
    plt.show()


if __name__ == "__main__":
    # results = process_all_networks()
    # save_data_to_file(results, cmp_result_path)

    results = load_data_from_file(cmp_result_path)
    visualize_results(results)
