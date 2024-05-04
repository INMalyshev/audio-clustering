import os.path


report_path = "report.txt"


def get_cluster_readers(path):
    cluster_readers = {}
    cluster_id = None
    with open(path, "rt") as f:
        for line in f.readlines():
            if line.startswith("[ "):
                cluster_id = int(line.strip().lstrip("[").rstrip("]"))
                cluster_readers[cluster_id] = {}
            elif len(line) > 1:
                reader_id = int(os.path.split(line)[-1].split("-")[0])
                if reader_id not in cluster_readers[cluster_id]:
                    cluster_readers[cluster_id][reader_id] = 0
                cluster_readers[cluster_id][reader_id] += 1
    return cluster_readers


def calculate_properties(cluster_readers):
    cluster_reader = {}
    clusters_count = 0
    for cluster in cluster_readers:
        if cluster == -1:
            continue
        clusters_count += 1
        max_count = 0
        max_reader = None
        repeats_count = 0
        for reader in cluster_readers[cluster]:
            if cluster_readers[cluster][reader] == max_count:
                repeats_count += 1
            elif cluster_readers[cluster][reader] > max_count:
                repeats_count = 0
                max_reader = reader
                max_count = cluster_readers[cluster][reader]
        if repeats_count > 0:
            cluster_reader[cluster] = None
        else:
            cluster_reader[cluster] = max_reader

    total = 0
    unrecognized = 0
    positive = 0
    negative = 0
    for cluster in cluster_readers:
        for reader in cluster_readers[cluster]:
            file_count = cluster_readers[cluster][reader]

            total += file_count
            if cluster == -1:
                unrecognized += file_count
            elif reader == cluster_reader[cluster]:
                positive += file_count
            else:
                negative += file_count

    reader_count = {}
    undefined_clusters_count = 0
    for cluster in cluster_reader:
        reader = cluster_reader[cluster]
        if reader is None:
            undefined_clusters_count += 1
        else:
            if reader not in reader_count:
                reader_count[reader] = 0
            reader_count[reader] += 1

    duplicate_clusters_count = 0
    for reader in reader_count:
        count = reader_count[reader]
        if count > 1:
            duplicate_clusters_count += count - 1

    return {
        "total": total, # Всего аудиофайлов
        "unrecognized": unrecognized, # Некластеризованных аудиофайлов
        "positive": positive, # Правильно кластеризованных аудиофайлов
        "negative": negative, # Неправильно кластеризованных аудиофайлов
        "clusters_count": clusters_count, # Всего кластеров
        "undefined_clusters_count": undefined_clusters_count, # Кластеров у которых нет однозначной моды
        "duplicate_clusters_count": duplicate_clusters_count, # Кластеров, которые должны быть частью других кластеров
    }


def calculate_metrics(properties):
    total = properties["total"]
    unrecognized = properties["unrecognized"]
    positive = properties["positive"]
    negative = properties["negative"]
    clusters_count = properties["clusters_count"]
    undefined_clusters_count = properties["undefined_clusters_count"]
    duplicate_clusters_count = properties["duplicate_clusters_count"]

    negative_rate = negative / total if total != 0 else None
    positive_rate = positive / total if total != 0 else None
    unrecognized_rate = unrecognized / total if total != 0 else None
    undefined_clusters_rate = undefined_clusters_count / clusters_count if clusters_count != 0 else None
    duplicate_clusters_rate = duplicate_clusters_count / clusters_count if clusters_count != 0 else None

    return {
        "negative_rate": negative_rate,
        "positive_rate": positive_rate,
        "unrecognized_rate": unrecognized_rate,
        "undefined_clusters_rate": undefined_clusters_rate,
        "duplicate_clusters_rate": duplicate_clusters_rate,
    }


def analyze_report(path):
    cluster_readers = get_cluster_readers(path)
    properties = calculate_properties(cluster_readers)
    metrics = calculate_metrics(properties)

    return metrics


if __name__ == "__main__":
    metrics = analyze_report(report_path)

    from pprint import pprint

    pprint(metrics)
