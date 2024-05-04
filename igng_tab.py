from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import os

from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QFileDialog,
    QWidget,
    QPushButton,
    QListWidget,
    QLabel,
    QTabWidget,
    QTabWidget,
    QVBoxLayout,
    QLineEdit,
    QHBoxLayout,
    QSizePolicy,
)
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtCore import pyqtSlot, QLocale

from config import AppConfig
from igng import (
    IncrementalGrowingNeuralGas,
    train_igng_from_files,
    calculate_vectors_from_files,
    perform_clustering_from_files,
    generate_clustering_report,
)
from utils import load_features, convert_to_2d
from visualisation import visualize_igng, visualize_2d_vectors
import app_messages


@dataclass
class GNGSettings:
    sigma: float = 0.1
    epsilon_b: float = 0.1
    epsilon_n: float = 0.05
    a_max: int = 10
    a_mature: int = 4

    def __str__(self):
        return f"GNGSettings({self.sigma}, {self.epsilon_b}, {self.epsilon_n}, {self.a_max}, {self.a_mature})"


class IgngSettingsWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.default_gng_settings = GNGSettings()
        self.effective_gng_settings = GNGSettings()

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        double_validator = QDoubleValidator()
        double_validator.setLocale(QLocale(QLocale.English))

        int_validator = QIntValidator()

        # Настройки растущего нейронного газа
        self.layout.addWidget(QLabel("Параметры алгоритма расширяющегося нейронного газа"), 0, 0, 1, 2)

        # Максимальное расстояние между соседними нейронами
        self.sigma_input = QLineEdit()
        self.sigma_input.setValidator(double_validator)
        self.layout.addWidget(QLabel("sigma"), 1, 0)
        self.layout.addWidget(self.sigma_input, 1, 1)
        self.sigma_input.setToolTip("Максимальное расстояние между соседними нейронами")

        # Коэффициент корректировки ближайшего нейрона
        self.epsilon_b_input = QLineEdit()
        self.epsilon_b_input.setValidator(double_validator)
        self.layout.addWidget(QLabel("epsilon_b"), 2, 0)
        self.layout.addWidget(self.epsilon_b_input, 2, 1)
        self.epsilon_b_input.setToolTip("Коэффициент корректировки ближайшего нейрона")

        # Коэффициент корректировки связанных с ближайшим нейронов
        self.epsilon_n_input = QLineEdit()
        self.epsilon_n_input.setValidator(double_validator)
        self.layout.addWidget(QLabel("epsilon_n"), 3, 0)
        self.layout.addWidget(self.epsilon_n_input, 3, 1)
        self.epsilon_n_input.setToolTip("Коэффициент корректировки связанных с ближайшим нейронов")

        # Максимальный возраст связи
        self.a_max_input = QLineEdit()
        self.a_max_input.setValidator(int_validator)
        self.layout.addWidget(QLabel("a_max"), 4, 0)
        self.layout.addWidget(self.a_max_input, 4, 1)
        self.a_max_input.setToolTip("Максимальный возраст связи")

        # Возраст значащего нейрона
        self.a_mature_input = QLineEdit()
        self.a_mature_input.setValidator(int_validator)
        self.layout.addWidget(QLabel("a_mature"), 5, 0)
        self.layout.addWidget(self.a_mature_input, 5, 1)
        self.a_mature_input.setToolTip("Возраст значащего нейрона")

        # Сброс настроек алгоритма растущего нейронного газа
        self.reset_gng_settings_btn = QPushButton("Сбросить настройки")
        self.layout.addWidget(self.reset_gng_settings_btn, 6, 0, 1, 2)
        self.reset_gng_settings_btn.clicked.connect(self.reset_gng_settings)

        self.update_gng_settings(GNGSettings())

    def show_gng_settings(self):
        self.sigma_input.setText(str(self.effective_gng_settings.sigma))
        self.epsilon_b_input.setText(str(self.effective_gng_settings.epsilon_b))
        self.epsilon_n_input.setText(str(self.effective_gng_settings.epsilon_n))
        self.a_max_input.setText(str(self.effective_gng_settings.a_max))
        self.a_mature_input.setText(str(self.effective_gng_settings.a_mature))

    def get_gng_settings(self) -> GNGSettings:
        self.effective_gng_settings.sigma = float(self.sigma_input.text())
        self.effective_gng_settings.epsilon_b = float(self.epsilon_b_input.text())
        self.effective_gng_settings.epsilon_n = float(self.epsilon_n_input.text())
        self.effective_gng_settings.a_max = int(self.a_max_input.text())
        self.effective_gng_settings.a_mature = int(self.a_mature_input.text())

        return self.effective_gng_settings

    def update_gng_settings(self, spec: GNGSettings):
        if spec.sigma is not None:
            self.effective_gng_settings.sigma = spec.sigma

        if spec.epsilon_b is not None:
            self.effective_gng_settings.epsilon_b = spec.epsilon_b

        if spec.epsilon_n is not None:
            self.effective_gng_settings.epsilon_n = spec.epsilon_n

        if spec.a_max is not None:
            self.effective_gng_settings.a_max = spec.a_max

        if spec.a_mature is not None:
            self.effective_gng_settings.a_mature = spec.a_mature

        self.show_gng_settings()

    def reset_gng_settings(self):
        self.update_gng_settings(GNGSettings())

        self.show_gng_settings()


@dataclass
class DBSCANSettings:
    epsilon: float = 2.0
    min_samples: int = 2

    def __str__(self):
        return f"DBSCANSettings({self.epsilon}, {self.min_samples})"


class DBSCANSettingsWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.default_settings = DBSCANSettings()
        self.effective_settings = DBSCANSettings()

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        double_validator = QDoubleValidator()
        double_validator.setLocale(QLocale(QLocale.English))

        int_validator = QIntValidator()

        self.layout.addWidget(QLabel("Параметры алгоритма DBSCAN"), 0, 0, 1, 2)

        self.epsilon_input = QLineEdit()
        self.epsilon_input.setValidator(double_validator)
        self.layout.addWidget(QLabel("epsilon"), 1, 0)
        self.layout.addWidget(self.epsilon_input, 1, 1)

        self.min_samples_input = QLineEdit()
        self.min_samples_input.setValidator(int_validator)
        self.layout.addWidget(QLabel("min_samples"), 2, 0)
        self.layout.addWidget(self.min_samples_input, 2, 1)

        self.reset_settings_btn = QPushButton("Сбросить настройки")
        self.layout.addWidget(self.reset_settings_btn, 3, 0, 1, 2)
        self.reset_settings_btn.clicked.connect(self.reset_settings)

        self.update_settings(DBSCANSettings())

    def show_settings(self):
        self.epsilon_input.setText(str(self.effective_settings.epsilon))
        self.min_samples_input.setText(str(self.effective_settings.min_samples))

    def get_settings(self) -> DBSCANSettings:
        self.effective_settings.epsilon = float(self.epsilon_input.text())
        self.effective_settings.min_samples = int(self.min_samples_input.text())
        return self.effective_settings

    def update_settings(self, spec: DBSCANSettings):
        if spec.epsilon is not None:
            self.effective_settings.epsilon = spec.epsilon

        if spec.min_samples is not None:
            self.effective_settings.min_samples = spec.min_samples

        self.show_settings()

    def reset_settings(self):
        self.update_settings(DBSCANSettings())

        self.show_settings()


class IgngTrainingWidget(QWidget):
    def __init__(self, config: AppConfig, *args, **kwargs):
        self.config = config

        super().__init__(*args, **kwargs)

        layout = QGridLayout()
        self.setLayout(layout)

        self.selected_files = []

        layout.addWidget(QLabel("Список выбранных файлов"), 0, 0, 1, 2)

        self.selected_file_list = QListWidget(self)
        self.selected_file_list.setSortingEnabled(True)
        layout.addWidget(self.selected_file_list, 1, 0, 1, 2)

        self.select_files_btn = QPushButton("Выбрать файлы")
        self.select_files_btn.clicked.connect(self._select_files_handler)
        layout.addWidget(self.select_files_btn, 3, 0, 1, 1)

        self.reset_selected_files_btn = QPushButton("Отменить выбор")
        self.reset_selected_files_btn.clicked.connect(self._reset_selected_files_handler)
        layout.addWidget(self.reset_selected_files_btn, 3, 1, 1, 1)

        self.perform_training_btn = QPushButton("Выполнить обучение нейронной сети")
        layout.addWidget(self.perform_training_btn, 4, 0, 1, 2)

        self.visualize_network_btn = QPushButton("Визуализировать нейронную сеть")
        layout.addWidget(self.visualize_network_btn, 5, 0, 1, 2)

        self._load_feature_files_to_selected()
        self._show_selected_files()

    def _perform_training_handler(self):
        pass

    def _load_feature_files_to_selected(self):
        self.selected_files = []
        features_dir = os.path.join(self.config.data_dir, self.config.preprocessed_files_subdir)
        for feature_file in os.listdir(features_dir):
            feature_path = os.path.join(features_dir, feature_file)
            self.selected_files.append(feature_path)

    def _reset_selected_files_handler(self):
        self._load_feature_files_to_selected()
        self._show_selected_files()

    def _show_selected_files(self):
        self.selected_file_list.clear()
        self.selected_file_list.addItems(self.selected_files)

    def _select_files_handler(self):
        features_dir = os.path.join(self.config.data_dir, self.config.preprocessed_files_subdir)
        dialog = QFileDialog(self)
        dialog.setDirectory(features_dir)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("Features (*.features)")
        dialog.setViewMode(QFileDialog.ViewMode.List)
        if dialog.exec():
            filenames = dialog.selectedFiles()
            self.selected_files = filenames

            self._show_selected_files()


class IgngVectorsWidget(QWidget):
    def __init__(self, config: AppConfig, *args, **kwargs):
        self.config = config

        self.vectors: List[Path] = []

        super().__init__(*args, **kwargs)

        layout = QGridLayout()
        self.setLayout(layout)

        layout.addWidget(QLabel("Список выделенных векторов"), 0, 0, 1, 2)

        self.vector_file_list = QListWidget(self)
        self.vector_file_list.setSortingEnabled(True)
        layout.addWidget(self.vector_file_list, 1, 0, 1, 2)

        self.calculate_vectors_btn = QPushButton("Выделить векторы из предобработанных файлов")
        layout.addWidget(self.calculate_vectors_btn, 2, 0, 1, 1)

        self.reset_vectors_btn = QPushButton("Сбросить результат выделения векторов")
        self.reset_vectors_btn.clicked.connect(self._reset_vectors_handler)
        layout.addWidget(self.reset_vectors_btn, 2, 1, 1, 1)

        self.visualize_vectors_btn = QPushButton("Визуализировать выделенные векторы")
        self.visualize_vectors_btn.clicked.connect(self._visualize_vectors_handler)
        layout.addWidget(self.visualize_vectors_btn, 3, 0, 1, 2)

        self.update_vectors()

    def _load_vectors(self):
        vectors_dir = os.path.join(self.config.data_dir, self.config.vector_files_subdir)
        self.vectors = []
        for file in os.listdir(vectors_dir):
            path = os.path.join(vectors_dir, file)
            self.vectors.append(path)

    def _show_vectors(self):
        self.vector_file_list.clear()
        self.vector_file_list.addItems(self.vectors)

    def update_vectors(self):
        self._load_vectors()
        self._show_vectors()

    def _reset_vectors_handler(self):
        vectors_dir = os.path.join(self.config.data_dir, self.config.vector_files_subdir)
        for file in os.listdir(vectors_dir):
            path = os.path.join(vectors_dir, file)
            os.remove(path)
        self.update_vectors()

    def _visualize_vectors_handler(self):
        self.update_vectors()
        files = self.vectors
        vectors = load_features(files)
        vectors_2d = convert_to_2d(vectors)
        kwargs_list = [
            {
                "color": "grey",
                "marker": ".",
            }
            for _ in range(len(vectors_2d))
        ]
        visualize_2d_vectors(vectors_2d, kwargs_list)


class ClusteringWidget(QWidget):
    def __init__(self, config: AppConfig, *args, **kwargs):
        self.config = config
        super().__init__(*args, **kwargs)

        layout = QGridLayout()
        self.setLayout(layout)

        layout.addWidget(QLabel("Кластеризация выделенных векторов"), 0, 0, 1, 2)

        self.cluster_vectors_and_generate_report_btn = QPushButton(
            "Выполнить кластеризацию выделенных векторов и сформировать отчет")
        layout.addWidget(self.cluster_vectors_and_generate_report_btn, 1, 0, 1, 2)

        self.cluster_vectors_and_visualize_btn = QPushButton(
            "Выполнить кластеризацию выделенных векторов и визуализировать результат")
        layout.addWidget(self.cluster_vectors_and_visualize_btn, 2, 0, 1, 2)


class IgngTab(QWidget):
    def __init__(self, config: AppConfig, *args, **kwargs):
        self.config: AppConfig = config
        self.igng: Optional[IncrementalGrowingNeuralGas] = None

        super().__init__(*args, **kwargs)

        main_layout = QGridLayout()
        self.setLayout(main_layout)

        self.igng_training_widget = IgngTrainingWidget(self.config)
        main_layout.addWidget(self.igng_training_widget, 0, 0, 2, 1)
        self.igng_training_widget.setMinimumWidth(600)
        self.igng_training_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.settings_widget = IgngSettingsWidget()
        main_layout.addWidget(self.settings_widget, 0, 1, 1, 1)
        self.settings_widget.setMaximumHeight(200)
        self.settings_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.vectors_widget = IgngVectorsWidget(self.config)
        main_layout.addWidget(self.vectors_widget, 1, 1, 1, 1)
        self.vectors_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.clustering_widget = ClusteringWidget(self.config)
        main_layout.addWidget(self.clustering_widget, 2, 0, 1, 1)
        self.clustering_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.clustering_settings_widget = DBSCANSettingsWidget()
        main_layout.addWidget(self.clustering_settings_widget, 2, 1, 1, 1)
        self.clustering_settings_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.igng_training_widget.perform_training_btn.clicked.connect(self._perform_training_handler)
        self.igng_training_widget.visualize_network_btn.clicked.connect(self._visualise_igng_network_handler)
        self.vectors_widget.calculate_vectors_btn.clicked.connect(self._calculate_vectors_handler)
        self.clustering_widget.cluster_vectors_and_generate_report_btn.clicked.connect(self._perform_clustering_and_generate_report_handler)
        self.clustering_widget.cluster_vectors_and_visualize_btn.clicked.connect(self._perform_clustering_and_visualize_handler)

        self._load_saved_network()

    def _build_igng_from_settings(self):
        gng_settings: GNGSettings = self.settings_widget.get_gng_settings()
        print(gng_settings)
        self.igng = IncrementalGrowingNeuralGas(
            sigma=gng_settings.sigma,
            epsilon_b=gng_settings.epsilon_b,
            epsilon_n=gng_settings.epsilon_n,
            a_max=gng_settings.a_max,
            a_mature=gng_settings.a_mature,
        )

    def _perform_training_handler(self):
        print("--->")
        no_err = True
        self._build_igng_from_settings()
        try:
            train_igng_from_files(self.igng, self.igng_training_widget.selected_files)
        except ZeroDivisionError as e:
            no_err = False
            app_messages.show_info_messagebox(
                title="Неправильно заданы настройки алгоритма кластеризации.",
                message="В результате итериции цикла обучения нейронной сети возникло одно"
                        " из следующих состояний: количество выделенных кластеров равно 1,"
                        " количество выделенных кластеров равно количеству кластеризуемых"
                        " сигналов. Необходимо изменить настройки кластеризации.",
            )
            raise Exception(e)
        except KeyError as e:
            no_err = False
            app_messages.show_info_messagebox(
                title="Неправильно заданы настройки алгоритма кластеризации.",
                message="В результате кластеризации не было выделено кластеров."
                        " Необходимо изменить настройки кластеризации.",
            )
            raise Exception(e)

        if no_err:
            igng_backup_path = Path(os.path.join(self.config.data_dir, self.config.network_backup_name))
            self.igng.dump_network_gml(igng_backup_path)
        print("--->")

    def _visualise_igng_network_handler(self):
        print("---><")
        features = load_features(self.igng_training_widget.selected_files)
        visualize_igng(self.igng, features)
        print(self.igng.get_effective_network_status())
        print("---><")

    def _load_saved_network(self):
        igng_backup_path = Path(os.path.join(self.config.data_dir, self.config.network_backup_name))
        if os.path.exists(igng_backup_path):
            self._build_igng_from_settings()
            self.igng.load_network_gml(igng_backup_path)
            print("Нейронная сеть загружена")

    def _calculate_vectors_handler(self):
        features = self.igng_training_widget.selected_files
        srcs = [self._feature_path_2_vector_path(f) for f in features]
        calculate_vectors_from_files(self.igng, features, srcs)
        self.vectors_widget.update_vectors()

    def _feature_path_2_vector_path(self, path):
        name = os.path.split(path)[-1].split(".")[0] + ".vector"
        path = os.path.join(self.config.data_dir, self.config.vector_files_subdir, name)
        return Path(path)

    def _perform_clustering_and_generate_report_handler(self):
        self.vectors_widget.update_vectors()
        vector_files = self.vectors_widget.vectors

        dbscan_settings: DBSCANSettings = self.clustering_settings_widget.get_settings()
        print(dbscan_settings)
        _labels = perform_clustering_from_files(
            vector_files,
            eps=dbscan_settings.epsilon,
            min_samples=dbscan_settings.min_samples,
        )

        report_path = Path(os.path.join(os.path.abspath(os.path.curdir), "report.txt"))
        generate_clustering_report(vector_files, _labels, report_dst=report_path)

        os.system(f"open {report_path}")

    def _perform_clustering_and_visualize_handler(self):
        colors = ["red", "blue", "black", "green", "purple", "magenta"]
        self.vectors_widget.update_vectors()
        vector_files = self.vectors_widget.vectors

        dbscan_settings: DBSCANSettings = self.clustering_settings_widget.get_settings()
        print(dbscan_settings)
        _labels = perform_clustering_from_files(
            vector_files,
            eps=dbscan_settings.epsilon,
            min_samples=dbscan_settings.min_samples,
        )

        kwargs_list = []
        print(_labels)
        for lbl in _labels:
            kwargs = {}
            if lbl == -1:
                kwargs["label"] = "Undefined"
                kwargs["color"] = "grey"
                kwargs["marker"] = "."
            else:
                kwargs["label"] = f"Cluster {lbl}"
                # kwargs["color"] = colors[lbl % len(colors)]
                kwargs["color"] = "black"
                kwargs["marker"] = f"${lbl}$"
            kwargs_list.append(kwargs)

        vectors = load_features(vector_files)
        vectors_2d = convert_to_2d(vectors)
        visualize_2d_vectors(vectors_2d, kwargs_list)
