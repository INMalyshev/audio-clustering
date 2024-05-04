import os
from typing import List, Mapping
from pathlib import Path

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
)

from config import AppConfig
from preprocessing import Processor
from utils import load_features, convert_to_2d
from visualisation import visualize_2d_vectors


class PreprocessingTab(QWidget):
    def __init__(self, config: AppConfig, *args, **kwargs):
        self.config: AppConfig = config
        self.processor: Processor = Processor(self.config)

        self.selected_files: List[str] = []
        self.processed_files: List[str] = []

        super().__init__(*args, **kwargs)

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Выбранные файлы
        self.layout.addWidget(QLabel("Список выбранных файлов"), 0, 0, 1, 2)

        self.selected_file_list = QListWidget(self)
        self.selected_file_list.setSortingEnabled(True)
        self.layout.addWidget(self.selected_file_list, 1, 0, 1, 2)

        select_files_button = QPushButton("Выбрать файлы")
        select_files_button.clicked.connect(self._select_files_handler)
        self.layout.addWidget(select_files_button, 2, 0, 1, 1)

        reset_selected_file_list_button = QPushButton("Сбросить выбранные файлы")
        reset_selected_file_list_button.clicked.connect(self._reset_selected_files_handler)
        self.layout.addWidget(reset_selected_file_list_button, 2, 1, 1, 1)

        # Предобработанные файлы
        self.layout.addWidget(QLabel("Список предобработанных файлов"), 3, 0, 1, 2)

        self.processed_file_list = QListWidget(self)
        self.processed_file_list.setSortingEnabled(True)
        self.layout.addWidget(self.processed_file_list, 4, 0, 1, 2)

        process_selected_files_button = QPushButton("Предобработать выбранные файлы")
        process_selected_files_button.clicked.connect(self._process_selected_files_handler)
        self.layout.addWidget(process_selected_files_button, 5, 0, 1, 1)

        reset_processed_files_button = QPushButton("Сбросить предобработанные файлы")
        reset_processed_files_button.clicked.connect(self._reset_processed_files_handler)
        self.layout.addWidget(reset_processed_files_button, 5, 1, 1, 1)

        visualize_processed_files_button = QPushButton("Визуализировать предобработанные файлы")
        visualize_processed_files_button.clicked.connect(self._visualize_processed_files_handler)
        self.layout.addWidget(visualize_processed_files_button, 6, 0, 1, 2)

        # Анализ сохраненного состояния
        self._update_processed_files()

    def _show_selected_files(self):
        self.selected_file_list.clear()
        self.selected_file_list.addItems(self.selected_files)

    def _update_processed_files(self):
        self._load_processed_files()
        self._show_processed_files()

    def _load_processed_files(self):
        self.processed_files = []
        features_dir = os.path.join(self.config.data_dir, self.config.preprocessed_files_subdir)
        for feature_file in os.listdir(features_dir):
            feature_path = os.path.join(features_dir, feature_file)
            self.processed_files.append(feature_path)

    def _show_processed_files(self):
        self.processed_file_list.clear()
        self.processed_file_list.addItems(self.processed_files)

    def _select_files_handler(self):
        dialog = QFileDialog(self)
        dialog.setDirectory(os.path.curdir)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("Audio (*.flac)")
        dialog.setViewMode(QFileDialog.ViewMode.List)
        if dialog.exec():
            filenames = dialog.selectedFiles()
            for filename in filenames:
                file_path = str(Path(filename))
                if file_path not in self.selected_files:
                    self.selected_files.append(file_path)

            self._show_selected_files()

    def _reset_selected_files_handler(self):
        self.selected_files = []
        self._show_selected_files()

    def _process_selected_files_handler(self):
        srcs = [Path(f) for f in self.selected_files]
        dsts = [self._source_path_to_feature_path(f) for f in srcs]
        self.selected_files = []
        self.processor.preprocess_audio_files_and_write_to_files(srcs, dsts)
        self._show_selected_files()
        self._update_processed_files()

    def _reset_processed_files_handler(self):
        features_dir = os.path.join(self.config.data_dir, self.config.preprocessed_files_subdir)
        for file in os.listdir(features_dir):
            feature_path = os.path.join(features_dir, file)
            os.remove(feature_path)
        self._update_processed_files()

    def _visualize_processed_files_handler(self):
        self._update_processed_files()
        vectors = load_features([Path(f) for f in self.processed_files])
        vectors_2d = convert_to_2d(vectors)
        kwargs_list = [
            {
                "color": "grey",
                "marker": ".",
            }
            for _ in range(len(vectors_2d))
        ]
        visualize_2d_vectors(vectors_2d, kwargs_list)

    def _source_path_to_feature_path(self, path: Path):
        name = os.path.split(path)[-1].split(".")[0] + ".features"
        features_dir = os.path.join(self.config.data_dir, self.config.preprocessed_files_subdir)
        new_path = os.path.join(features_dir, name)
        return Path(new_path)
