from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QStackedLayout,
    QFileDialog,
    QWidget,
    QPushButton,
    QListWidget,
    QLabel,
    QTabWidget,
    QMainWindow,
)

from igng_tab import IgngTab
from preprocessing_tab import PreprocessingTab
from config import AppConfig


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = 'Метод кластеризации голосовых аудиозаписей'
        self.left = 0
        self.top = 0
        self.width = 720
        self.height = 200
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.table_widget = MainTabWidget(self)
        self.setCentralWidget(self.table_widget)

        self.show()


class MainTabWidget(QWidget):

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)

        self.config: AppConfig = AppConfig()

        self.layout = QGridLayout(self)

        # Initialize tab screen
        self.tabs = QTabWidget()
        # self.control_tab = ControlTab()
        self.settings_tab = IgngTab(self.config)
        self.preprocessing_tab = PreprocessingTab(self.config)
        # self.tabs.resize(300, 200)

        # Add tabs
        # self.tabs.addTab(self.control_tab, "Управление")
        self.tabs.addTab(self.settings_tab, "Кластеризация")
        self.tabs.addTab(self.preprocessing_tab, "Предобработка")

        # Add tabs to widget
        self.layout.addWidget(self.tabs, 0, 0)
        self.setLayout(self.layout)
