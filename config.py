from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Optional


data_dir = os.path.join(os.path.abspath(os.path.curdir), "app_data")


@dataclass
class AppConfig:
    data_dir: Path = data_dir
    preprocessed_files_subdir: Path = Path("features")
    network_backup_name: Path = Path("network.gml")
    vector_files_subdir: Path = Path("vectors")
    compression_ratio: Optional[float] = 0.1
