import librosa
import numpy as np
import math
from sklearn.preprocessing import Normalizer
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from typing import List
from pathlib import Path

from config import AppConfig


class Processor:
    def __init__(self, config: AppConfig):
        self.config = config

    class SilenceDetector:
        def __init__(self, threshold=20, bits_per_sample=16):
            self.cur_SPL = 0
            self.threshold = threshold
            self.bits_per_sample = bits_per_sample
            self.normal = pow(2.0, bits_per_sample - 1)

        def is_silence(self, chunk):
            self.cur_SPL = self.sound_pressure_level(chunk)
            is_sil = self.cur_SPL < self.threshold
            return is_sil

        def sound_pressure_level(self, chunk):
            value = math.pow(self.local_energy(chunk), 0.5)
            value = value / len(chunk) + 1e-12
            value = 20.0 * math.log(value, 10)
            return value

        def local_energy(self, chunk):
            power = 0.0
            for i in range(len(chunk)):
                sample = chunk[i] * self.normal
                power += sample * sample
            return power

    def vad(self, audio, sampele_rate):
        chunk_size = int(sampele_rate * 0.05)  # 50ms
        index = 0
        sil_detector = Processor.SilenceDetector(15)
        nonsil_audio = []
        while index + chunk_size < len(audio):
            if not sil_detector.is_silence(audio[index: index + chunk_size]):
                nonsil_audio.extend(audio[index: index + chunk_size])
            index += chunk_size
        return np.array(nonsil_audio)

    # FRAMING
    def extract_frames(self, y, sr, rate=0.15, drate=0.005):
        sample_count = math.floor(sr * rate)
        d_sample_count = math.floor(sr * drate)
        res = []
        i = 0
        while i < len(y) - sample_count:
            res.append(y[i:i + sample_count])
            i += d_sample_count
        return res

    def extract_features(self, y, sr, n_mfcc=12):
        mfcc = np.mean(
            librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=n_mfcc,
                n_fft=2048,
            ).T
            , axis=0)

        return mfcc

    def preprocess_signal(self, y, sr):
        res = []
        for frame in self.extract_frames(y, sr):
            vec = self.extract_features(frame, sr)
            res.append(vec)
        return res

    def features_from_audio(self, path, compression_rate=0.1):
        y, sr = librosa.load(path, sr=None)
        y_without_silence = self.vad(y.flatten(), sr)
        features = self.preprocess_signal(y_without_silence, sr)
        features_normalized = Normalizer().fit_transform(features)

        N = math.floor(len(features_normalized) * compression_rate)
        cluster_labels = AgglomerativeClustering(n_clusters=N).fit(features_normalized).labels_
        df = pd.DataFrame(data=features_normalized)
        df["cluster"] = cluster_labels
        df_grouped = df.groupby('cluster', group_keys=True).mean()

        return df_grouped

    def preprocess_audio_files_and_write_to_files(self, srcs: List[Path], dsts: List[Path]):
        if len(srcs) != len(dsts):
            raise Exception("Размеры списков аудиофайлов и файлов с параметрами должны совпадать")

        for src, dst in zip(srcs, dsts):
            print(f"---> processing {src} <---")
            y, sr = librosa.load(src, sr=None)
            y_without_silence = self.vad(y.flatten(), sr)
            mfcc_features = self.preprocess_signal(y_without_silence, sr)
            features_normalized = Normalizer().fit_transform(mfcc_features)
            result_features = features_normalized

            if self.config.compression_ratio is not None:
                compressed_size = math.floor(len(features_normalized) * self.config.compression_ratio)
                cluster_labels = AgglomerativeClustering(n_clusters=compressed_size).fit(features_normalized).labels_
                df = pd.DataFrame(data=features_normalized)
                df["cluster"] = cluster_labels
                df_grouped: pd.DataFrame = df.groupby('cluster', group_keys=True).mean()
                result_features = df_grouped.values.tolist()

            with open(dst, "wt") as dstf:
                for features in result_features:
                    data = [str(f) for f in features]
                    dstf.write(",".join(data) + "\n")
