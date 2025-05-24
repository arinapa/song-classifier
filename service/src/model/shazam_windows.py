from service.src.model.basemodel import BaseRecognitionModel
from service.src.model.song import Song

import os
import torch
import torchaudio
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np


class ShazamModel(BaseRecognitionModel):
    def __init__(self, music_library_path, n_neighbors=1, n_fft=1024, hop_length=512, pooling_steps=3):
        super().__init__(music_library_path)
        self.n_neighbors = n_neighbors
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pooling_steps = pooling_steps

        self.song_paths = []
        self.fingerprints = []

        self.knn = None

        self._build_fingerprint_index()

    def _get_fingerprint(self, file_path):
        waveform, sample_rate = torchaudio.load(file_path)

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        spec_transform = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, power=2)
        spectrogram = spec_transform(waveform)

        # Добавляем batch + channel измерения
        spectrogram = spectrogram.unsqueeze(0)  # [1, 1, freq, time]

        # Max pooling
        for _ in range(self.pooling_steps):
            spectrogram = F.max_pool2d(spectrogram, kernel_size=(2, 4), stride=(2, 4))

        # Убираем batch и channel
        return spectrogram.squeeze().flatten().numpy()

    def _build_fingerprint_index(self):
        for filename in os.listdir(self.music_library_path):
            if filename.endswith(".wav") or filename.endswith(".mp3"):
                full_path = os.path.join(self.music_library_path, filename)
                try:
                    fingerprint = self._get_fingerprint(full_path)
                    self.song_paths.append(full_path)
                    self.fingerprints.append(fingerprint)
                except Exception as e:
                    print(f"Ошибка обработки {full_path}: {e}")

        if self.fingerprints:
            self.fingerprints = np.vstack(self.fingerprints)
            self.knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric="cosine")
            self.knn.fit(self.fingerprints)

    def __call__(self, music_file) -> Song | None:
        if not self.knn:
            return None

        try:
            query_fp = self._get_fingerprint(music_file).reshape(1, -1)
            distances, indices = self.knn.kneighbors(query_fp, n_neighbors=1)
            best_match_path = self.song_paths[indices[0][0]]
            return Song(path=best_match_path)
        except Exception as e:
            print(f"Ошибка распознавания: {e}")
            return None



