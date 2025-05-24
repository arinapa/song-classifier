from service.src.model.basemodel import BaseRecognitionModel
from service.src.model.song import Song
import os
import torch
import torchaudio
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np
import logging
from tqdm import tqdm # чтобы нескучно было тестить, потом уберем

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ShazamModelWind(BaseRecognitionModel):
    def __init__(self, music_library_path, n_neighbors=1, n_fft=1024, hop_length=512, pooling_steps=3, window_size=10.0):
        super().__init__(music_library_path)
        self.n_neighbors = n_neighbors
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pooling_steps = pooling_steps
        self.window_size = window_size

        self.song_paths = []
        self.fingerprints = []
        self.window_info = []
        self.knn = None
        self.max_length = 0

        self._build_fingerprint_index()

    def _get_fingerprints_from_audio(self, file_path: str) -> list[np.ndarray]:
        try:
            waveform, sample_rate = torchaudio.load(file_path)

            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            window_samples = int(self.window_size * sample_rate)
            total_samples = waveform.size(-1)

            fingerprints = []
            for start in range(0, total_samples, window_samples):
                end = start + window_samples
                if end > total_samples:
                    break

                window = waveform[:, start:end]

                spec_transform = torchaudio.transforms.Spectrogram(
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    power=2
                )
                spectrogram = spec_transform(window).unsqueeze(0)

                for _ in range(self.pooling_steps):
                    spectrogram = F.max_pool2d(spectrogram, kernel_size=(2, 4), stride=(2, 4))

                fingerprint = spectrogram.squeeze().flatten().numpy()
                fingerprints.append(fingerprint)

            return fingerprints

        except Exception as e:
            logger.error(f"Ошибка при создании фингерпринтов для {file_path}: {str(e)}")
            return []

    def _build_fingerprint_index(self):
        logger.info(f"Начинаем построение индекса из {self.music_library_path}")

        audio_files = [
            f for f in os.listdir(self.music_library_path)
            if f.lower().endswith((".wav", ".mp3", ".flac"))
        ]

        if not audio_files:
            logger.warning("В директории нет аудиофайлов!")
            return

        raw_fingerprints = []

        for filename in tqdm(audio_files, desc="Processing songs"):
            full_path = os.path.join(self.music_library_path, filename)
            try:
                fingerprints = self._get_fingerprints_from_audio(full_path)

                if not fingerprints:
                    continue

                for i, fp in enumerate(fingerprints):
                    raw_fingerprints.append(fp)
                    self.window_info.append({
                        'path': full_path,
                        'window_index': i,
                        'start_time': i * self.window_size,
                        'end_time': (i + 1) * self.window_size
                    })

                self.song_paths.append(full_path)

            except Exception as e:
                logger.error(f"Ошибка обработки {full_path}: {str(e)}")

        if not raw_fingerprints:
            logger.error("Нет допустимых фингерпринтов для индексации!")
            return

        # Вычисляем максимальную длину
        self.max_length = max(fp.shape[0] for fp in raw_fingerprints)

        # Дополняем фингерпринты до одинаковой длины, может надо по-другому, я хз
        self.fingerprints = np.vstack([
            np.pad(fp, (0, self.max_length - fp.shape[0]), mode="constant")
            for fp in raw_fingerprints
        ])

        logger.info(f"Индекс построен. Всего окон: {len(self.fingerprints)}")
        logger.info(f"Размерность фингерпринтов: {self.fingerprints.shape}")

        self.knn = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(self.fingerprints)),
            metric="cosine",
            algorithm="auto"
        )
        self.knn.fit(self.fingerprints)

    def __call__(self, music_file: str) -> list[Song] | None:
        if not os.path.exists(music_file):
            logger.error(f"Файл не найден: {music_file}")
            return None

        if self.knn is None:
            logger.error("KNN индекс не инициализирован!")
            return None

        query_fps = self._get_fingerprints_from_audio(music_file)

        if not query_fps:
            logger.error("Не удалось создать фингерпринты для запроса")
            return None

        query_fp = query_fps[0]
        query_fp = np.pad(query_fp, (0, max(0, self.max_length - query_fp.shape[0])), mode="constant")
        query_fp = query_fp[:self.max_length].reshape(1, -1)

        distances, indices = self.knn.kneighbors(query_fp)

        logger.info(f"Результаты поиска:")
        logger.info(f"Расстояния: {distances}")
        logger.info(f"Индексы: {indices}")

        found_songs = []
        for i in range(len(indices[0])):
            if distances[0][i] > 1.0:
                continue

            match_idx = indices[0][i]
            window_data = self.window_info[match_idx]

            found_songs.append(Song(
                file_path=window_data['path'],
                similarity=1 - distances[0][i],
                start_time=window_data['start_time'],
                end_time=window_data['end_time']
            ))

        return found_songs if found_songs else None
