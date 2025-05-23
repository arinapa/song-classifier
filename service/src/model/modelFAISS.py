import os
import glob
import numpy as np
import faiss
from typing import List, Optional
from service.src.model.song import Song
from basemodel import BaseRecognitionModel
import librosa
import cv2
from tqdm.auto import tqdm

class ModelFAISS(BaseRecognitionModel):
    def __init__(self, music_library_path: str, vector_size: int = 512, n_probes: int = 10):
        super().__init__(music_library_path)
        self.music_library_path = music_library_path
        self.vector_size = vector_size
        self.n_probes = n_probes
        self.index, self.song_mapping = self._build_index()

    def _features_to_vector(self, features: List[np.ndarray]) -> np.ndarray:
        """Преобразует сырые признаки в нормализованный вектор фиксированной длины"""
        # Конкатенируем все точки и обрезаем/дополняем до нужного размера
        flat_features = np.concatenate(features).flatten()
        if len(flat_features) > self.vector_size:
            return flat_features[:self.vector_size]
        return np.pad(flat_features, (0, self.vector_size - len(flat_features)))

    def _build_index(self):
        """Строит FAISS индекс из библиотеки музыки"""
        # Собираем все векторы и метки
        vectors = []
        song_ids = []
        
        for artist_song_dir in tqdm(os.listdir(self.music_library_path)):
            artist_song_path = os.path.join(self.music_library_path, artist_song_dir)
            if not os.path.isdir(artist_song_path):
                continue

            # Загружаем и преобразуем признаки
            song_features = []
            for f in glob.glob(f"{artist_song_path}/*.npy"):
                arr = np.load(f)
                song_features.append(arr)
            
            if song_features:
                vector = self._features_to_vector(song_features)
                vectors.append(vector)
                song_ids.append(artist_song_dir)

        if not vectors:
            return None, {}

        # Нормализуем и создаем индекс
        vectors = np.array(vectors).astype('float32')
        faiss.normalize_L2(vectors)
        
        # Используем композитный индекс для эффективности
        quantizer = faiss.IndexFlatIP(self.vector_size)
        index = faiss.IndexIVFFlat(quantizer, self.vector_size, 100)
        index.train(vectors)
        index.add(vectors)
        index.nprobe = self.n_probes
        
        return index, {i: song_id for i, song_id in enumerate(song_ids)}

    def _query_to_vector(self, music_file: str) -> np.ndarray:
        """Преобразует входной файл в вектор признаков"""
        temp_dir = "temp_faiss"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Генерируем спектрограммы
        self.extract_spectrograms("query", music_file, temp_dir, 10)
        
        # Извлекаем признаки
        query_features = []
        for f in glob.glob(f"{temp_dir}/query/*.png"):
            threshold = self.search_threshold(f, 50, 5, 1000)
            spots = self.find_bright_spots(f, min_distance=50, threshold=threshold)
            query_features.append(np.array(spots))
        
        shutil.rmtree(temp_dir)
        return self._features_to_vector(query_features)

    def __call__(self, music_file: str) -> Optional[Song]:
        """Поиск песни с использованием FAISS"""
        if self.index is None:
            return None

        # Преобразуем запрос в вектор
        query_vector = self._query_to_vector(music_file).astype('float32')
        faiss.normalize_L2(query_vector.reshape(1, -1))

        # Ищем в индексе
        distances, indices = self.index.search(query_vector.reshape(1, -1), 1)
        
        if indices.size == 0:
            return None

        # Получаем метку песни
        song_id = self.song_mapping[indices[0][0]]
        artist, title = song_id.split("_", 1)
        
        return Song(artist=artist, title=title)
