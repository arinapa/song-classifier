import os
import numpy as np
import faiss
import librosa
from typing import List, Optional
from tqdm.auto import tqdm
from model.song import Song
from model.basemodel import BaseRecognitionModel

class ModelFAISS(BaseRecognitionModel):
    def __init__(
        self,
        music_library_path: str,
        datadealer=None,
        vector_size: int = 512,
        n_probes: int = 10,
        load_from: str = None,
    ):
        super().__init__(music_library_path)
        self.vector_size = vector_size
        self.n_probes = n_probes
        self.datadealer = datadealer
        self.index = None
        self.song_mapping = {}

        if load_from and os.path.exists(f"{load_from}.index"):
            self._load_model(load_from)
        else:
            self._build_index()

    def _load_model(self, path: str):
        self.index = faiss.read_index(f"{path}.index")
        self.song_mapping = dict(np.load(f"{path}_mapping.npz", allow_pickle=True)["mapping"].item())

    def _save_model(self, path: str):
        faiss.write_index(self.index, f"{path}.index")
        np.savez(f"{path}_mapping.npz", mapping=self.song_mapping)

    def _extract_features_from_audio(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        max_samples = 30 * sr
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]

        if len(waveform.shape) > 1:
            waveform = librosa.to_mono(waveform)
        if sr != 22050:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=22050)

        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=22050,
            n_mfcc=20,
            hop_length=512,
        )
        mfcc_delta = librosa.feature.delta(mfcc)
        features = np.vstack([mfcc, mfcc_delta]).flatten()

        return np.pad(
            features,
            (0, max(0, self.vector_size - len(features))),
        )[: self.vector_size]

    def _build_index(self):
        vectors = []
        song_ids = []

        print("Построение индекса FAISS...")
        if self.datadealer:
            for index, _ in tqdm(self.datadealer):
                song_data = self.datadealer(index)
                if song_data is None:
                    continue

                metadata, waveform, sr = song_data
                features = self._extract_features_from_audio(waveform, sr)
                vectors.append(features)
                song_ids.append(f"{metadata['Исполнитель']}_{metadata['Название']}")

        if not vectors:
            raise ValueError("Нет данных для индексации")

        vectors = np.array(vectors).astype("float32")
        faiss.normalize_L2(vectors)

        quantizer = faiss.IndexFlatIP(self.vector_size)
        self.index = faiss.IndexIVFFlat(quantizer, self.vector_size, 100)
        self.index.train(vectors)
        self.index.add(vectors)
        self.index.nprobe = self.n_probes
        self.song_mapping = {i: song_id for i, song_id in enumerate(song_ids)}

    def _query_to_vector(self, music_file: str) -> np.ndarray:
    
        waveform, sr = librosa.load(music_file, sr=None, mono=True)
        return self._extract_features_from_audio(waveform, sr)

    def __call__(self, music_file: str) -> Optional[Song]:
        if self.index is None:
            return None

        try:
            query_vector = self._query_to_vector(music_file).astype("float32")
            faiss.normalize_L2(query_vector.reshape(1, -1))
            distances, indices = self.index.search(query_vector.reshape(1, -1), 1)
            
            if indices.size == 0:
                return None

            song_id = self.song_mapping[indices[0][0]]
            artist, title = song_id.split("_", 1)
            return Song(
                path=f"{artist}_{title}",
                name=title,
                artist=artist
            )
        except Exception as e:
            print(f"Ошибка поиска: {str(e)}")
            return None

    def search_by_file(self, query_path: str, top_k: int = 5) -> List[tuple]:
        try:
            query_vector = self._query_to_vector(query_path).astype("float32")
            faiss.normalize_L2(query_vector.reshape(1, -1))
            distances, indices = self.index.search(query_vector.reshape(1, -1), top_k)
            
            results = []
            for i in range(top_k):
                if indices[0][i] >= 0:
                    song_id = self.song_mapping[indices[0][i]]
                    artist, title = song_id.split("_", 1)
                    results.append((
                        Song(path=f"{artist}_{title}", name=title, artist=artist),
                        distances[0][i]
                    ))
            return results
        except Exception as e:
            print(f"Ошибка поиска похожих: {str(e)}")
            return []
            
    def save(self, path: str):
        if not self.index:
            raise ValueError("Модель не обучена!")
        self._save_model(path)
