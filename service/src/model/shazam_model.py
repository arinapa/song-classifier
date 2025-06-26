import torch
import torchaudio
import torch.nn.functional as F
import hashlib
from collections import defaultdict
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
import joblib
import librosa

from model.basemodel import BaseRecognitionModel
from model.song import Song


class ShazamModel(BaseRecognitionModel):
    def __init__(self, music_library_path: str, datadealer=None,
                 n_fft: int = 2048, hop_length: int = 256,
                 pooling_steps: int = 2, segment_length: float = 5.0,
                 window_size: Tuple[int, int] = (15, 30), fan_value: int = 5,
                 min_time_delta: int = 1, max_time_delta: int = 5,
                 load_from: Optional[str] = None):
        super().__init__(music_library_path)

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pooling_steps = pooling_steps
        self.segment_length = segment_length
        self.window_size = window_size
        self.fan_value = fan_value
        self.min_time_delta = min_time_delta
        self.max_time_delta = max_time_delta

        self.song_db: Dict[int, Song] = {}
        self.fingerprint_db: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.song_fingerprint_counts: Dict[int, int] = defaultdict(int)

        if load_from:
            self.load(load_from)
        else:
            self._build_fingerprint_index(datadealer)

    def _build_fingerprint_index(self, datadealer):
        mock = [3, 4, 5, 28, 29, 30, 31, 34, 45, 46, 47, 48]
        mock_1 = [1]
        for index,_ in datadealer: # datadealer
            song_data, waveform, sample_rate = datadealer(index)
            song_id = index
            song = Song(id=song_id, name=song_data['Название'], path=song_data['Название файла'])
            self.song_db[song_id] = song
            if isinstance(waveform, np.ndarray):
                waveform = torch.from_numpy(waveform).float()
            fingerprints = self._get_fingerprints(waveform, sample_rate, song_id)

            for hash_key, occurrences in fingerprints.items():
                self.fingerprint_db[hash_key].extend(occurrences)
                self.song_fingerprint_counts[song_id] += len(occurrences)
            print(f"Обработана: {song.name} (ID: {song_id}), отпечатков: {self.song_fingerprint_counts[song_id]}")

    def _get_fingerprints(self, waveform: torch.Tensor, sample_rate: int, song_id: int) -> Dict[str, List[Tuple[int, int]]]:
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        window_samples = int(self.segment_length * sample_rate)
        total_samples = waveform.size(-1)
        fingerprints = defaultdict(list)

        for start in range(0, total_samples, window_samples):
            end = start + window_samples
            if end > total_samples:
                break

            segment = waveform[:, start:end]

            spec_transform = torchaudio.transforms.Spectrogram(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                power=2
            )
            spectrogram = spec_transform(segment)

            for _ in range(self.pooling_steps):
                spectrogram = F.max_pool2d(spectrogram, kernel_size=(2, 4), stride=(2, 4))

            log_spec = spectrogram.log2()[0].cpu().numpy()
            log_spec[np.isneginf(log_spec)] = 0

            threshold = np.median(log_spec) + 5
            log_spec[log_spec < threshold] = 0

            X, Y = self.window_size
            local_maxima = np.zeros_like(log_spec, dtype=bool)

            for i in range(0, log_spec.shape[0], X):
                for j in range(0, log_spec.shape[1], Y):
                    window = log_spec[i:i+X, j:j+Y]
                    if window.size == 0:
                        continue
                    max_idx = np.unravel_index(np.argmax(window), window.shape)
                    local_maxima[i + max_idx[0], j + max_idx[1]] = True

            peaks = np.argwhere(local_maxima)

            


            for i in range(len(peaks)):
                for j in range(i + 1, min(i + self.fan_value + 1, len(peaks))):
                    f1, t1 = peaks[i]
                    f2, t2 = peaks[j]
                    delta_t = t2 - t1
                    if self.min_time_delta <= delta_t <= self.max_time_delta:
                        hash_str = f"{f1}|{f2}|{delta_t}"
                        hash_key = hashlib.sha1(hash_str.encode('utf-8')).hexdigest()[:20]
                        fingerprints[hash_key].append((song_id, t1))
        return dict(fingerprints)

    def __call__(self, music_file: str, top_k: int = 3) -> Optional[List[Song]]:
        try:
            waveform, sample_rate = librosa.load(music_file, sr=None)

         
            if sample_rate != 48000:
                print(f"Ресемплинг с {sample_rate} Hz → 48000 Hz")
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=48000)
                sample_rate = 48000

            
            waveform = librosa.util.normalize(waveform)

            
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=0)

            
            waveform = torch.from_numpy(waveform).float()

            query_fingerprints = self._get_fingerprints(waveform, sample_rate, song_id=-1)

            song_match_counts = defaultdict(int)
            time_deltas = defaultdict(list)

            for hash_key, query_occurrences in query_fingerprints.items():
                if hash_key in self.fingerprint_db:
                    for (db_song_id, db_time) in self.fingerprint_db[hash_key]:
                        for (_, query_time) in query_occurrences:
                            song_match_counts[db_song_id] += 1
                            time_deltas[db_song_id].append(db_time - query_time)

            if not song_match_counts:
                print("Совпадений не найдено")
                return None

            scored_songs = []

            for song_id, count in song_match_counts.items():
                if count < 1:
                    continue
                hist, _ = np.histogram(time_deltas[song_id], bins=20)
                max_bin_count = np.max(hist)
                score = max_bin_count / np.sqrt(self.song_fingerprint_counts[song_id] + 1)
                scored_songs.append((song_id, score))

            if not scored_songs:
                print("Надежных совпадений не найдено")
                return None

            scored_songs.sort(key=lambda x: x[1], reverse=True)

            top_matches = scored_songs[:top_k]

            for song_id, score in top_matches:
                print(f"Найдена песня: {self.song_db[song_id].name} (ID: {song_id}), score: {score:.2f}")

            return [self.song_db[song_id] for song_id, _ in top_matches]

        except Exception as e:
            print(f"Ошибка распознавания: {e}")
            return None


    def save(self, path: str):
        data = {
            "fingerprint_db": dict(self.fingerprint_db),
            "song_db": self.song_db,
            "fingerprint_counts": dict(self.song_fingerprint_counts),
        }
        joblib.dump(data, path)
        

    def load(self, path: str):
        data = joblib.load(path)
        self.fingerprint_db = defaultdict(list, data["fingerprint_db"])
        self.song_db = data["song_db"]
        self.song_fingerprint_counts = defaultdict(int, data["fingerprint_counts"])
        
