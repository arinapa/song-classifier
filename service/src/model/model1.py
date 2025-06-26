import os
import numpy as np
import librosa
import cv2
import tempfile
import shutil
import matplotlib.pyplot as plt
import pickle
from tqdm.auto import tqdm
from typing import List, Optional, Tuple, Dict
from data.song import Song
from model.basemodel import BaseRecognitionModel

class Model1(BaseRecognitionModel):
    def __init__(self, music_library_path: str, datadealer=None):
        super().__init__(music_library_path)
        self.music_library_path = music_library_path
        self.datadealer = datadealer
        self.feature_mapping: Dict[str, np.ndarray] = {}
        self.is_trained = False
        
        self.target_sr = 22050
        self.n_points = 20  
        self.segment_duration = 10 
        self.min_distance = 20  
        self.threshold = 200  
        self.expected_shape = (self.n_points * 2,)  

    def save(self, path: str) -> None:
       
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            
            valid_mapping = {}
            for song_id, vec in self.feature_mapping.items():
                if isinstance(vec, np.ndarray) and vec.shape == self.expected_shape:
                    valid_mapping[song_id] = vec
                else:
                    print(f"Пропущен некорректный вектор для {song_id}")

            if not valid_mapping:
                raise ValueError("Нет валидных векторов для сохранения")

            data = {
                'feature_mapping': {k: v.tolist() for k, v in valid_mapping.items()},
                'is_trained': len(valid_mapping) > 0,
                'params': {
                    'target_sr': self.target_sr,
                    'n_points': self.n_points,
                    'segment_duration': self.segment_duration,
                    'min_distance': self.min_distance,
                    'threshold': self.threshold
                }
            }

            temp_path = path + '.tmp'
            with open(temp_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            if os.path.exists(path):
                os.remove(path)
            os.rename(temp_path, path)
            
            print(f"Модель успешно сохранена в {os.path.abspath(path)}")
            print(f"Сохранено векторов: {len(valid_mapping)}")

        except Exception as e:
            print(f"Ошибка сохранения модели: {str(e)}")
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
            raise
    def load(self, path: str) -> None:
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            raise IOError(f"Ошибка загрузки файла модели: {str(e)}")

        params = data.get('params', {})
        self.target_sr = params.get('target_sr', self.target_sr)
        self.n_points = params.get('n_points', self.n_points)
        self.segment_duration = params.get('segment_duration', self.segment_duration)
        self.min_distance = params.get('min_distance', self.min_distance)
        self.threshold = params.get('threshold', self.threshold)
        self.expected_shape = (self.n_points * 2,)

        self.feature_mapping = {}
        corrupted = 0
        
        raw_mapping = data.get('feature_mapping', {})
        for song_id, vec_list in raw_mapping.items():
            try:
                arr = np.array(vec_list, dtype=np.float32).reshape(-1)
                if arr.shape == self.expected_shape:
                    self.feature_mapping[song_id] = arr
                else:
                    print(f"Пропущен {song_id}: неверная форма {arr.shape} (ожидается {self.expected_shape})")
                    corrupted += 1
            except Exception as e:
                print(f"Ошибка обработки вектора {song_id}: {str(e)}")
                corrupted += 1

        self.is_trained = data.get('is_trained', False) and len(self.feature_mapping) > 0
        
        if corrupted > 0:
            print(f"Предупреждение: пропущено {corrupted} некорректных векторов")
        print(f"Успешно загружено {len(self.feature_mapping)} векторов")

    def _extract_spectrogram(self, waveform: np.ndarray, output_path: str) -> bool:

        try:
            plt.figure(figsize=(10, 4), dpi=100)
            D = librosa.stft(waveform, n_fft=2048, hop_length=512)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            librosa.display.specshow(S_db, sr=self.target_sr, x_axis='time', y_axis='log', cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close()
            return True
        except Exception as e:
            print(f"Ошибка создания спектрограммы: {str(e)}")
            return False

    def _get_fixed_points(self, image_path: str) -> np.ndarray:
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return np.zeros((self.n_points, 2))
            
            img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img_blur = cv2.GaussianBlur(img_norm, (5, 5), 0)
            
            corners = cv2.goodFeaturesToTrack(
                img_blur,
                maxCorners=self.n_points,
                qualityLevel=0.01,
                minDistance=self.min_distance
            )
            
            output_points = np.zeros((self.n_points, 2))
            if corners is not None:
                corners = np.squeeze(corners)
                if corners.ndim == 1:
                    corners = np.array([corners])
                valid_points = min(len(corners), self.n_points)
                output_points[:valid_points] = corners[:valid_points]
            
            return output_points
            
        except Exception as e:
            print(f"Ошибка обработки изображения: {str(e)}")
            return np.zeros((self.n_points, 2))

    def _process_audio(self, waveform: np.ndarray, sr: int) -> np.ndarray:
  
        temp_dir = tempfile.mkdtemp()
        try:
            if sr != self.target_sr:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.target_sr)
            
            samples_per_segment = self.target_sr * self.segment_duration
            total_samples = len(waveform)
            num_segments = max(1, total_samples // samples_per_segment)
            all_points = []
            
            for i in range(num_segments):
                start = i * samples_per_segment
                end = min((i + 1) * samples_per_segment, total_samples)
                segment = waveform[start:end]
                
                if len(segment) < samples_per_segment // 2:
                    continue
                
                spec_path = os.path.join(temp_dir, f"spec_{i}.png")
                
                if self._extract_spectrogram(segment, spec_path):
                    points = self._get_fixed_points(spec_path)
                    all_points.append(points)
            
            if len(all_points) > 0:
                mean_points = np.mean(all_points, axis=0)
                if mean_points.shape == (self.n_points, 2):
                    return mean_points.flatten()
            
            return np.zeros(self.expected_shape)
            
        except Exception as e:
            print(f"Ошибка обработки аудио: {str(e)}")
            return np.zeros(self.expected_shape)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def build_feature_mapping(self) -> None:
       
        if not self.datadealer:
            raise ValueError("DataDealer не инициализирован")
        
        print("Построение базы признаков...")
        self.feature_mapping = {}
        success_count = 0
        total = len(self.datadealer.media_data)
        
        if total == 0:
            raise ValueError("DataDealer не содержит данных")
        
        try:
            for idx in tqdm(range(total), total=total):
                try:
                    result = self.datadealer(idx)
                    if result is None:
                        continue
                        
                    song_data, waveform, sr = result
                    song_id = f"{song_data['Исполнитель']}_{song_data['Название']}"
                    
                    features = self._process_audio(waveform, sr)
                    
                    if features.shape == self.expected_shape:
                        self.feature_mapping[song_id] = features
                        success_count += 1
                    else:
                        print(f"Пропущен {song_id}: неверная форма {features.shape}")
                
                except Exception as e:
                    print(f"Ошибка обработки трека {idx}: {str(e)}")
                    continue
            
            self.is_trained = success_count > 0
            print(f"Успешно обработано {success_count}/{total} треков")
            
            if success_count == 0:
                raise RuntimeError("Не удалось обработать ни одного трека")
                
        except Exception as e:
            print(f"Ошибка построения базы: {str(e)}")
            self.is_trained = False
            raise

    def search_similar(self, music_file: str, top_k: int = 5) -> List[Tuple[Song, float]]:

        if not self.is_trained or not self.feature_mapping:
            print("Модель не обучена или база пуста!")
            return []
            
        if not os.path.exists(music_file):
            print(f"Файл {music_file} не найден")
            return []
            
        try:
            waveform, sr = librosa.load(music_file, sr=None)
            query_vec = self._process_audio(waveform, sr)
            
            if query_vec.shape != self.expected_shape:
                print(f"Неверная размерность вектора запроса: {query_vec.shape}")
                return []
            
            results = []
            for song_id, target_vec in self.feature_mapping.items():
                try:
                    distance = np.linalg.norm(query_vec - target_vec)
                    artist, title = song_id.split("_", 1)
                    results.append((
                        Song(path=song_id, name=title, artist=artist),
                        float(distance)
                    ))
                except Exception as e:
                    print(f"Ошибка сравнения с {song_id}: {str(e)}")
                    continue
            
            results.sort(key=lambda x: x[1])
            return results[:min(top_k, len(results))]
            
        except Exception as e:
            print(f"Ошибка при поиске: {str(e)}")
            return []

    def __call__(self, music_file: str) -> Optional[Song]:
        results = self.search_similar(music_file, top_k=1)
        return results[0][0] if results else None
