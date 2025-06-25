
import os
import glob
import numpy as np
from typing import List, Tuple
from service.src.model.song import Song
from basemodel import BaseRecognitionModel
import librosa
import cv2
from tqdm.auto import tqdm

class Model1(BaseRecognitionModel):
    def __init__(self, music_library_path: str):
        super().__init__(music_library_path)
        self.music_library_path = music_library_path

    def extract_spectrograms(self, name: str, file_path: str, output_dir: str, segment_duration: int = 10):
        y, sr = librosa.load(file_path, sr=None)
        total_samples = len(y)
        samples_per_segment = sr * segment_duration
        num_segments = total_samples // samples_per_segment

        for i in range(num_segments):
            start_sample = i * samples_per_segment
            end_sample = start_sample + samples_per_segment
            y_segment = y[start_sample:end_sample]

            D = librosa.stft(y_segment)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

            track_folder = os.path.join(output_dir, name)
            os.makedirs(track_folder, exist_ok=True)
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='gray')
            plt.axis('off')
            plt.savefig(f'{track_folder}/spectrogram_segment_{i + 1}.png', bbox_inches='tight', pad_inches=0)
            plt.close()

    def find_bright_spots(self, image_path: str, min_distance: int = 20, threshold: int = 200) -> List[Tuple[int, int]]:
       
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, thresh_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        bright_spots = [(int(x), int(y)) for y, x in np.column_stack(np.where(thresh_image > 0))]

        filtered_bright_spots = []
        if bright_spots:
            filtered_bright_spots.append(bright_spots[0])
            for spot in bright_spots[1:]:
                if np.linalg.norm(np.array(spot) - np.array(filtered_bright_spots[-1])) >= min_distance:
                    filtered_bright_spots.append(spot)

        return filtered_bright_spots

    def search_threshold(self, image_path: str, min_distance: int, min_dots: int, max_dots: int) -> int:
     
        min_threshold, max_threshold = 0, 255
        no_changes = 2

        while max_threshold - min_threshold > 1 and no_changes >= 2:
            p1_threshold = min_threshold + (max_threshold - min_threshold) // 3
            p2_threshold = min_threshold + (max_threshold - min_threshold) // 3 * 2

            spots_p1 = self.find_bright_spots(image_path, min_distance, p1_threshold)
            spots_p2 = self.find_bright_spots(image_path, min_distance, p2_threshold)

            if len(spots_p1) <= min_dots:
                max_threshold = p1_threshold
                no_changes += 1
            else:
                no_changes -= 1

            if len(spots_p2) >= max_dots:
                min_threshold = p2_threshold
                no_changes += 1
            else:
                no_changes -= 1

        return (min_threshold + max_threshold) // 2

    def find_one_song_features(self, file_name: str) -> List[np.ndarray]:
     
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        self.extract_spectrograms(file_name, file_name, temp_dir, 10)

        lst_files = glob.glob(f"{temp_dir}/{file_name}/*.png")
        threshold = self.search_threshold(lst_files[len(lst_files) // 2], 50, 5, 1000)

        spots_list = []
        for f in lst_files:
            spots = self.find_bright_spots(f, min_distance=50, threshold=threshold)
            spots_list.append(np.array(spots))

        shutil.rmtree(temp_dir)
        return spots_list

    def __call__(self, music_file: str) -> Song | None:
      
        features = self.find_one_song_features(music_file)

        global_metrics = []
        for artist_song_dir in tqdm(os.listdir(self.music_library_path)):
            artist_song_path = os.path.join(self.music_library_path, artist_song_dir)
            if not os.path.isdir(artist_song_path):
                continue

            metrics = []
            for f in glob.glob(f"{artist_song_path}/*.npy"):
                try:
                    arr = np.load(f)
                    metric = np.linalg.norm(features[0][:15] - arr[:15])  # Вычисляем метрику
                    metrics.append(metric)
                except Exception as e:
                    print(f"Ошибка при обработке файла {f}: {e}")

            if metrics:
                global_metrics.append((artist_song_dir, np.min(metrics)))
            else:
                global_metrics.append((artist_song_dir, float('inf')))

        if not global_metrics:
            return None

        # Находим песню с минимальной метрикой
        best_match = min(global_metrics, key=lambda x: x[1])
        artist_song_name = best_match[0]
        artist, song = artist_song_name.split("_", 1)

        return Song(artist=artist, title=song)
