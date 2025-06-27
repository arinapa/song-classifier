import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Добавьте это перед другими импортами
import sys
import time
import numpy as np
import torch
import librosa
from data.s3_DataDealer import S3DataDealer  

from model.modelFAISS import ModelFAISS
from model.song import Song


class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = DualLogger("FAISS_model_test_log.txt") # файл куда сохраним все


def test_model_params(test_data: list, noisy_test_data: list):
    start_time = time.time()
    model = ModelFAISS(music_library_path=None, load_from="model_data_FAISS.pkl")
    index_time = time.time() - start_time
    print(f"[Загрузка модели] Время: {index_time:.2f} сек")

    print("-" * 90)
    total = len(test_data)
    search_time_total = 0

    for file_path in test_data:
        t0 = time.time()
        results = model.search_by_file(file_path, top_k=1)
        t1 = time.time()

        search_time_total += (t1 - t0)
        print(f"Ищем: {file_path}")
        if results:
            song, distance = results[0]  # Получаем первый (и единственный) результат
            print(f"Найдено: {song.path} (расстояние: {distance:.2f})")
            print(f"Время поиска: {t1 - t0:.2f} сек")
        else:
            print("Не распознано")
        print("-" * 45)

    avg_time_clean = search_time_total / total if total else 0
    print(f"[Поиск на чистом аудио] Среднее время: {avg_time_clean:.4f} сек")

    print("-" * 90)
    search_time_total_noisy = 0
    successful_noisy = 0

    for file_path in noisy_test_data:
        t0 = time.time()
        results = model.search_by_file(file_path, top_k=1)
        t1 = time.time()

        search_time_total_noisy += (t1 - t0)
        print(f"Ищем: {file_path}")
        if results:
            song, distance = results[0]
            print(f"Найдено: {song.path} (расстояние: {distance:.2f})")
            print(f"Время поиска: {t1 - t0:.2f} сек")
            successful_noisy += 1
        else:
            print("Не распознано")
        print("-" * 45)

    avg_time_noisy = search_time_total_noisy / len(noisy_test_data) if noisy_test_data else 0
    success_rate = successful_noisy / len(noisy_test_data) if noisy_test_data else 0
    
    print(f"[Поиск на шумном аудио] Среднее время: {avg_time_noisy:.4f} сек")
    print(f"Процент успешных распознаваний: {success_rate*100:.1f}%")

    print("-" * 90)
    return {
        "index_time": index_time,
        "avg_time_clean": avg_time_clean,
        "avg_time_noisy": avg_time_noisy,
        "success_rate": success_rate
    }
import os

def get_files_in_directory(directory_path):
    
    file_list = []
    
    
    if not os.path.exists(directory_path):
        print(f"Директория {directory_path} не существует!")
        return file_list
    
    
    for entry in os.listdir(directory_path):
        
        full_path = os.path.join(directory_path, entry)

        if os.path.isfile(full_path):
            file_list.append(f"{directory_path}/{entry}")
    
    return file_list


if __name__ == "__main__":

    
    test_data = get_files_in_directory("songs") #папка с песнями -- оригиналами
    noisy_test_data = get_files_in_directory("songs_noisy") #кусочки песен 


    test_model_params(test_data, noisy_test_data)
