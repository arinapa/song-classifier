import sys
import time
import numpy as np
import torch
import librosa
from s3_DataDealer import S3DataDealer  

from model1 import Model1
from song import Song


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

sys.stdout = DualLogger("model1_test_log.txt") # файл куда сохраним все

def test_model_params(test_data: list, noisy_test_data: list):
    
    start_time = time.time()
    datadealer = S3DataDealer("songs/dataset_songs.csv")
    model = Model1(music_library_path=None, datadealer=datadealer) 
    model_path = "/home/liliiaxs/song-classifier-1/service/src/model1_data.pkl"# изменить модель
    model.load(model_path)
    index_time = time.time() - start_time
    print(f"[Загрузка модели] Время: {index_time:.2f} сек")
    print("-" * 90)
    total = len(test_data)
    search_time_total = 0

    for file_path in test_data:
        t0 = time.time()
        result = model.search_similar(file_path, top_k=1)
      #  result = model(file_path, top_k=1)
        t1 = time.time()
        i=len(file_path)-1
        pathh=""
        while (file_path[i]!="/") :
            pathh=file_path[i]+pathh
            i-=1
        search_time_total += (t1 - t0)
        print(f"Ищем: {pathh}")
        if result:
            for song in result:
                print(f"Найдено: {song[0].path}")
            print(f"Время поиска: {t1 - t0:.2f} сек")
        else:
            print("Не распознано")
        print("-" * 45)

    avg_time_clean = search_time_total / total if total else 0
    print(f"[Поиск на чистом аудио] Время: {avg_time_clean:.2f} сек")


    print("-" * 90)
    search_time_total_noisy = 0

    for file_path in noisy_test_data:
        t0 = time.time()
        result = model.search_similar(file_path, top_k=1)
        t1 = time.time()
        i=len(file_path)-1
        pathh=""
        while (file_path[i]!="/") :
            pathh=file_path[i]+pathh
            i-=1
        search_time_total_noisy += (t1 - t0)
        print(f"Ищем: {pathh}")
        if result:
            for song in result:
                print(f"Найдено: {song[0].path}")
            print(f"Время поиска: {t1 - t0:.2f} сек")
        else:
            print("Не распознано")
        print("-" * 45)


    
    avg_time_noisy = search_time_total_noisy / len(noisy_test_data)
    print(f"[Поиск на шумном аудио] Время: {avg_time_noisy:.2f} сек")

    print("-" * 90)
    return {
        "index_time": index_time,
        "avg_time_clean": avg_time_clean,
        "avg_time_noisy": avg_time_noisy,
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
    test_data = get_files_in_directory("/home/liliiaxs/song-classifier-1/service/src/songs") #папка с песнями -- оригиналами
    noisy_test_data = get_files_in_directory("/home/liliiaxs/song-classifier-1/service/src/songs_noisy") #кусочки песен 
    test_model_params(test_data, noisy_test_data)
