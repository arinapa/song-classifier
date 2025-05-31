import pandas
import io
import csv
import librosa
import numpy as np
import boto3 

class BaseDataDealer:
    def __init__(self, csv_path, credits=None):
        try:
            self.media_data = pandas.read_csv(csv_path)
        except FileNotFoundError:
            self.media_data = None
        self.creds = credits
        self.path=csv_path

    def __len__(self):
        return len(self.media_data)

    def __getitem__(self, index):
        pass  # должен возвращать данные о песне и байтовый поток файла / numpy массив с waveform/спектрограммой (как удобнее/как решите/сделайте настройку у класса)

    def __iter__(self, ):
        yield None

    def __call__(self, song_id):
        pass  # байтовый поток файла / numpy массив с waveform/спектрограммой (как удобнее/как решите/сделайте настройку у класса)

    def get_song_list(self,):
        pass

    def get_song(self, name_song):
        pass
