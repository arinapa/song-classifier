import pandas
import io
import csv
import librosa
import numpy as np
import boto3 
from base_DataDealer import BaseDataDealer

class LocalDataDealer(BaseDataDealer):
    def __getitem__(self, index):
        if self.media_data is None:
            return None
        try:
            return self.media_data.iloc[index].to_dict()
        except IndexError:
            print(f"Индекс {index} out of range таблички.")
            return None

    def __iter__(self):
        if self.media_data is None:
            return iter([])
        return self.media_data.iterrows()

    def __call__(self, song_id):
        song_data = self.media_data.iloc[song_id].to_dict()
        file_path = song_data.get('Название файла')
        if file_path:
            try:
                waveform, sr = librosa.load(file_path)
                return waveform
            except Exception as e:
                print(f"Error loading audio file: {e}")
                return None
        else:
            return None

    def get_song_list(self):
        return self.media_data

    def get_song(self, name_song):
        if self.media_data is None:
            return None
        mask = self.media_data['Название'].str.lower() == name_song.lower()
        current_song = self.media_data[mask]
        if not current_song.empty:
            return current_song['Название файла'].iloc[0]
        return None


