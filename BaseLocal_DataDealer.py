import pandas
import io
import csv
import librosa 
import numpy as np
class BaseDataDealer:
    def __init__(self, csv_path, credits=None):
        self.media_data = pandas.read_csv(csv_path)
        self.creds = credits
        self.path=csv_path
    def __len__(self):
        return len(self.media_data)
    def __getitem__(self, index):
        pass # должен возвращать данные о песне и байтовый поток файла / numpy массив с waveform/спектрограммой (как удобнее/как решите/сделайте настройку у класса)
    def __iter__(self, ):
        yield None 
    def __call__(self, song_id):
        pass # байтовый поток файла / numpy массив с waveform/спектрограммой (как удобнее/как решите/сделайте настройку у класса)
    def get_song_list(self,): 
        pass
    def get_song(self, name_song):
        pass

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
            waveform, sr = librosa.load(file_path)
            return waveform 
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

#проверки
if __name__ == "__main__":
    csv_file = "dataset_songs.csv"
    test_data = LocalDataDealer(csv_file)
    print( len(test_data))
    song_data= test_data[166] 
    print(song_data)
    audio_data = test_data(54) 
    if audio_data is not None:
        print("\nWaveform для Tell your friends:")
        print(audio_data)

    song_list = test_data.get_song_list()
    print("Все песни:")
    print(song_list)
    song_title = "Май bye"
    filepath = test_data.get_song(song_title)
    if filepath:
        print(filepath)
    else:
        print(f"Песня '{song_title}' не найдена")
