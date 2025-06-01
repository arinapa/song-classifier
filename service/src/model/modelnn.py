import os
import numpy as np
import torch
import torchaudio
from sklearn.neighbors import NearestNeighbors
from transformers import ClapProcessor, ClapModel
import joblib

from model.song import Song
from model.basemodel import BaseRecognitionModel  

class CLAP_KNN_Model(BaseRecognitionModel):
    def __init__(self, music_library_path, datadealer, n_neighbors=5):


        super().__init__(music_library_path)

        self.n_neighbors = n_neighbors

        self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused")

        self.embeddings = []
        self.song_paths = []
        self.knn = None

        self.datadealer = datadealer
        self.build_index(self.music_library_path, datadealer)


    def get_audio_embeddings(self, waveform, sample_rate):
        inputs = self.processor(audios=waveform, sampling_rate=sample_rate, return_tensors="pt")
        with torch.no_grad():
            embedding = self.model.get_audio_features(**inputs)
        return embedding[0].numpy() 

    def get_audio_embedding(self, music_file):
        waveform, sample_rate = torchaudio.load(music_file)
        return self.get_audio_embeddings(waveform, sample_rate)

    def build_index(self, folder_path, datadealer):
        # собираем эмбеддинги для всех аудиофайлов
        self.embeddings = []
        self.song_paths = []

        for index in datadealer:
            song_data, waveform, sample_rate = datadealer(index)
            emb = self.get_audio_embeddings(waveform, sample_rate)
            self.embeddings.append(emb)
            self.song_paths.append(song_data)

        self.embeddings = np.vstack(self.embeddings)
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric="cosine")
        self.knn.fit(self.embeddings)

    def __call__(self, music_file) -> Song | None:
        if not self.knn:
            return None

        try:
            query_emb = self.get_audio_embedding(music_file).reshape(1, -1)
            distances, indices = self.knn.kneighbors(query_emb, n_neighbors=1)
            best_match_path = self.song_paths[indices[0][0]]
            return Song(path=best_match_path['Название файла']) 
        except Exception as e:
            print(f"Error during song recognition: {e}")
            return None

    def search_by_file(self, query_path, top_k=5):
        query_emb = self.get_audio_embedding(query_path).reshape(1, -1)
        distances, indices = self.knn.kneighbors(query_emb, n_neighbors=top_k)
        return [self.song_paths[i] for i in indices[0]], distances[0]

    def save(self, path):
        joblib.dump((self.knn, self.embeddings, self.song_paths), path)

    def load(self, path):
        self.knn, self.embeddings, self.song_paths = joblib.load(path)
