import os
import numpy as np
import torch
import torchaudio
from sklearn.neighbors import NearestNeighbors
from transformers import ClapProcessor, ClapModel
import joblib

from service.src.model.song import Song
from service.src.model.basemodel import BaseRecognitionModel  

class CLAP_KNN_Model(BaseRecognitionModel):
    def __init__(self, music_library_path, n_neighbors=5):


        super().__init__(music_library_path)

        self.n_neighbors = n_neighbors

        self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused")

        self.embeddings = []
        self.song_paths = []
        self.knn = None


        self.build_index(self.music_library_path)


    def get_audio_embedding(self, file_path):
        waveform, sample_rate = torchaudio.load(file_path)
        inputs = self.processor(audios=waveform, sampling_rate=sample_rate, return_tensors="pt")
        with torch.no_grad():
            embedding = self.model.get_audio_features(**inputs)
        return embedding[0].numpy() 

    def build_index(self, folder_path):
        # собираем эмбеддинги для всех аудиофайлов
        self.embeddings = []
        self.song_paths = []

        for filename in os.listdir(folder_path):
            if filename.endswith(".wav"):
                full_path = os.path.join(folder_path, filename)
                
                emb = self.get_audio_embedding(full_path)
                self.embeddings.append(emb)
                self.song_paths.append(full_path)

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
            return Song(path=best_match_path) 
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
