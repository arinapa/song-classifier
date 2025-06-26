import os
import numpy as np
import torch
import random
import torchaudio
from sklearn.neighbors import NearestNeighbors
from transformers import ClapProcessor, ClapModel
import joblib
import librosa

from model.song import Song
from model.basemodel import BaseRecognitionModel  

def set_seed(seed=80):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class CLAP_KNN_Model(BaseRecognitionModel):

    def __init__(self, music_library_path, datadealer=None, n_neighbors=5, load_from=None):
        super().__init__(music_library_path)
        set_seed(80)

        self.n_neighbors = n_neighbors
        self.datadealer = datadealer
        self.knn = None
        self.embeddings = []
        self.song_paths = []

        if load_from:
            self.load(load_from)
            self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
            self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
            self.model.eval()
        else:
            self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
            self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
            self.model.eval()
            self.build_index(self.music_library_path, datadealer)


    def get_audio_embeddings(self, waveform, sample_rate):
        inputs = self.processor(audios=waveform, sampling_rate=sample_rate, return_tensors="pt")
        with torch.no_grad():
            embedding = self.model.get_audio_features(**inputs)
        return embedding[0].numpy() 

    def get_audio_embedding(self, music_file):
        waveform, sample_rate = librosa.load(music_file, sr=None)
    
        if sample_rate != 48000:
            print(f"Ресемплинг файла {music_file} с {sample_rate} Hz → 48000 Hz")
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=48000)
            sample_rate = 48000  

        return self.get_audio_embeddings(waveform, sample_rate)


    def build_index(self, folder_path, datadealer):
        
        self.embeddings = []
        self.song_paths = []

        for index,_ in datadealer:
            if (datadealer(index) == None):
                print("datadealer returned none")

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
            artist=best_match_path.get("Исполнитель", "Unknown")
            title = best_match_path.get("Название", "Unknown") 
            return Song(path=f"{artist}_{title}", name=title, artist=artist)
        except Exception as e:
            print(f"Error during song recognition: {e}")
            return None

    def search_by_file(self, query_path, top_k=5):
        query_emb = self.get_audio_embedding(query_path).reshape(1, -1)
        distances, indices = self.knn.kneighbors(query_emb, n_neighbors=top_k)
        
        results = []
        for i in range(top_k):
            if indices[0][i] >= 0:
                song_id = self.song_paths[indices[0][i]]
                artist = song_id.get("Исполнитель", "Unknown")
                title = song_id.get("Название", "Unknown")
                results.append((
                    Song(path=f"{artist}_{title}", name=title, artist=artist),
                    distances[0][i]
                ))
        return results


    def save(self, path):
        joblib.dump((self.knn, self.embeddings, self.song_paths), path)

    def load(self, path):
        self.knn, self.embeddings, self.song_paths = joblib.load(path)
