import pandas
import io
import re
import csv
import librosa
import numpy as np
import boto3 
from base_DataDealer import BaseDataDealer
from s3_config import ACCESS_KEY, SECRET_KEY, ENDPOINT_URL, REGION_NAME

class S3DataDealer(BaseDataDealer):
    def __init__(self, csv_path):
        creds = {
            'access_key': ACCESS_KEY,
            'secret_key': SECRET_KEY
        }
        super().__init__(csv_path, credits=creds)
        self.endpoint_url = ENDPOINT_URL
        self.region_name = REGION_NAME
        try:
            session = boto3.session.Session()
            self.s3_client = session.client(
                service_name='s3',
                endpoint_url=self.endpoint_url,
                region_name=self.region_name,
                aws_access_key_id=self.creds.get('access_key'),
                aws_secret_access_key=self.creds.get('secret_key')
            )
            
            bucket_name, csv_name = self.path.split('/', 1)
            self.bucket_name = bucket_name
            if not self.s3_client.list_objects_v2(Bucket=bucket_name, Prefix=csv_name).get('KeyCount', 0):
                raise ValueError(f"CSV файл не найден на s3")
            obj = self.s3_client.get_object(Bucket=bucket_name, Key=csv_name)
            csv_stream = io.BytesIO(obj['Body'].read())
            self.media_data = pandas.read_csv(csv_stream)
            self.media_data.columns = self.media_data.columns.str.strip()
        except Exception as e:
            print(f"S3DataDealer не подключился к s3: {str(e)}")
            raise

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

    def __call__(self, song_id):    
        try:
            song_data = self.media_data.loc[song_id].to_dict()
            key = str(song_data.get('Название файла', '')).strip()
            key = re.sub(r'^["\']|["\']$', '', key) 
            check = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=key)
            if check.get('KeyCount', 0) == 0:
                print(f"Файл не найден в S3: {key}")
                return None
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            body = obj['Body'].read()
            audio_stream = io.BytesIO(body)
            waveform, sr = librosa.load(audio_stream, sr=None) 
    
            if sr != 48000: #ресемплинг частоты (выдавало ошибку)
                print(f"Ресемплинг с {sr} Гц до {48000} Гц")
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=48000)
                sr = 48000        
            return song_data, waveform, sr    
        except KeyError as e:
            print(f"Ошибка доступа к песне {song_id}: {str(e)}")
            return None
        except self.s3_client.exceptions.NoSuchKey:
            print(f"Файл не найден в S3: {key}")
            return None
    def get_data_by_song_name(self, song_name, artist=None):
        if self.media_data is None:
            return None    
        try:
            self.media_data.columns = self.media_data.columns.str.strip()        
            mask = self.media_data['Название'].str.lower().str.contains(song_name.lower(), regex=False, na=False)    
            if artist:
                artist_mask = self.media_data['Исполнитель'].str.lower().str.contains(artist.lower(), regex=False, na=False)
                mask = mask & artist_mask
            found_songs = self.media_data[mask]    
            if not found_songs.empty:
                return found_songs.to_dict('records')
            return None     
        except KeyError as e:
            print(f"Ошибка {str(e)}")
            return None
        except Exception as e:
            print(f"Ошибка {str(e)}")
            return None
