import os
from model.modelFAISS import ModelFAISS
from data.s3_DataDealer import S3DataDealer  

datadealer = S3DataDealer("songs/dataset_songs.csv")

model = ModelFAISS(
    music_library_path=None,  
    datadealer=datadealer,   
    n_probes=5,
    load_from="model_data_FAISS.pkl"  
)

query_file = "songs/Radiohead - Creep.mp3"
if not os.path.exists(query_file):
    raise FileNotFoundError(f"Файл {query_file} не найден!")

result = model(query_file)
if result:
    print(f"Распознано: {result.path}")

similar_songs = model.search_by_file(query_file, top_k=5)
for song, dist in similar_songs:
    print(f"Похожая: {song.path} (расстояние: {dist:.4f})")
