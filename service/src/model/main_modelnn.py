import os
from model.modelnn import CLAP_KNN_Model
from data.s3_DataDealer import S3DataDealer  

music_library_path = "path/to/your/music/library"


model = CLAP_KNN_Model(None, None, n_neighbors=5, load_from="model_data.pkl" )



query_file = "songs/nerv.mp3"
result = model(query_file)
print(f"Распознанная песня: {result.path if result else 'Не распознано'}")



similar_songs, distances = model.search_by_file(query_file, top_k=10)
print("Похожие песни:")
for song, dist in zip(similar_songs, distances):
    print(f"- {song['Название файла']} (расстояние: {dist:.4f})")

