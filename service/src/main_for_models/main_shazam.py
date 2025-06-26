import os
from model.shazam_model_last import ShazamModel
from data.s3_DataDealer import S3DataDealer  

datadealer = S3DataDealer("songs/dataset_songs.csv")

model = ShazamModel(None, load_from="../models/model_shazam.pkl" ) 
# model = ShazamModel(None, datadealer, 2048, 512, 3, 5.0, (15, 30), 5, 1, 3, None) # для обучения и сохранения
# model.save("../models/model_shazam.pkl")


query_file = "../songs/night.mp3"

result = model(query_file, top_k=10)

if result:
    for song in result:
        print(f"Похожая песня: {song.title} — {song.path}") # ну или название можно выводить
else:
    print("Не распознано")
