from model.model1 import Model1
from data.s3_DataDealer import S3DataDealer
import os
import numpy as np

def main():
    try:
        datadealer = S3DataDealer("songs/dataset_songs.csv")
        model = Model1(music_library_path=None, datadealer=datadealer)
        
        model_path = "model1_data.pkl"
        if os.path.exists(model_path):
            try:
                model.load(model_path)
                print("Модель загружена")
                for k, v in model.feature_mapping.items():
                    if not isinstance(v, np.ndarray) or v.shape != (model.n_points * 2,):
                        print(f"Обнаружен некорректный вектор для {k}, перестраиваем модель...")
                        raise ValueError("Invalid vector shape in loaded model")
            except Exception as e:
                print(f"Ошибка загрузки модели: {e}, строим новую...")
                model.build_feature_mapping()
                model.save(model_path)
        else:
            print("Построение новой модели...")
            model.build_feature_mapping()
            model.save(model_path)
        
        if not model.is_trained:
            raise RuntimeError("Модель не была обучена корректно")
        
        query_file = "songs/Eminem_-_Without_Me_48018514.mp3"
        if os.path.exists(query_file):
            print("\nТоп-5 похожих песен:")
            results = model.search_similar(query_file, top_k=5)
            
            if not results:
                print("Не найдено подходящих результатов")
            else:
                for i, (song, distance) in enumerate(results, 1):
                    print(f"{i}. {song.artist} - {song.name} (сходство: {1/(distance+1e-6):.2f})")
                
                best_match = results[0][0]
                print(f"\nЛучшее совпадение: {best_match.artist} - {best_match.name}")
        else:
            print(f"Файл {query_file} не найден")
            
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()