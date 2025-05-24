from service.src.model.shazam_windows import ShazamModelWind

music_library_path = "tracks"


model = ShazamModelWind(
    music_library_path=music_library_path,
    n_neighbors=10,        
    n_fft=1024,
    hop_length=512,
    pooling_steps=3,
    window_size=10.2
)



query_audio = "mock_tracks/marm.wav"


results = model(query_audio)

if results:
    print("Найденные треки:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Файл: {result.file_path}")
        print(f"   Сходство: {result.similarity:.2%}")
        print(f"   Время фрагмента: {result.start_time:.1f}–{result.end_time:.1f} сек")
        if result.name:
            print(f"   Название: {result.name}")
        if result.artist:
            print(f"   Исполнитель: {result.artist}")
        if result.album:
            print(f"   Альбом: {result.album}")
        if result.year:
            print(f"   Год: {result.year}")
        if result.link:
            print(f"   Ссылка: {result.link}")
        print()
else:
    print("Похожие треки не найдены.")
