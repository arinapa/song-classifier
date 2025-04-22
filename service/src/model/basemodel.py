from service.src.model.song import Song

class BaseRecognitionModel:
    def __init__(self, music_library_path):
        self.music_library_path = music_library_path
    def __call__(self, music_file) -> Song | None: 
        return None