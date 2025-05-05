import wave
import numpy 

def open_file(path):
    with wave.open(path, 'rb') as wav_file:
            channels = wav_file.getnchannels()
            cnt_frames = wav_file.getnframes()
            frames = wav_file.readframes(cnt_frames)
            numpy_array = numpy.frombuffer(frames, dtype=numpy.int16)  
            if channels == 2:
                numpy_array = numpy_array.reshape(-1, 2).T
            return numpy_array
    
