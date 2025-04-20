import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import io
import cv2
from typing import List, Tuple

def extract_spectrogram_and_spots(y, sr, min_distance=20, threshold=200) -> List[Tuple[int, int]]:
    
    
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='gray')
    plt.axis('off')
    
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    
    
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)
    
   
    _, thresh_image = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    bright_spots = [(int(x), int(y)) for y, x in np.column_stack(np.where(thresh_image > 0))]

    bright_spots.sort(key=lambda spot: (spot[0], spot[1])) #надо сортировать
    
    filtered_bright_spots=[]
    

    if bright_spots:
        filtered_bright_spots.append(bright_spots[0])
        for spot in bright_spots[1:]:
           
            if len(filtered_bright_spots) <= 10:
                if all(np.linalg.norm(np.array(spot) - np.array(existing_spot)) >= min_distance for existing_spot in filtered_bright_spots):
                    filtered_bright_spots.append(spot)
            else:
                
                distances = [np.linalg.norm(np.array(spot) - np.array(filtered_bright_spots[-i])) for i in range(1, 11)]
                if all(distance >= min_distance for distance in distances):
                    filtered_bright_spots.append(spot)

    
    

    return filtered_bright_spots



def audio_to_array(file_path, start_time, end_time): #кусочек в секундах

    y, sr = librosa.load(file_path, sr=None)
    

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    
    y_segment = y[start_sample:end_sample]
    
    return y_segment, sr

image_path = '../tracks/Бесконечность - Земфира.mp3'



segment, sr = audio_to_array(image_path, 10, 11) 

bright_spots = extract_spectrogram_and_spots(segment, sr, min_distance=100, threshold=200, verbose = True)

print(bright_spots)
