import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os

def extract_spectrograms(name, file_path, output_dir, segment_duration=1):
    y, sr = librosa.load(file_path, sr=None) # y -- массив с аудиоданными

    total_samples = len(y) 
    samples_per_segment = sr * segment_duration
    num_segments = total_samples // samples_per_segment
    for i in range(num_segments):


        start_sample = i * samples_per_segment
        end_sample = start_sample + samples_per_segment
        y_segment = y[start_sample:end_sample]

        D = librosa.stft(y_segment)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)


    

        # Визуализация спектрограммы
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='gray')
        # plt.colorbar(format='%+2.0f dB')
        #bplt.title(f'Спектрограмма сегмента {i + 1}')
        # plt.tight_layout()
        plt.axis('off')

        track_folder = os.path.join(output_dir, name)
        if not os.path.exists(track_folder):
            os.makedirs(track_folder)

        plt.savefig(f'{output_dir}/{name}/spectrogram_segment_{i + 1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

extract_spectrograms('Короткий', '../tracks/Короткий.mp3', '../spectrograms')







