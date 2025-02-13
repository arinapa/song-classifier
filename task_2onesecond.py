import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def extract_spectrograms_and_fft(file_path, output_dir, segment_duration=1):
    """
    Извлекает спектрограммы и выполняет FFT для 1-секундных фрагментов аудиофайла.

    :param file_path: Путь к аудиофайлу.
    :param output_dir: Директория для сохранения спектрограмм и графиков FFT.
    :param segment_duration: Длительность сегмента в секундах (по умолчанию 1).
    """
    # Загрузка аудиофайла
    y, sr = librosa.load(file_path, sr=None)
    
    # Вычисление общего количества сегментов
    total_samples = len(y)
    samples_per_segment = sr * segment_duration
    num_segments = total_samples // samples_per_segment

    for i in range(num_segments):
        # Извлечение сегмента
        start_sample = i * samples_per_segment
        end_sample = start_sample + samples_per_segment
        y_segment = y[start_sample:end_sample]

        # Создание спектрограммы
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y_segment)), ref=np.max)

        # Визуализация спектрограммы
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='coolwarm')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Спектрограмма сегмента {i + 1}')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/spectrogram_segment_{i + 1}.png')
        plt.close()

        # Выполнение DFT с использованием numpy
        Y = np.fft.fft(y_segment)
        Y_magnitude = np.abs(Y)

        # Частотная ось
        frequencies = np.fft.fftfreq(len(Y), 1/sr)

        # Визуализация результата DFT
        plt.figure(figsize=(10, 4))
        plt.plot(frequencies[:len(frequencies)//2], Y_magnitude[:len(Y_magnitude)//2])
        plt.title(f'DFT сегмента {i + 1}')
        plt.xlabel('Частота (Гц)')
        plt.ylabel('Амплитуда')
        plt.xlim(0, 2000)  # Ограничиваем ось частот для лучшей визуализации
        plt.grid()
        plt.savefig(f'{output_dir}/dft_segment_{i + 1}.png')
        plt.close()

# Пример использования функции
extract_spectrograms_and_fft('../tracks/Бесконечность - Земфира.mp3', '../spectrograms')
