import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Загрузка аудиофайла
file_path = '../tracks/Бесконечность - Земфира.mp3'
  # замените на путь к вашему mp3 файлу
y, sr = librosa.load(file_path, sr=None)

# Получение спектрограммы
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

# Визуализация спектрограммы
plt.figure(figsize=(10, 4))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='coolwarm')
plt.colorbar(format='%+2.0f dB')
plt.title('Спектрограмма')
plt.tight_layout()
plt.savefig('../spectrograms/my_plot.png')


# Выбор 10-секундного фрагмента
duration = 10  # продолжительность в секундах
samples = int(sr * duration)
y_segment = y[:samples]

# Выполнение DFT с использованием numpy
Y = np.fft.fft(y_segment)
Y_magnitude = np.abs(Y)

# Частотная ось
frequencies = np.fft.fftfreq(len(Y), 1/sr)

# Визуализация результата DFT
plt.figure(figsize=(10, 4))
plt.plot(frequencies[:len(frequencies)//2], Y_magnitude[:len(Y_magnitude)//2])  # только положительные частоты
plt.title('DFT (Быстрое преобразование Фурье)')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.xlim(0, 2000)  # ограничим ось частот для лучшей визуализации
plt.grid()
plt.savefig('../spectrograms/dft.png')
