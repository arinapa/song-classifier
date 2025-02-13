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
