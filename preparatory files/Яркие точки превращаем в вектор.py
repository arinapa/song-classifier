import cv2
import numpy as np

def find_bright_spots(image_path, output_path, min_distance=20, threshold=200):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    
    _, thresh_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    
    bright_spots = [(int(x), int(y)) for y, x in np.column_stack(np.where(thresh_image > 0))]

    

    # Фильтруем точки по евклидовому расстоянию
    filtered_bright_spots = []
    for spot in bright_spots:
        if all(np.linalg.norm(np.array(spot) - np.array(existing_spot)) >= min_distance for existing_spot in filtered_bright_spots):
            filtered_bright_spots.append(spot)
    
   # Чисто для себя рисуем красные круги вокруг найденных точек и сохраняем картинку
    color_image = cv2.imread(image_path)

    
    for (cX, cY) in filtered_bright_spots:
        cv2.circle(color_image, (cX, cY), 5, (0, 0, 255), -1) 

    
    cv2.imwrite(output_path, color_image)
    # потом это надо удалить

    return filtered_bright_spots


image_path = '../spectrograms/Короткий/spectrogram_segment_4.png'

output_path = '../spectrograms/Короткий/spectrogram_segment_4_with_spots.png'


bright_spots = find_bright_spots(image_path, output_path, min_distance=10, threshold=220) 
# наверное, надо будет написать функцию, которая определяет яркость относительно каждой диаграммки

print("Найденные яркие точки:", bright_spots)
