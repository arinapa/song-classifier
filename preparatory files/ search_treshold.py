def search_threshold(y, sr, min_distance, min_dots, max_dots, max_iter=9):
    min_treshold = 0
    max_treshold = 255
    best_threshold = 128 
    
    for _ in range(max_iter):
        if max_treshold - min_treshold <= 1:
            break
            
        mid = (min_treshold + max_treshold) // 2
        bright_spots = extract_spectrogram_and_spots(y, sr, min_distance, mid)
        
        if len(bright_spots) < min_dots:
            max_treshold = mid
        elif len(bright_spots) > max_dots:
            min_treshold = mid
        else:
            return mid
            
        best_threshold = mid
    
    return best_threshold
