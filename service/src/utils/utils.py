import wave
def open_file(path): #TODO сделать открытие файлов в нужном формате
    wav_file=wave.open(path, 'rb')
    return wav_file 
