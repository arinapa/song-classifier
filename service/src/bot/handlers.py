from aiogram.types import Message
from aiogram.filters import CommandStart, Command
from aiogram import Router, F, types
from pathlib import Path
import json
import os
import logging
from pydub import AudioSegment
from datetime import datetime

from data.s3_DataDealer import S3DataDealer
from model.song import Song
from model.model1 import Model1
from model.modelFAISS import ModelFAISS
from model.shazam_model import ShazamModel
from model.modelnn import CLAP_KNN_Model
from bot.create_bot import bot
import bot.utils.keyboards as kb

logging.basicConfig(level=logging.INFO)
router = Router() #какой функцией обработать команду
datadealer = S3DataDealer("songs/dataset_songs.csv")

user_model = {}
user_number_of_songs = {}

models = {
    'model_1' : Model1(music_library_path=None, datadealer=datadealer),
    'model_2' : ModelFAISS(None, datadealer=datadealer, n_probes=5, load_from="models/model_data_FAISS.pkl"),
    'model_3' : CLAP_KNN_Model(None, None, n_neighbors=5, load_from="models/model_data.pkl"),
    'model_4' : ShazamModel(None, None, 2048, 256, 2, 5.0, (15, 30), 5, 1, 5, load_from="models/model_shazam.pkl" )
}

def get_response(song):
    song_info = datadealer.get_data_by_song_name(song.name if song.name else song.title)
    song_info = song_info[0]
    response = (
        f"Название: {song_info['Название']}\n"
        f"Исполнитель: {song_info['Исполнитель']}\n"
        f"Жанр: {song_info['Жанр']}\n"
        f"Язык: {song_info['Язык']}"
    )
    return response

def handler_audio_main(file_path, user_id):
    if user_model[user_id]=='model_1':
        Model = models[user_model[user_id]]
        Model.load("models/model1_data.pkl")
        main_song = Model(file_path)
    if user_model[user_id]=='model_4':
        main_song = models[user_model[user_id]](file_path,1)[0]
    else:
        main_song = models[user_model[user_id]](file_path)

    return f"Распознанная песня:\n{get_response(main_song)}"

def handler_audio_similar(file_path, user_id):
    similar_songs_names = "Похожие песни:\n"
    if user_model[user_id]=='model_1':
        similar_songs = models[user_model[user_id]].search_similar(file_path, top_k=user_number_of_songs[user_id])
        cnt=1
        for song, dist in similar_songs:
            similar_songs_names += f"Номер: {cnt}\n{get_response (song)}\n\n"
            cnt+=1
    elif user_model[user_id]=='model_4':
        similar_songs = models[user_model[user_id]](file_path, top_k=user_number_of_songs[user_id])
        cnt=1
        for song in similar_songs:
            similar_songs_names += f"Номер: {cnt}\n{get_response (song)}\n\n"
            cnt+=1
    else:
        similar_songs = models[user_model[user_id]].search_by_file(file_path, top_k=user_number_of_songs[user_id])
        cnt=1
        for song, dist in similar_songs:
            similar_songs_names += f"Номер: {cnt}\n{get_response (song)}\n\n"
            cnt+=1
    return similar_songs_names

def create_data_message(message: Message): #только для сообщений
    return {'user_id': message.from_user.id, 'type' : message.content_type, 'text': message.text, 'date': message.date.isoformat()}

def save_to_json(data_message): #история запросов
    file_path = 'bot/users_data/history.json'
    if not os.path.exists(file_path):
        all_messages = []
    else:
        with open(file_path, 'r') as file:
            all_messages = json.load(file)
    all_messages.append(data_message)
    with open(file_path, 'w') as file:
        json.dump(all_messages, file, ensure_ascii = False, indent = 4) #сохраняем все в json

#обработка команд через /
@router.message(CommandStart()) #/start
async def cmd_start (message: Message):
    await message.answer ('Привет! Это телеграм-бот для идентификации трека, играющего в окружении пользователя.', reply_markup = kb.main)

    data_message = create_data_message(message)
    save_to_json(data_message)

@router.message(Command ('SOS')) #/SOS
async def cmd_info (message: Message):
    await message.answer ('В случае неработоспособности бота обращайтесь в техподдержку по номеру +78129123456.')

    data_message = create_data_message(message)
    save_to_json(data_message)

#обработка команд кнопок
@router.message(F.text == 'Старт') #Старт
async def button_start (message: Message):
    await message.answer ('Привет! Это телеграм-бот для идентификации трека, играющего в окружении пользователя.', reply_markup = kb.main)

    data_message = create_data_message(message)
    save_to_json(data_message)

@router.message(F.text == 'SOS') #кнопка техподдержки (может понадобится...)
async def button_info (message: Message):
    await message.answer ('В случае неработоспособности бота обращайтесь в техподдержку по номеру +78129123456.')

    data_message = create_data_message(message)
    save_to_json(data_message)

@router.message(F.text == 'Распознать песню') #кнопка обработки
async def button_info (message: Message):
    await message.reply("Пожалуйста, введите число от 1 до 10 - количество самых похожих песен, которые выдаст бот.")
    
@router.message(lambda message: message.text and message.text.isdigit())  #количество похожих песен
async def button_info (message: types.Message):
    number = int(message.text)
    if 1 <= number <= 10:
        user_number_of_songs[message.from_user.id]=number
        await message.answer ('Выберите модель, по которой будет распознаваться ваша песня:', reply_markup=kb.type_model)
    else:
        await message.reply("Пожалуйста, введите число от 1 до 10.")
    
    data_message = create_data_message(message)
    save_to_json(data_message)
        
@router.callback_query(lambda m: m.data in ['model_1', 'model_2', 'model_3', 'model_4']) #после выбора модели
async def process_callback (callback_query: types.CallbackQuery):
    await bot.send_message(callback_query.from_user.id, 'Отлично! Загрузите файл в формате mp3 или запишите голосовое сообщение с песней, которую хотите распознать.')
    user_model[callback_query.from_user.id]=callback_query.data
    data_message = {'user_id': callback_query.from_user.id, 'type' : 'выбор модели', 'text': callback_query.data, 'date': datetime.now().isoformat()}
    save_to_json(data_message)

@router.message(F.content_type == types.ContentType.VOICE) #Голосовые
async def voice_message_handler(message: Message):
    dir = 'bot/users_data/voice_files'
    Path(dir).mkdir(parents = True, exist_ok = True) #создадим папку
    
    voice = message.voice
    file = await bot.get_file(voice.file_id)
    ogg_path = os.path.join(dir, f"{voice.file_id}.ogg")
    path = os.path.join(dir, f"{voice.file_id}.mp3")

    await bot.download_file(file_path = file.file_path, destination = ogg_path)
    await message.answer ('Загружено! Обрабатываем трек... (это может занять некоторое время)')

    #Сохраняем в формате mp3
    audio = AudioSegment.from_file(ogg_path, format = "ogg")
    audio.export(path, format="mp3")

    #Удяляем файл формата ogg
    os.remove(ogg_path)

    data_message = create_data_message(message)
    save_to_json(data_message)

    main_song_name = handler_audio_main(path, message.from_user.id)
    await message.answer (main_song_name) #возвращаем имя песни
    
    similar_songs_names = handler_audio_similar(path, message.from_user.id)
    await message.answer (similar_songs_names)

@router.message(F.content_type == types.ContentType.AUDIO) #mp3 файлы
async def file_message_handler(message: Message):
    dir = 'bot/users_data/files'
    Path(dir).mkdir(parents = True, exist_ok = True) #создадим папку
    
    audio = message.audio
    file = await bot.get_file(audio.file_id)
    path = os.path.join(dir, audio.file_name)

    await bot.download_file(file_path = file.file_path, destination = path)
    await message.answer ('Загружено! Обрабатываем трек... (это может занять некоторое время)')

    data_message = create_data_message(message)
    save_to_json(data_message)

    main_song_name = handler_audio_main(path, message.from_user.id)
    await message.answer (main_song_name) #возвращаем имя песни
    
    similar_songs_names = handler_audio_similar(path, message.from_user.id)
    await message.answer (similar_songs_names)

@router.message()  # обработка для всех остальных сообщений
async def handle_invalid_input(message: types.Message):
    await message.reply("Пожалуйста, введите корректные данные")
    data_message = create_data_message(message)
    save_to_json(data_message)

