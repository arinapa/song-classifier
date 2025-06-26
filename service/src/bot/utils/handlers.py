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
# from model.song2 import Song
from model.song import Song
from model.model1 import Model1
from model.modelFAISS import ModelFAISS
from model.shazam_windows import ShazamModelWind
from model.modelnn import CLAP_KNN_Model
from bot.create_bot import bot
import bot.utils.keyboards as kb

user_model = {}
user_number_of_songs = {}

logging.basicConfig(level=logging.INFO)
router = Router() #какой функцией обработать команду

def handler_audio(file_path, user_id):
    if user_model[user_id]==CLAP_KNN_Model:
        Model = user_model[user_id](None, None, n_neighbors=5, load_from="model_data.pkl")
    else:
        Model = user_model[user_id](music_library_path='../Song')
    return Model(file_path)

def handler_audio_similar(file_path, user_id):
    if user_model[user_id]==CLAP_KNN_Model:
        Model = user_model[user_id](None, None, n_neighbors=5, load_from="model_data.pkl")
        similar_songs, distances = Model.search_by_file(file_path, top_k=user_number_of_songs[user_id])
        return zip(similar_songs, distances)
    else:
        Model = user_model[user_id](music_library_path='../Song')
        #TODO обработка других моделей
        return Model(file_path)

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
    await message.answer ('В случае неработоспособности бота обращайтесь в техподдержку по номеру +78121234567.')

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
    await message.answer ('В случае неработоспособности бота обращайтесь в техподдержку по номеру +78121234567.')

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
    if callback_query.data == 'model_1':
        user_model[callback_query.from_user.id]=Model1
    elif callback_query.data == 'model_2':
        user_model[callback_query.from_user.id]=ModelFAISS
    elif callback_query.data == 'model_3':
        user_model[callback_query.from_user.id]=CLAP_KNN_Model
    else:
        user_model[callback_query.from_user.id]=ShazamModelWind
    data_message = {'user_id': callback_query.from_user.id, 'type' : 'выбор модели', 'text': callback_query.data, 'date': datetime.now().isoformat()}
    save_to_json(data_message)

@router.message(F.content_type == types.ContentType.VOICE) #Голосовые
async def voice_message_handler(message: Message):
    dir = 'bot/users_data/voice_files'
    Path(dir).mkdir(parents = True, exist_ok = True) #создадим папку
    
    voice = message.voice
    file = await bot.get_file(voice.file_id)
    path = os.path.join(dir, f"{voice.file_id}.ogg")
    mp3_path = os.path.join(dir, f"{voice.file_id}.mp3")

    await bot.download_file(file_path = file.file_path, destination = path)
    await message.answer ('Загружено! Обрабатываем трек... (это может занять некоторое время)')

    #Сохраняем в формате mp3
    audio = AudioSegment.from_ogg(path)
    audio.export(mp3_path, format="mp3")

    #Удяляем файл формата ogg
    os.remove(path)

    data_message = create_data_message(message)
    save_to_json(data_message)

    song = handler_audio(mp3_path, message.from_user.id)
    await message.answer (song.name, song.artist) #возвращаем имя песни

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

    main_song = handler_audio(path, message.from_user.id)
    main_song_name = "Не распознано"
    if main_song:
        main_song_name = f"Распознанная песня: {main_song.path}"
    await message.answer (main_song_name) #возвращаем имя песни
    
    similar_songs = handler_audio_similar(path, message.from_user.id)
    similar_songs_names = "Похожие песни:\n"
    # print("Похожие песни:")
    cnt=1
    for song, dist in similar_songs:
        similar_songs_names += f"{cnt}. {song['Название файла']}   расстояние: {dist:.4f}\n"
        cnt+=1
        # print(f"- {song['Название файла']} (расстояние: {dist:.4f})")
    
    await message.answer (similar_songs_names)

@router.message()  # обработка для всех остальных сообщений
async def handle_invalid_input(message: types.Message):
    await message.reply("Пожалуйста, введите корректные данные")
    data_message = create_data_message(message)
    save_to_json(data_message)

