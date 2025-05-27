from aiogram.types import Message
from aiogram.filters import CommandStart, Command
from aiogram import Router, F, types
from pathlib import Path
import json
import os
import logging
from pydub import AudioSegment

from service.src.model.basemodel import BaseRecognitionModel
from service.src.model.model1 import Model1
from service.src.model.modelFAISS import ModelFAISS
# from service.src.model.shazam_windows import ShazamModelWind
from bot.create_bot import bot
import bot.utils.keyboards as kb

BaseModel = BaseRecognitionModel('../Песни проект')

logging.basicConfig(level=logging.INFO)
router = Router() #какой функцией обработать команду

def handler_audio(file_path):
    Model = ModelFAISS('../Песни проект')
    return Model(file_path)

def save_to_json(message: Message): #история запросов
    data_message = {'user_id': message.from_user.id, 'type' : message.content_type, 'text': message.text, 'date': message.date.isoformat()}
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
    await message.answer ("Запуск...", reply_markup = kb.main)

    await message.answer ('Телеграм-бот для идентификации трека, играющего в окружении пользователя.')

    save_to_json(message)

@router.message(Command ('SOS')) #/SOS
async def cmd_info (message: Message):
    await message.answer ('В случае неработоспособности бота обращайтесь в техподдержку по номеру +78121234567.')

    save_to_json(message)

#обработка команд кнопок
@router.message(F.text == 'Старт') #Старт
async def button_info (message: Message):
    await message.answer ('Это телеграм-бот для идентификации трека, играющего в окружении пользователя.')

    save_to_json(message)

@router.message(F.text == 'SOS') #кнопка техподдержки (может понадобится...)
async def button_info (message: Message):
    await message.answer ('В случае неработоспособности бота обращайтесь в техподдержку по номеру +78121234567.')

    save_to_json(message)

@router.message(F.text == 'Распознать песню') #кнопка обработки
async def button_info (message: Message):
    await message.answer ('Загрузите файл в формате mp3 или запишите голосовое сообщение с песней, которую хотите распознать.')

    save_to_json(message)

@router.message(F.content_type == types.ContentType.VOICE) #Голосовые
async def voice_message_handler(message: Message):
    dir = 'bot/users_data/voice_files'
    Path(dir).mkdir(parents = True, exist_ok = True) #создадим папку
    
    voice = message.voice
    file = await bot.get_file(voice.file_id)
    path = os.path.join(dir, f"{voice.file_id}.ogg")
    mp3_path = os.path.join(dir, f"{voice.file_id}.mp3")

    await bot.download_file(file_path = file.file_path, destination = path)
    await message.answer ('Загружено! Обрабатываем трек...')

    #Сохраняем в формате mp3
    audio = AudioSegment.from_ogg(path)
    audio.export(mp3_path, format="mp3")

    #Удяляем файл формата ogg
    os.remove(path)

    save_to_json(message)

    song = handler_audio(mp3_path)
    await message.answer (song.name, song.artist) #возвращаем имя песни

@router.message(F.content_type == types.ContentType.AUDIO) #mp3 файлы
async def file_message_handler(message: Message):
    dir = 'bot/users_data/files'
    Path(dir).mkdir(parents = True, exist_ok = True) #создадим папку
    
    audio = message.audio
    file = await bot.get_file(audio.file_id)
    path = os.path.join(dir, audio.file_name)

    await bot.download_file(file_path = file.file_path, destination = path)
    await message.answer ('Загружено! Обрабатываем трек...')

    save_to_json(message)

    song = handler_audio(path)
    await message.answer (song.name, song.artist) #возвращаем имя песни
