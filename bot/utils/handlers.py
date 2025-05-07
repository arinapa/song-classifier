from aiogram.types import Message
from aiogram.filters import CommandStart, Command
from aiogram import Router, F, types
from pathlib import Path
import json
import os
import logging
#TODO from service.src.model.basemodel import BaseRecognitionModel

from create_bot import bot
import utils.keyboards as kb


logging.basicConfig(level=logging.INFO)

router = Router() #какой функцией обработать команду

def save_to_json(message: Message):
    data_message = {'user_id': message.from_user.id, 'type' : message.content_type, 'text': message.text, 'date': message.date.isoformat()}
    file_path = 'users_data/history.json'
    if not os.path.exists(file_path):
        all_messages = []
    else:
        with open(file_path, 'r') as file:
            all_messages = json.load(file)
    all_messages.append(data_message)
    with open(file_path, 'w') as file:
        json.dump(all_messages, file, ensure_ascii = False, indent = 4) #сохраняем все в json

# def handler_audio(){
#     BaseRecognitionModel
# }

#обработка команд через /
@router.message(CommandStart()) #/start
async def cmd_start (message: Message):
    await message.answer ("Запуск...", reply_markup = kb.main)

    save_to_json(message)

@router.message(Command ('info')) #/info
async def cmd_info (message: Message):
    await message.answer ('Телеграм-бот для идентификации трека, играющего в окружении пользователя.')

    save_to_json(message)

#обработка команд кнопок
@router.message(F.text == 'Инфо') #Инфо
async def button_info (message: Message):
    await message.answer ('Телеграм-бот для идентификации трека, играющего в окружении пользователя.')

    save_to_json(message)

@router.message(F.content_type == types.ContentType.VOICE) #Голосовые
async def voice_message_handler(message: Message):
    dir = 'users_data/voice_files'
    Path(dir).mkdir(parents = True, exist_ok = True) #создадим папку
    
    voice = message.voice
    file = await bot.get_file(voice.file_id)
    path = os.path.join(dir, f"{voice.file_id}.ogg")

    await bot.download_file(file_path = file.file_path, destination = path)
    await message.answer ('Загружено!')

    save_to_json(message)

@router.message(F.content_type == types.ContentType.AUDIO) #mp3 файлы
async def file_message_handler(message: Message):
    dir = 'users_data/files'
    Path(dir).mkdir(parents = True, exist_ok = True) #создадим папку
    
    audio = message.audio
    file = await bot.get_file(audio.file_id)
    path = os.path.join(dir, audio.file_name)

    await bot.download_file(file_path = file.file_path, destination = path)
    await message.answer ('Загружено!')

    save_to_json(message)

#@router.message() #декоратор
#async def cmd_start (message: Message):
#    await message.answer ("Запуск") #что делаем при получении соо
