from aiogram.types import Message, ContentType
from aiogram.filters import CommandStart, Command
from aiogram import Router, F
import json
import os
import logging

import utils.keyboards as kb
import utils.voices as v

logging.basicConfig(level=logging.INFO)

router = Router() #какой функцией обработать команду

def save_to_json(message):
    with open('bot.json', 'r') as file:
        all_messages=json.load (file)
    all_messages.append(message)
    with open('bot.json', 'w') as file:
        json.dump(all_messages,file,ensure_ascii=False, indent=4) #сохраняем все в json

@router.message() #обрабатывает каждое сообщение, чтобы его сохранить в json
async def echo(message: Message):
    data_text = {'user_id': message.from_user.id, 'text': message.text, 'date': message.date.isoformat()}
    save_to_json(data_text)

#обработка команд через /
@router.message(CommandStart()) #/start
async def cmd_start (message: Message):
    await message.answer ("Запуск...", reply_markup=kb.main)

@router.message(Command ('info')) #/info
async def cmd_info (message: Message):
    await message.answer ('Телеграм-бот для идентификации трека, играющего в окружении пользователя.')

#обработка команд кнопок
@router.message(F.text == 'Инфо') #Инфо
async def cmd_info (message: Message):
    await message.answer ('Телеграм-бот для идентификации трека, играющего в окружении пользователя.')

@router.message(F.content_type == [ContentType.VOICE])
async def voice_message_handler(message: Message):
    voice = await message.voice.get_file()
    path = "/files/voices"

    await v.handle_file(file=voice, file_name=f"{voice.file_id}.ogg", path=path)

#@router.message() #декоратор
#async def cmd_start (message: Message):
#    await message.answer ("Запуск") #что делаем при получении соо
