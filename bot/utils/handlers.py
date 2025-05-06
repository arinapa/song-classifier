from aiogram.types import Message, ContentType
from aiogram.filters import CommandStart, Command
from aiogram import Router, F
import logging

import utils.keyboards as kb
import utils.voices as v

router = Router() #какой функцией обработать команду

@router.errors_handler
async def errors_handler(update, exception):
    logging.error(f'Update: {update} caused error: {exception}')
    return True

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
