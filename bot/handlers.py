from aiogram.types import Message
from aiogram.filters import CommandStart, Command
from aiogram import Router, F

import utils.keyboards as kb

router = Router() #какой функцией обработать команду

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


#@router.message() #декоратор
#async def cmd_start (message: Message):
#    await message.answer ("Запуск") #что делаем при получении соо
