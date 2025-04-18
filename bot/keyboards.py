from aiogram.types import (ReplyKeyboardMarkup, KeyboardButton)

#кнопочки для красоты

main = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text='Инфо')],
                           [KeyboardButton(text='Что-то еще'),
                           KeyboardButton(text='Потом будет')]], 
                           resize_keyboard=True,
                           input_field_placeholder='Выберите кнопочку...')
