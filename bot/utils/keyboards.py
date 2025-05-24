from aiogram.types import (ReplyKeyboardMarkup, KeyboardButton)

#кнопочки для красоты

main = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text='Распознать песню')],
                           [KeyboardButton(text='SOS')]], 
                           resize_keyboard=True,
                           input_field_placeholder='Выберите кнопку...')
