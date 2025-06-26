from aiogram.types import (ReplyKeyboardMarkup, InlineKeyboardMarkup, KeyboardButton, InlineKeyboardButton)

#кнопочки для красоты

main = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text='Распознать песню')],
                           [KeyboardButton(text='SOS')]], 
                           resize_keyboard=True,
                           input_field_placeholder='Выберите кнопку...')

type_model = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text='простейшая модель', callback_data='model_1')],
                            [InlineKeyboardButton(text='ускоренная модель', callback_data='model_2')],
                            [InlineKeyboardButton(text='модель "поиск соседей"', callback_data='model_3')],
                            [InlineKeyboardButton(text='shazam-like модель', callback_data='model_4')]],
                            resize_keyboard=True)