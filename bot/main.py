import asyncio #ассинхронный ввод-вывод
from aiogram import Bot, Dispatcher
from utils.handlers import router
#TODO from service.src.model.basemodel import BaseRecognitionModel

token = '7908037638:AAHYIKclCOY4UhISLiojkkk1Tdl2GxtLlNs'

async def main():
    bot = Bot(token)
    dp = Dispatcher()
    dp.include_router (router) #подключаем из headlers

    await dp.start_polling(bot) #ждем стартовых команд от бота

if __name__=='__main__':
    asyncio.run(main())
