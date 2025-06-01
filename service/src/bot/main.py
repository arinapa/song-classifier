import asyncio #ассинхронный ввод-вывод
from aiogram import Dispatcher
from bot.utils.handlers import router
from bot.create_bot import bot

async def main():
    dp = Dispatcher()
    dp.include_router (router) #подключаем из headlers

    await dp.start_polling(bot) #ждем стартовых команд от бота

if __name__=='__main__':
    asyncio.run(main())
