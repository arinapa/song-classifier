from pathlib import Path
from aiogram.types import File
from service.src.model.basemodel import BaseRecognitionModel

async def handle_file(bot, file: File, file_name: str, path: str):
    #Path(f"{path}").mkdir(parents=True, exist_ok=True) #тут должен быть путь
    await bot.download_file(file_path=file.file_path, destination=f"{path}/{file_name}")
    #BaseRecognitionModel
