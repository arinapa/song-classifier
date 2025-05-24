from dataclasses import dataclass
from typing import Optional


# потом подредактируем


@dataclass
class Song:
    file_path: str
    similarity: float
    start_time: float #для себя, чтобы понять что к чему
    end_time: float
    name: Optional[str] = None
    artist: Optional[str] = None
    year: Optional[int] = None
    album: Optional[str] = None
    link: Optional[str] = None
    file: Optional[str] = None
