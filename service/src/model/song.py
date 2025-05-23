from dataclasses import dataclass
 
@dataclass
class Song(path):
    name: str
    artist: str
    year: int | None
    album: str | None
    link: str | None
    file: str | None
    