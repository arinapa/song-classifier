import pandas
import psycopg2
import numpy
conn = psycopg2.connect(
   database="postgres",
    user='postgres',
    password='15042001',
    host='localhost',
    port= '5432'
)
conn.autocommit = True
cur = conn.cursor()
cur.execute("CREATE DATABASE DB_SONGS")
cur.close()
conn.close() 
conn = psycopg2.connect( #подключение к созданное бд
        dbname="DB_SONGS",
        user="DB_USER",
        password="DB_PASSWORD",
        host="DB_HOST",
        port="DB_PORT" 
    )
    
cur = conn.cursor()
cur.execute("""
    CREATE TABLE IF NOT EXISTS songs (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    artist VARCHAR(255) NOT NULL,
    language VARCHAR(255),
    genre VARCHAR(255),
    spectrogram_vector REAL[]  );
    """)
conn.commit()
cur.close()

def get_vector(song): #создание векторов ярких точек 
    #здесь наверное будет функция которая сейчас в отдельном файле или не будет 
    return 1


def add_new_song(conn, title, artist, language, genre, spectrogram_vector):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO songs (title, artist, language, genre, spectrogram_vector)
        VALUES (%s, %s, %s, %s, %s);
        """, (title, artist, language, genre, spectrogram_vector))
    conn.commit()
    cur.close()

csv_file = "./dataset_songs.csv"  #таблица с информацией о песнях
df = pandas.read_csv(csv_file)

for index, row in df.iterrows():
    title = row['Название']
    artist = row['Исполнитель']
    language = row['Язык']
    genre = row['Жанр']
    current_vector = get_vector(title) 
    add_new_song(conn, title, artist, language, genre, current_vector)
conn.close()
