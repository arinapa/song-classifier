import s3fs
import os

class downloader:
    def __init__(this, name, url, access_key, secret_key):
        this.name = name
        this.file = s3fs.S3FileSystem(
            key=access_key,
            secret=secret_key,
            client_kwargs=url
        )

    def downlнoad_file(s3_bucket, s3_key, path):
        s3_path = f"{s3_bucket.name}/{s3_key}"
        with s3_bucket.file.open(s3_path, 'rb') as current_file:
            with open(path, 'wb') as local_file:
                local_file.write(current_file.read())
    #загрузили файл в local_path

    
