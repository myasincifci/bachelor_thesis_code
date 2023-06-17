import gdown
import tarfile

url = 'https://drive.google.com/u/0/uc?id=1IsnAlys_Bu3mgSd8Iji7TcwJqkq8uotB&export=download'

output = './model_zoo/models.tar.xz'

if __name__ == "__main__":
    gdown.download(url, output, quiet=False)

    with tarfile.open('./model_zoo/models.tar.xz') as f:
        f.extractall('./model_zoo')