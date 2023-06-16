import gdown
import tarfile

url = 'https://drive.google.com/u/0/uc?id=1n1aQriLVscRcJ6LAcVHMqv7jLtv2YWGP&export=download'
output = './datasets/data.tar.xz'

if __name__ == "__main__":
    gdown.download(url, output, quiet=False)

    with tarfile.open('test.tar.xz') as f:
        f.extractall('./datasets')