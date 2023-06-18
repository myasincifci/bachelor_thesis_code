from distutils.core import setup

setup(name='Tempo2',
      version='0.1',
      description='Temporal Coherence',
      author='Mehmet Yasin Cifci',
      author_email='cifci@campus.tu-berlin.de',
      packages=['tempo2', 'tempo2.data'],
      install_requires=[
          "numpy==1.21.6",
          "torch==1.13.0",
          "torchvision==0.14.0",
          "tqdm==4.64.1",
          "lightly==1.2.35",
          "lightly-utils==0.0.2",
          "matplotlib==3.5.2",
          "Pillow==8.1.1",
          "tensorboard==2.10.1",
          "zennit==0.5.0",
          "pytorch-lightning==1.8.1",
          "scipy==1.9.1",
          "gdown==4.6.4",
      ]
     )