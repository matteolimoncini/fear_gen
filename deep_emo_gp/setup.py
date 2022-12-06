from setuptools import setup

setup(name='deepEmoGP',
      version='0.1',
      description='Deep construction of an affective latent space via multimodal enactment',
      url='http://github.com/phuselab/deepEmoGP',
      author='Vittorio Cuculo',
      author_email='vittorio.cuculo@unimi.it',
      license='MIT',
      packages=['deepemogp'],
      install_requires=[
          'cvxopt>=1.1.9', 'PyWavelets>=0.5.2', 'biosppy>=0.5.1',
      ],
      dependency_links=[
          'git+https://github.com/SheffieldML/pydeepgp.git@master#egg=dgp-2.0'],
      zip_safe=False)
