from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Word2Vec',
  ext_modules = cythonize("train.py"),
)
