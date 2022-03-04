from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='Train',
    ext_modules=cythonize("training.py"),
)
