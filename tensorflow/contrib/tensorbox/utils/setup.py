from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

sourcefiles = ['stitch_wrapper.pyx', './hungarian/hungarian.cpp', 'stitch_rects.cpp']

extensions = [Extension("stitch_wrapper", sourcefiles, language="c++")]

setup(
    ext_modules = cythonize(extensions)
)
