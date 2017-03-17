"""Pip packaging for Poplar Tensorflow plugin
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
  long_description = f.read()

setup(
  name='poplar_plugin',

  version='1.0.0',

  description='Poplar Tensorflow plugin',
  long_description='',
  url='https://graphcore.ai',
  author='Graphcore Ltd',
  author_email='info@graphcore.ai',
  license='Apache 2.0',

  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Software Development :: Libraries',
    'License :: OSI Approved :: Apache Software License',

    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
  ],

  keywords='tensorflow poplar ipu',

  packages=["poplar_plugin"],

  install_requires=['tensorflow'],

  extras_require={},

  package_data={
    'poplar_plugin':
      ['libpoplar_plugin.so',
       'tf.gp'],
  },

  data_files=[],

  entry_points={},
)
