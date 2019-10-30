# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TensorFlow Lite is for mobile and embedded devices.

TensorFlow Lite is the official solution for running machine learning models on
mobile and embedded devices. It enables on-device machine learning inference
with low latency and a small binary size on Android, iOS, and other operating
systems.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
import subprocess

from distutils.command.build_ext import build_ext
import numpy

from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
from setuptools.command.build_py import build_py
PACKAGE_NAME = 'tflite_runtime'
PACKAGE_VERSION = os.environ['PACKAGE_VERSION']
DOCLINES = __doc__.split('\n')
TENSORFLOW_DIR = os.environ['TENSORFLOW_DIR']

# Setup cross compiling
TARGET = os.environ.get('TENSORFLOW_TARGET', None)
if TARGET == 'rpi':
  os.environ['CXX'] = 'arm-linux-gnueabihf-g++'
  os.environ['CC'] = 'arm-linux-gnueabihf-gcc'
elif TARGET == 'aarch64':
  os.environ['CXX'] = 'aarch64-linux-gnu-g++'
  os.environ['CC'] = 'aarch64-linux-gnu-gcc'
MAKE_CROSS_OPTIONS = ['TARGET=%s' % TARGET]  if TARGET else []

RELATIVE_MAKE_DIR = os.path.join('tensorflow', 'lite', 'tools', 'make')
MAKE_DIR = os.path.join(TENSORFLOW_DIR, RELATIVE_MAKE_DIR)
DOWNLOADS_DIR = os.path.join(MAKE_DIR, 'downloads')
RELATIVE_MAKEFILE_PATH = os.path.join(RELATIVE_MAKE_DIR, 'Makefile')
DOWNLOAD_SCRIPT_PATH = os.path.join(MAKE_DIR, 'download_dependencies.sh')


# Check physical memory and if we are on a reasonable non small SOC machine
# with more than 4GB, use all the CPUs, otherwisxe only 1.
def get_build_cpus():
  physical_bytes = os.sysconf('SC_PAGESIZE') * os.sysconf('SC_PHYS_PAGES')
  if physical_bytes < (1<<30) * 4:
    return 1
  else:
    return multiprocessing.cpu_count()


def make_args(target='', quiet=True):
  """Construct make command line."""
  args = (['make', 'SHELL=/bin/bash',
           'BUILD_WITH_NNAPI=false', '-C', TENSORFLOW_DIR]
          + MAKE_CROSS_OPTIONS +
          ['-f', RELATIVE_MAKEFILE_PATH, '-j',
           str(get_build_cpus())])
  if quiet:
    args.append('--quiet')
  if target:
    args.append(target)
  return args


def make_output(target):
  """Invoke make on the target and return output."""
  return subprocess.check_output(make_args(target)).decode('utf-8').strip()


def make():
  """Invoke make to build tflite C++ sources.

  Build dependencies:
     apt-get install swig libjpeg-dev zlib1g-dev python3-dev python3-nump
  """
  subprocess.check_call(make_args(quiet=False))


def download_dependencies():
  """Download build dependencies if haven't done yet."""
  if not os.path.isdir(DOWNLOADS_DIR) or not os.listdir(DOWNLOADS_DIR):
    subprocess.check_call(DOWNLOAD_SCRIPT_PATH)


class CustomBuildExt(build_ext, object):
  """Customized build extension."""

  def get_ext_filename(self, ext_name):
    if TARGET:
      ext_path = ext_name.split('.')
      return os.path.join(*ext_path) + '.so'
    return super(CustomBuildExt, self).get_ext_filename(ext_name)

  def run(self):
    download_dependencies()
    make()

    return super(CustomBuildExt, self).run()


class CustomBuildPy(build_py, object):

  def run(self):
    self.run_command('build_ext')
    return super(CustomBuildPy, self).run()


LIB_TFLITE = 'tensorflow-lite'
LIB_TFLITE_DIR = make_output('libdir')

ext = Extension(
    name='%s._interpreter_wrapper' % PACKAGE_NAME,
    language='c++',
    sources=['interpreter_wrapper/interpreter_wrapper.i',
             'interpreter_wrapper/interpreter_wrapper.cc',
             'interpreter_wrapper/numpy.cc',
             'interpreter_wrapper/python_error_reporter.cc',
             'interpreter_wrapper/python_utils.cc'],
    swig_opts=['-c++',
               '-I%s' % TENSORFLOW_DIR,
               '-module', 'interpreter_wrapper',
               '-outdir', PACKAGE_NAME],
    extra_compile_args=['-std=c++11'],
    include_dirs=[TENSORFLOW_DIR,
                  os.path.join(TENSORFLOW_DIR, 'tensorflow', 'lite', 'tools',
                               'pip_package'),
                  numpy.get_include(),
                  os.path.join(DOWNLOADS_DIR, 'flatbuffers', 'include'),
                  os.path.join(DOWNLOADS_DIR, 'absl')],
    libraries=[LIB_TFLITE],
    library_dirs=[LIB_TFLITE_DIR])

setup(
    name=PACKAGE_NAME.replace('_', '-'),
    version=PACKAGE_VERSION,
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]),
    url='https://www.tensorflow.org/lite/',
    author='Google, LLC',
    author_email='packages@tensorflow.org',
    license='Apache 2.0',
    include_package_data=True,
    keywords='tflite tensorflow tensor machine learning',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(exclude=[]),
    ext_modules=[ext],
    install_requires=[
        'numpy >= 1.12.1',
    ],
    cmdclass={
        'build_ext': CustomBuildExt,
        'build_py': CustomBuildPy,
    }
)
