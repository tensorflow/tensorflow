# lint as: python3
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# limitations under the License..
# ==============================================================================
"""TensorFlow is an open source machine learning framework for everyone.

[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/tensorflow)
[![PyPI](https://badge.fury.io/py/tensorflow.svg)](https://badge.fury.io/py/tensorflow)

TensorFlow is an open source software library for high performance numerical
computation. Its flexible architecture allows easy deployment of computation
across a variety of platforms (CPUs, GPUs, TPUs), and from desktops to clusters
of servers to mobile and edge devices.

Originally developed by researchers and engineers from the Google Brain team
within Google's AI organization, it comes with strong support for machine
learning and deep learning and the flexible numerical computation core is used
across many other scientific domains.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fnmatch
import os
import re
import sys

from setuptools import Command
from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase
from setuptools.dist import Distribution

# This version string is semver compatible, but incompatible with pip.
# For pip, we will remove all '-' characters from this string, and use the
# result for pip.
# Also update tensorflow/tensorflow.bzl and
# tensorflow/core/public/version.h
_VERSION = '2.3.0'

REQUIRED_PACKAGES = [
    'absl-py >= 0.7.0',
    'astunparse == 1.6.3',
    'gast == 0.3.3',
    'google_pasta >= 0.1.8',
    'h5py >= 2.10.0, < 2.11.0',
    'keras_preprocessing >= 1.1.1, < 1.2',
    # TODO(mihaimaruseac): numpy 1.19.0 has ABI breakage
    # https://github.com/numpy/numpy/pull/15355
    'numpy >= 1.16.0, < 1.19.0',
    'opt_einsum >= 2.3.2',
    'protobuf >= 3.9.2',
    'tensorboard >= 2.3.0, < 3',
    'tensorflow_estimator >= 2.3.0, < 2.4.0',
    'termcolor >= 1.1.0',
    'wrapt >= 1.11.1',
    'wheel >= 0.26',
    'six >= 1.12.0',
    # scipy < 1.4.1 causes segfaults due to pybind11
    'scipy == 1.4.1',
]

if sys.byteorder == 'little':
  # grpcio does not build correctly on big-endian machines due to lack of
  # BoringSSL support.
  # See https://github.com/tensorflow/tensorflow/issues/17882.
  REQUIRED_PACKAGES.append('grpcio >= 1.8.6')

project_name = 'tensorflow'
if '--project_name' in sys.argv:
  project_name_idx = sys.argv.index('--project_name')
  project_name = sys.argv[project_name_idx + 1]
  sys.argv.remove('--project_name')
  sys.argv.pop(project_name_idx)

# tf-nightly should depend on tb-nightly
if 'tf_nightly' in project_name:
  for i, pkg in enumerate(REQUIRED_PACKAGES):
    if 'tensorboard' in pkg:
      REQUIRED_PACKAGES[i] = 'tb-nightly >= 2.3.0a0, < 2.4.0a0'
    elif 'tensorflow_estimator' in pkg:
      REQUIRED_PACKAGES[i] = 'tf-estimator-nightly'

DOCLINES = __doc__.split('\n')
if project_name.endswith('-gpu'):
  project_name_no_gpu = project_name[:-len('-gpu')]
  _GPU_PACKAGE_NOTE = 'Note that %s package by default supports both CPU and '\
      'GPU. %s has the same content and exists solely for backward '\
      'compatiblity. Please migrate to %s for GPU support.'\
      % (project_name_no_gpu, project_name, project_name_no_gpu)
  DOCLINES.append(_GPU_PACKAGE_NOTE)


# pylint: disable=line-too-long
CONSOLE_SCRIPTS = [
    'toco_from_protos = tensorflow.lite.toco.python.toco_from_protos:main',
    'tflite_convert = tensorflow.lite.python.tflite_convert:main',
    'toco = tensorflow.lite.python.tflite_convert:main',
    'saved_model_cli = tensorflow.python.tools.saved_model_cli:main',
    # We need to keep the TensorBoard command, even though the console script
    # is now declared by the tensorboard pip package. If we remove the
    # TensorBoard command, pip will inappropriately remove it during install,
    # even though the command is not removed, just moved to a different wheel.
    'tensorboard = tensorboard.main:run_main',
    'tf_upgrade_v2 = tensorflow.tools.compatibility.tf_upgrade_v2_main:main',
    'estimator_ckpt_converter = '
    'tensorflow_estimator.python.estimator.tools.checkpoint_converter:main',
]
# pylint: enable=line-too-long

# remove the tensorboard console script if building tf_nightly
if 'tf_nightly' in project_name:
  CONSOLE_SCRIPTS.remove('tensorboard = tensorboard.main:run_main')

TEST_PACKAGES = [
    'scipy >= 0.15.1',
]


class BinaryDistribution(Distribution):

  def has_ext_modules(self):
    return True


class InstallCommand(InstallCommandBase):
  """Override the dir where the headers go."""

  def finalize_options(self):
    ret = InstallCommandBase.finalize_options(self)
    self.install_headers = os.path.join(self.install_platlib, 'tensorflow',
                                        'include')
    self.install_lib = self.install_platlib
    return ret


class InstallHeaders(Command):
  """Override how headers are copied.

  The install_headers that comes with setuptools copies all files to
  the same directory. But we need the files to be in a specific directory
  hierarchy for -I <include_dir> to work correctly.
  """
  description = 'install C/C++ header files'

  user_options = [
      ('install-dir=', 'd', 'directory to install header files to'),
      ('force', 'f', 'force installation (overwrite existing files)'),
  ]

  boolean_options = ['force']

  def initialize_options(self):
    self.install_dir = None
    self.force = 0
    self.outfiles = []

  def finalize_options(self):
    self.set_undefined_options('install', ('install_headers', 'install_dir'),
                               ('force', 'force'))

  def mkdir_and_copy_file(self, header):
    install_dir = os.path.join(self.install_dir, os.path.dirname(header))
    # Get rid of some extra intervening directories so we can have fewer
    # directories for -I
    install_dir = re.sub('/google/protobuf_archive/src', '', install_dir)

    # Copy external code headers into tensorflow/include.
    # A symlink would do, but the wheel file that gets created ignores
    # symlink within the directory hierarchy.
    # NOTE(keveman): Figure out how to customize bdist_wheel package so
    # we can do the symlink.
    external_header_locations = [
        'tensorflow/include/external/eigen_archive/',
        'tensorflow/include/external/com_google_absl/',
    ]
    for location in external_header_locations:
      if location in install_dir:
        extra_dir = install_dir.replace(location, '')
        if not os.path.exists(extra_dir):
          self.mkpath(extra_dir)
        self.copy_file(header, extra_dir)

    if not os.path.exists(install_dir):
      self.mkpath(install_dir)
    return self.copy_file(header, install_dir)

  def run(self):
    hdrs = self.distribution.headers
    if not hdrs:
      return

    self.mkpath(self.install_dir)
    for header in hdrs:
      (out, _) = self.mkdir_and_copy_file(header)
      self.outfiles.append(out)

  def get_inputs(self):
    return self.distribution.headers or []

  def get_outputs(self):
    return self.outfiles


def find_files(pattern, root):
  """Return all the files matching pattern below root dir."""
  for dirpath, _, files in os.walk(root):
    for filename in fnmatch.filter(files, pattern):
      yield os.path.join(dirpath, filename)


so_lib_paths = [
    i for i in os.listdir('.')
    if os.path.isdir(i) and fnmatch.fnmatch(i, '_solib_*')
]

matches = []
for path in so_lib_paths:
  matches.extend(['../' + x for x in find_files('*', path) if '.py' not in x])

if os.name == 'nt':
  EXTENSION_NAME = 'python/_pywrap_tensorflow_internal.pyd'
else:
  EXTENSION_NAME = 'python/_pywrap_tensorflow_internal.so'

headers = (
    list(find_files('*.proto', 'tensorflow/compiler')) +
    list(find_files('*.proto', 'tensorflow/core')) +
    list(find_files('*.proto', 'tensorflow/python')) +
    list(find_files('*.def', 'tensorflow/compiler')) +
    list(find_files('*.h', 'tensorflow/c')) +
    list(find_files('*.h', 'tensorflow/cc')) +
    list(find_files('*.h', 'tensorflow/compiler')) +
    list(find_files('*.h.inc', 'tensorflow/compiler')) +
    list(find_files('*.h', 'tensorflow/core')) +
    list(find_files('*.h', 'tensorflow/python')) +
    list(find_files('*.h', 'tensorflow/stream_executor')) +
    list(find_files('*.h', 'google/com_google_protobuf/src')) +
    list(find_files('*.inc', 'google/com_google_protobuf/src')) +
    list(find_files('*', 'third_party/eigen3')) +
    list(find_files('*.h', 'tensorflow/include/external/com_google_absl')) +
    list(find_files('*.inc', 'tensorflow/include/external/com_google_absl')) +
    list(find_files('*', 'tensorflow/include/external/eigen_archive')))

setup(
    name=project_name,
    version=_VERSION.replace('-', ''),
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]),
    url='https://www.tensorflow.org/',
    download_url='https://github.com/tensorflow/tensorflow/tags',
    author='Google Inc.',
    author_email='packages@tensorflow.org',
    # Contained modules and scripts.
    packages=find_packages(),
    entry_points={
        'console_scripts': CONSOLE_SCRIPTS,
    },
    headers=headers,
    install_requires=REQUIRED_PACKAGES,
    tests_require=REQUIRED_PACKAGES + TEST_PACKAGES,
    # Add in any packaged data.
    include_package_data=True,
    package_data={
        'tensorflow': [
            EXTENSION_NAME,
        ] + matches,
    },
    zip_safe=False,
    distclass=BinaryDistribution,
    cmdclass={
        'install_headers': InstallHeaders,
        'install': InstallCommand,
    },
    # PyPI package information.
    classifiers=sorted([
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]),
    license='Apache 2.0',
    keywords='tensorflow tensor machine learning',
)
