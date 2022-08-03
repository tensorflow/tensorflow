# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
# We use this to build installer wheels whose only job would be to install the
# third-party TensorFlow packages from Google's official partners.
# Note: This is experimental for now and is used internally for testing.
import sys

from setuptools import setup


# This version string is semver compatible, but incompatible with pip.
# For pip, we will remove all '-' characters from this string, and use the
# result for pip.
# Also update tensorflow/tensorflow.bzl and
# tensorflow/core/public/version.h
_VERSION = '2.11.0'


# We use the same setup.py for all tensorflow_* packages and for the nightly
# equivalents (tf_nightly_*). The package is controlled from the argument line
# when building the pip package.
project_name = 'tensorflow'
if '--project_name' in sys.argv:
  project_name_idx = sys.argv.index('--project_name')
  project_name = sys.argv[project_name_idx + 1]
  sys.argv.remove('--project_name')
  sys.argv.pop(project_name_idx)


# Returns standard if a tensorflow-* package is being built, and nightly if a
# tf_nightly-* package is being built.
def standard_or_nightly(standard, nightly):
  return nightly if 'tf_nightly' in project_name else standard

# Install the trusted partner packages with the same version as the installer
# wheel
REQUIRED_PACKAGES = [
    # Install the TensorFlow package built by AWS if the user is running
    # Linux on an Aarch64 machine.
    standard_or_nightly('tensorflow-cpu-aws', 'tf-nightly-cpu-aws ') + '==' +
    _VERSION +
    ';platform_system=="Linux" and (platform_machine=="arm64" or '
    'platform_machine=="aarch64")',
    # Install the TensorFlow package built by Intel if the user is on a Windows
    # machine.
    standard_or_nightly('tensorflow-intel', 'tf-nightly-intel') +
    '==' + _VERSION + ';platform_system=="Windows"',
]
REQUIRED_PACKAGES = [p for p in REQUIRED_PACKAGES if p is not None]

DOCLINES = __doc__.split('\n')
if project_name.endswith('-gpu'):
  project_name_no_gpu = project_name[:-len('-gpu')]
  _GPU_PACKAGE_NOTE = 'Note that %s package by default supports both CPU and '\
      'GPU. %s has the same content and exists solely for backward '\
      'compatibility. Please migrate to %s for GPU support.'\
      % (project_name_no_gpu, project_name, project_name_no_gpu)
  DOCLINES.append(_GPU_PACKAGE_NOTE)

setup(
    name=project_name,
    version=_VERSION.replace('-', ''),
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]),
    long_description_content_type='text/markdown',
    url='https://www.tensorflow.org/',
    download_url='https://github.com/tensorflow/tensorflow/tags',
    author='Google Inc.',
    author_email='packages@tensorflow.org',
    install_requires=REQUIRED_PACKAGES,
    zip_safe=False,
    # Supported Python versions
    python_requires='>=3.7',
    # PyPI package information.
    classifiers=sorted([
        'Development Status :: 5 - Production/Stable',
        # TODO(angerson) Add IFTTT when possible
        'Environment :: GPU :: NVIDIA CUDA :: 11.2',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
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

