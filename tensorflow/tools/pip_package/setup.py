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
# limitations under the License.
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
across many other scientific domains. TensorFlow is licensed under [Apache
2.0](https://github.com/tensorflow/tensorflow/blob/master/LICENSE).
"""

import datetime
import fnmatch
import os
import platform
import re
import sys

from setuptools import Command
from setuptools import find_namespace_packages
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase
from setuptools.dist import Distribution

# This version string is semver compatible, but incompatible with pip.
# For pip, we will remove all '-' characters from this string, and use the
# result for pip.
# Also update tensorflow/tensorflow.bzl and
# tensorflow/core/public/version.h
_VERSION = '2.19.0'

# We use the same setup.py for all tensorflow_* packages and for the nightly
# equivalents (tf_nightly_*). The package is controlled from the argument line
# when building the pip package.
project_name = 'tensorflow'
if os.environ.get('project_name', None):
  project_name = os.environ['project_name']

collaborator_build = os.environ.get('collaborator_build', False)

# Returns standard if a tensorflow-* package is being built, and nightly if a
# tf_nightly-* package is being built.
def standard_or_nightly(standard, nightly):
  return nightly if 'tf_nightly' in project_name else standard

# All versions of TF need these packages. We indicate the widest possible range
# of package releases possible to be as up-to-date as possible as well as to
# accommodate as many pre-installed packages as possible.
REQUIRED_PACKAGES = [
    'absl-py >= 1.0.0',
    'astunparse >= 1.6.0',
    'flatbuffers >= 24.3.25',
    'gast >=0.2.1,!=0.5.0,!=0.5.1,!=0.5.2',
    'google_pasta >= 0.1.1',
    'libclang >= 13.0.0',
    'opt_einsum >= 2.3.2',
    'packaging',
    (
        'protobuf>=3.20.3,<6.0.0dev,!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5'
    ),
    'requests >= 2.21.0, < 3',
    'setuptools',
    'six >= 1.12.0',
    'termcolor >= 1.1.0',
    'typing_extensions >= 3.6.6',
    'wrapt >= 1.11.0',
    'tensorflow-io-gcs-filesystem >= 0.23.1 ; python_version < "3.12"',
    'grpcio >= 1.24.3, < 2.0' if sys.byteorder == 'little' else None,
    'tb-nightly ~= 2.19.0.a',
    'keras-nightly >= 3.6.0.dev',
    'numpy >= 1.26.
