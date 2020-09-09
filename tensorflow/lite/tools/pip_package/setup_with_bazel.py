# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

import os

from setuptools import find_packages
from setuptools import setup
PACKAGE_NAME = 'tflite_runtime'
PACKAGE_VERSION = os.environ['PACKAGE_VERSION']
DOCLINES = __doc__.split('\n')

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
    package_dir={'': '.'},
    package_data={'': ['*.so']},
    install_requires=[
        'numpy >= 1.16.0',
        'pybind11 >= 2.4.3',
    ])
