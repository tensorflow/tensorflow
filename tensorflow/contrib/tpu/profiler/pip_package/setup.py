# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Cloud TPU profiler package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup

_VERSION = '1.3.0-a1'

CONSOLE_SCRIPTS = [
    'capture_tpu_profile=cloud_tpu_profiler.main:main',
]

REQUIRED_PACKAGES = [
    'tensorflow >= 1.2.0',
]

setup(
    name='cloud_tpu_profiler',
    version=_VERSION.replace('-', ''),
    description='Trace and profile Cloud TPU performance',
    long_description='Tools for capture TPU profile',
    url='https://www.tensorflow.org/tfrc/',
    author='Google Inc.',
    author_email='opensource@google.com',
    packages=['cloud_tpu_profiler'],
    package_data={
        'cloud_tpu_profiler': ['data/*'],
    },
    entry_points={
        'console_scripts': CONSOLE_SCRIPTS,
    },
    install_requires=REQUIRED_PACKAGES,
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        
        'License :: OSI Approved :: Apache Software License',
        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',  
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache 2.0',
    keywords='tensorflow performance tpu',)
