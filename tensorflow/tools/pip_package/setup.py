# Copyright 2015 Google Inc. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fnmatch
import os
import sys

from setuptools import find_packages, setup, Extension
from setuptools.dist import Distribution

_VERSION = '0.6.0'

REQUIRED_PACKAGES = [
    'numpy >= 1.8.2',
    'six >= 1.10.0',
    'protobuf == 3.0.0a3',
]

# python3 requires wheel 0.26
if sys.version_info.major == 3:
  REQUIRED_PACKAGES.append('wheel >= 0.26')
else:
  REQUIRED_PACKAGES.append('wheel')


# pylint: disable=line-too-long
CONSOLE_SCRIPTS = [
    'tensorboard = tensorflow.tensorboard.tensorboard:main',
]
# pylint: enable=line-too-long

TEST_PACKAGES = [
    'scipy >= 0.15.1',
]

class BinaryDistribution(Distribution):
  def is_pure(self):
    return False

matches = []
for root, dirnames, filenames in os.walk('external'):
  for filename in fnmatch.filter(filenames, '*'):
    matches.append(os.path.join(root, filename))

matches = ['../' + x for x in matches if '.py' not in x]

setup(
    name='tensorflow',
    version=_VERSION,
    description='TensorFlow helps the tensors flow',
    long_description='',
    url='http://tensorflow.com/',
    author='Google Inc.',
    author_email='opensource@google.com',
    # Contained modules and scripts.
    packages=find_packages(),
    entry_points={
        'console_scripts': CONSOLE_SCRIPTS,
    },
    install_requires=REQUIRED_PACKAGES,
    tests_require=REQUIRED_PACKAGES + TEST_PACKAGES,
    # Add in any packaged data.
    include_package_data=True,
    package_data={
        'tensorflow': ['python/_pywrap_tensorflow.so',
                       'tensorboard/dist/index.html',
                       'tensorboard/dist/tf-tensorboard.html',
                       'tensorboard/lib/css/global.css',
                     ] + matches,
    },
    zip_safe=False,
    distclass=BinaryDistribution,
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
        ],
    license='Apache 2.0',
    keywords='tensorflow tensor machine learning',
    )
