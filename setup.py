#!/usr/bin/env python
#  Copyright 2015-present Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from setuptools import setup, find_packages

META_DATA = dict(
    name='skflow',
    version='0.1.0',
    url='https://github.com/tensorflow/skflow',
    license='Apache-2',
    description='Simplified Interface for TensorFlow for Deep Learning',
    author=['Scikit Flow Authors'],
    author_email='terrytangyuan@Gmail.com',
    packages=find_packages(),
    install_requires=[
        'sklearn',
        'scipy',
        'numpy',
    ],
    classifiers=[
        'Programming Language :: Python',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering'
    ],
    keywords=[
        'Deep Learning',
        'Neural Networks',
        'Google',
        'TensorFlow',
        'Machine Learning'
    ]
)


if __name__ == '__main__':
    setup(**META_DATA)

