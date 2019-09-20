# Copyright 2016 Google Inc. All rights reserved.
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

import os
from datetime import datetime
from setuptools import setup


def version():
    version = os.getenv('VERSION', None)
    if version:
        # Most git tags are prefixed with 'v' (example: v1.2.3) this is
        # never desirable for artifact repositories, so we strip the
        # leading 'v' if it's present.
        return version[1:] if version.startswith('v') else version
    else:
        # Default version is an ISO8601 compiliant datetime. PyPI doesn't allow
        # the colon ':' character in its versions, and time is required to allow
        # for multiple publications to master in one day. This datetime string
        # uses the "basic" ISO8601 format for both its date and time components
        # to avoid issues with the colon character (ISO requires that date and
        # time components of a date-time string must be uniformly basic or
        # extended, which is why the date component does not have dashes.
        #
        # Publications using datetime versions should only be made from master
        # to represent the HEAD moving forward.
        version = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        print("VERSION environment variable not set, using datetime instead: {}"
              .format(version))

    return version

setup(
    name='flatbuffers',
    version=version(),
    license='Apache 2.0',
    author='FlatBuffers Contributors',
    author_email='me@rwinslow.com',
    url='https://google.github.io/flatbuffers/',
    long_description=('Python runtime library for use with the '
                      '`Flatbuffers <https://google.github.io/flatbuffers/>`_ '
                      'serialization format.'),
    packages=['flatbuffers'],
    include_package_data=True,
    requires=[],
    description='The FlatBuffers serialization format for Python',
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    project_urls={
        'Documentation': 'https://google.github.io/flatbuffers/',
        'Source': 'https://github.com/google/flatbuffers',
    },
)