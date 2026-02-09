# coding=utf-8
# Copyright 2025 TF.Text Authors.
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

# Copyright 2024 TF.Text Authors.
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

"""TF.Text is a TensorFlow library of text related ops, modules, and subgraphs.

TF.Text is a TensorFlow library of text related ops, modules, and subgraphs. The
library can perform the preprocessing regularly required by text-based models,
and includes other features useful for sequence modeling not provided by core
TensorFlow.

See the README on GitHub for further documentation.
http://github.com/tensorflow/text
"""

import os

from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install
from setuptools.dist import Distribution

project_name = 'tensorflow-text'
project_version = '2.20.0'


class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def is_pure(self):
    return False

  def has_ext_modules(self):
    return True


class InstallPlatlib(install):
  """This is needed to set the library to platlib compliant."""

  def finalize_options(self):
    """For more info; see http://github.com/google/or-tools/issues/616 ."""
    install.finalize_options(self)
    self.install_lib = self.install_platlib
    self.install_libbase = self.install_lib
    self.install_lib = os.path.join(self.install_lib, self.extra_dirs)


DOCLINES = __doc__.split('\n')

setup(
    name=project_name,
    version=project_version.replace('-', ''),
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]),
    author='Google Inc.',
    author_email='packages@tensorflow.org',
    url='http://github.com/tensorflow/text',
    license='Apache 2.0',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    cmdclass={'install': InstallPlatlib},
    distclass=BinaryDistribution,
    install_requires=[
        ('tensorflow>=2.20.0, <2.21',),
    ],
    extras_require={
        'tensorflow_cpu': [
            'tensorflow-cpu>=2.20.0, <2.21',
        ],
        'tests': [
            'absl-py',
            'pytest',
            'tensorflow-datasets>=3.2.0',
        ],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='tensorflow text machine learning',
)
