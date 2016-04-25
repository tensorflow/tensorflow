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
import platform
import re
import sys

from setuptools import find_packages, setup, Command, Extension
from setuptools.command.install import install as InstallCommandBase
from setuptools.dist import Distribution

_VERSION = '0.8.0'

numpy_version = "1.8.2"
if platform.system() == "Darwin":
  # There are bugs with numpy pip installation on OS X prior to
  # 1.10.1, so on mac we require a higher version than on other
  # platforms.
  numpy_version = "1.10.1"

REQUIRED_PACKAGES = [
    'numpy >= %s' % numpy_version,
    'six >= 1.10.0',
    'protobuf == 3.0.0b2',
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


class InstallCommand(InstallCommandBase):
  """Override the dir where the headers go."""

  def finalize_options(self):
    ret = InstallCommandBase.finalize_options(self)
    self.install_headers = os.path.join(self.install_purelib,
                                        'tensorflow', 'include')
    return ret


class InstallHeaders(Command):
  """Override how headers are copied.

  The install_headers that comes with setuptools copies all files to
  the same directory. But we need the files to be in a specific directory
  hierarchy for -I <include_dir> to work correctly.
  """
  description = 'install C/C++ header files'

  user_options = [('install-dir=', 'd',
                   'directory to install header files to'),
                  ('force', 'f',
                   'force installation (overwrite existing files)'),
                 ]

  boolean_options = ['force']

  def initialize_options(self):
    self.install_dir = None
    self.force = 0
    self.outfiles = []

  def finalize_options(self):
    self.set_undefined_options('install',
                               ('install_headers', 'install_dir'),
                               ('force', 'force'))

  def mkdir_and_copy_file(self, header):
    install_dir = os.path.join(self.install_dir, os.path.dirname(header))
    # Get rid of some extra intervening directories so we can have fewer
    # directories for -I
    install_dir = re.sub('/google/protobuf/src', '', install_dir)

    # Copy eigen code into tensorflow/include,
    # tensorflow/include/external/eigen_archive/eigen-eigen-<revision>,
    # and tensorflow/include/eigen-eigen-<revision>.
    # A symlink would do, but the wheel file that gets created ignores
    # symlink within the directory hierarchy.
    # NOTE(keveman): Figure out how to customize bdist_wheel package so
    # we can do the symlink.
    if re.search(r'(external/eigen_archive/eigen-eigen-\w+)', install_dir):
      extra_dirs = [re.sub('/external/eigen_archive', '', install_dir),
                    re.sub(r'external/eigen_archive/eigen-eigen-\w+', '',
                           install_dir)]
      for extra_dir in extra_dirs:
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
  for path, _, files in os.walk(root):
    for filename in fnmatch.filter(files, pattern):
      yield os.path.join(path, filename)


matches = ['../' + x for x in find_files('*', 'external') if '.py' not in x]


headers = (list(find_files('*.h', 'tensorflow/core')) +
           list(find_files('*.h', 'google/protobuf/src')) +
           list(find_files('*', 'third_party/eigen3')) +
           list(find_files('*', 'external/eigen_archive')))


setup(
    name='tensorflow',
    version=_VERSION,
    description='TensorFlow helps the tensors flow',
    long_description='',
    url='http://tensorflow.org/',
    author='Google Inc.',
    author_email='opensource@google.com',
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
        'tensorflow': ['python/_pywrap_tensorflow.so',
                       'tensorboard/dist/index.html',
                       'tensorboard/dist/tf-tensorboard.html',
                       'tensorboard/lib/css/global.css',
                     ] + matches,
    },
    zip_safe=False,
    distclass=BinaryDistribution,
    cmdclass={
        'install_headers': InstallHeaders,
        'install': InstallCommand,
    },
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
