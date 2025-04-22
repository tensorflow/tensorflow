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


# A genrule //tensorflow/tools/pip_package:setup_py replaces dummy string by
# by the data provided in //tensorflow/tf_version.bzl.
# The version suffix can be set by passing the build parameters
# --repo_env=ML_WHEEL_BUILD_DATE=<date> and
# --repo_env=ML_WHEEL_VERSION_SUFFIX=<suffix>.
# To update the project version, update tf_version.bzl.
# This version string is semver compatible, but incompatible with pip.
# For pip, we will remove all '-' characters from this string, and use the
# result for pip.
_VERSION = '0.0.0'

# Update this version when a new libtpu stable version is released.
LATEST_RELEASE_LIBTPU_VERSION = '0.0.13'
NEXT_LIBTPU_VERSION = '0.0.14'

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
# accomodate as many pre-installed packages as possible.
# For packages that don't have yet a stable release, we pin using `~= 0.x` which
# means we accept any `0.y` version (y >= x) but not the first major release. We
# will need additional testing for that.
# NOTE: This assumes that all packages follow SemVer. If a package follows a
# different versioning scheme (e.g., PVP), we use different bound specifier and
# comment the versioning scheme.
REQUIRED_PACKAGES = [
    'absl-py >= 1.0.0',
    'astunparse >= 1.6.0',
    'flatbuffers >= 24.3.25',
    'gast >=0.2.1,!=0.5.0,!=0.5.1,!=0.5.2',
    'google_pasta >= 0.1.1',
    'libclang >= 13.0.0',
    'opt_einsum >= 2.3.2',
    'packaging',
    'protobuf>=4.21.6,<7.0.0dev',
    'requests >= 2.21.0, < 3',
    'setuptools',
    'six >= 1.12.0',
    'termcolor >= 1.1.0',
    'typing_extensions >= 3.6.6',
    'wrapt >= 1.11.0',
    # grpcio does not build correctly on big-endian machines due to lack of
    # BoringSSL support.
    # See https://github.com/tensorflow/tensorflow/issues/17882.
    'grpcio >= 1.24.3, < 2.0' if sys.byteorder == 'little' else None,
    # TensorFlow exposes the TF API for certain TF ecosystem packages like
    # keras. When TF depends on those packages, the package version needs to
    # match the current TF version. For tf_nightly, we install the nightly
    # variant of each package instead, which must be one version ahead of the
    # current release version. These also usually have "alpha" or "dev" in their
    # version name. During the TF release process the version of these
    # dependencies on the release branch is updated to the stable releases (RC
    # or final). For example, 'keras-nightly ~= 2.14.0.dev' will be replaced by
    # 'keras >= 2.14.0rc0, < 2.15' on the release branch after the branch cut.
    'tb-nightly ~= 2.19.0.a',
    'keras-nightly >= 3.6.0.dev',
    'numpy >= 1.26.0, < 2.2.0',
    'h5py >= 3.11.0',
    'ml_dtypes >= 0.5.1, < 1.0.0',
]

REQUIRED_PACKAGES = [p for p in REQUIRED_PACKAGES if p is not None]

FAKE_REQUIRED_PACKAGES = [
    # The depedencies here below are not actually used but are needed for
    # package managers like poetry to parse as they are confused by the
    # different architectures having different requirements.
    # The entries here should be a simple duplicate of those in the collaborator
    # build section.
    standard_or_nightly('tensorflow-intel', 'tf-nightly-intel') + '==' +
    _VERSION + ';platform_system=="Windows"',
]

if platform.system() == 'Linux' and platform.machine() == 'x86_64':
  REQUIRED_PACKAGES.append(FAKE_REQUIRED_PACKAGES)

if collaborator_build:
  # If this is a collaborator build, then build an "installer" wheel and
  # add the collaborator packages as the only dependencies.
  REQUIRED_PACKAGES = [
      # Install the TensorFlow package built by Intel if the user is on a
      # Windows machine.
      standard_or_nightly('tensorflow-intel', 'tf-nightly-intel')
      + '=='
      + _VERSION
      + ';platform_system=="Windows"',
  ]

# Set up extra packages, which are optional sets of other Python package deps.
# E.g. "pip install tensorflow[and-cuda]" below installs the normal TF deps,
# plus the CUDA libraries listed.
EXTRA_PACKAGES = {
    'and-cuda': [
        # TODO(nluehr): set nvidia-* versions based on build components.
        'nvidia-cublas-cu12 == 12.5.3.2',
        'nvidia-cuda-cupti-cu12 == 12.5.82',
        'nvidia-cuda-nvcc-cu12 == 12.5.82',
        'nvidia-cuda-nvrtc-cu12 == 12.5.82',
        'nvidia-cuda-runtime-cu12 == 12.5.82',
        'nvidia-cudnn-cu12 == 9.3.0.75',
        'nvidia-cufft-cu12 == 11.2.3.61',
        'nvidia-curand-cu12 == 10.3.6.82',
        'nvidia-cusolver-cu12 == 11.6.3.83',
        'nvidia-cusparse-cu12 == 12.5.1.3',
        'nvidia-nccl-cu12 == 2.25.1',
        'nvidia-nvjitlink-cu12 == 12.5.82',
    ],
    'gcs-filesystem': [
        ('tensorflow-io-gcs-filesystem>=0.23.1; '
         'sys_platform!="win32" and python_version<"3.13"'),
        ('tensorflow-io-gcs-filesystem>=0.23.1; '
         'sys_platform=="win32" and python_version<"3.12"'),
    ]
}

DOCLINES = __doc__.split('\n')

# pylint: disable=line-too-long
CONSOLE_SCRIPTS = [
    'tflite_convert = tensorflow.lite.python.tflite_convert:main',
    'toco = tensorflow.lite.python.tflite_convert:main',
    'saved_model_cli = tensorflow.python.tools.saved_model_cli:main',
    (
        'import_pb_to_tensorboard ='
        ' tensorflow.python.tools.import_pb_to_tensorboard:main'
    ),
    # We need to keep the TensorBoard command, even though the console script
    # is now declared by the tensorboard pip package. If we remove the
    # TensorBoard command, pip will inappropriately remove it during install,
    # even though the command is not removed, just moved to a different wheel.
    # We exclude it anyway if building tf_nightly.
    standard_or_nightly('tensorboard = tensorboard.main:run_main', None),
    'tf_upgrade_v2 = tensorflow.tools.compatibility.tf_upgrade_v2_main:main',
]
CONSOLE_SCRIPTS = [s for s in CONSOLE_SCRIPTS if s is not None]
# pylint: enable=line-too-long


class BinaryDistribution(Distribution):

  def has_ext_modules(self):
    return True


class InstallCommand(InstallCommandBase):
  """Override the dir where the headers go."""

  def finalize_options(self):
    ret = InstallCommandBase.finalize_options(self)  # pylint: disable=assignment-from-no-return
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
    # Windows platform uses "\" in path strings, the external header location
    # expects "/" in paths. Hence, we replaced "\" with "/" for this reason
    install_dir = install_dir.replace('\\', '/')
    # Get rid of some extra intervening directories so we can have fewer
    # directories for -I
    install_dir = re.sub('/google/protobuf_archive/src', '', install_dir)

    # Copy external code headers into tensorflow/include.
    # A symlink would do, but the wheel file that gets created ignores
    # symlink within the directory hierarchy.
    # NOTE(keveman): Figure out how to customize bdist_wheel package so
    # we can do the symlink.
    # pylint: disable=line-too-long
    external_header_locations = {
        '/tensorflow/include/external/com_google_absl': '',
        '/tensorflow/include/external/ducc': '/ducc',
        '/tensorflow/include/external/eigen_archive': '',
        '/tensorflow/include/external/ml_dtypes_py': '',
        '/tensorflow/include/tensorflow/compiler/xla': (
            '/tensorflow/include/xla'
        ),
        '/tensorflow/include/tensorflow/tsl': '/tensorflow/include/tsl',
    }
    # pylint: enable=line-too-long

    for location in external_header_locations:
      if location in install_dir:
        extra_dir = install_dir.replace(location,
                                        external_header_locations[location])
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

# If building a tpu package, LibTPU for Cloud TPU VM can be installed via:
# $ pip install <tf-tpu project> -f \
#  https://storage.googleapis.com/libtpu-releases/index.html
# libtpu is built and uploaded to this link every night (PST).
if '_tpu' in project_name:
  # For tensorflow-tpu releases, use a set libtpu version;
  # For tf-nightly-tpu, use the most recent libtpu-nightly. Because of the
  # timing of these tests, the UTC date from eight hours ago is expected to be a
  # valid version.
  _libtpu_version = standard_or_nightly(
      LATEST_RELEASE_LIBTPU_VERSION,
      NEXT_LIBTPU_VERSION + '.dev'
      + (
          datetime.datetime.now(tz=datetime.timezone.utc)
          - datetime.timedelta(hours=8)
      ).strftime('%Y%m%d') + '+nightly',
  )
  REQUIRED_PACKAGES.append([f'libtpu=={_libtpu_version}'])
  CONSOLE_SCRIPTS.extend([
      'start_grpc_tpu_worker = tensorflow.python.tools.grpc_tpu_worker:run',
      ('start_grpc_tpu_service = '
       'tensorflow.python.tools.grpc_tpu_worker_service:run'),
  ])

if os.name == 'nt':
  EXTENSION_NAME = 'python/_pywrap_tensorflow_internal.pyd'
else:
  EXTENSION_NAME = 'python/_pywrap_tensorflow_internal.so'

headers = (
    list(find_files('*.proto', 'tensorflow/compiler'))
    + list(find_files('*.proto', 'tensorflow/core'))
    + list(find_files('*.proto', 'tensorflow/python'))
    + list(find_files('*.proto', 'tensorflow/python/framework'))
    + list(find_files('*.proto', 'tensorflow/tsl'))
    + list(find_files('*.def', 'tensorflow/compiler'))
    + list(find_files('*.h', 'tensorflow/c'))
    + list(find_files('*.h', 'tensorflow/cc'))
    + list(find_files('*.h', 'tensorflow/compiler'))
    + list(find_files('*.h.inc', 'tensorflow/compiler'))
    + list(find_files('*.h', 'tensorflow/core'))
    + list(find_files('*.h', 'tensorflow/lite/kernels/shim'))
    + list(find_files('*.h', 'tensorflow/python'))
    + list(find_files('*.h', 'tensorflow/python/client'))
    + list(find_files('*.h', 'tensorflow/python/framework'))
    + list(find_files('*.h', 'tensorflow/stream_executor'))
    + list(find_files('*.h', 'tensorflow/compiler/xla/stream_executor'))
    + list(find_files('*.h', 'tensorflow/tsl'))
    + list(find_files('*.h', 'google/com_google_protobuf/src'))
    + list(find_files('*.inc', 'google/com_google_protobuf/src'))
    + list(find_files('*', 'third_party/gpus'))
    + list(find_files('*.h', 'tensorflow/include/external/com_google_absl'))
    + list(find_files('*.inc', 'tensorflow/include/external/com_google_absl'))
    + list(find_files('*.h', 'tensorflow/include/external/ducc/google'))
    + list(find_files('*', 'tensorflow/include/external/eigen_archive'))
    + list(find_files('*.h', 'tensorflow/include/external/ml_dtypes_py'))
)

# Quite a lot of setup() options are different if this is a collaborator package
# build. We explicitly list the differences here, then unpack the dict as
# options at the end of the call to setup() below. For what each keyword does,
# see https://setuptools.pypa.io/en/latest/references/keywords.html.
if collaborator_build:
  collaborator_build_dependent_options = {
      'cmdclass': {},
      'distclass': None,
      'entry_points': {},
      'headers': [],
      'include_package_data': None,
      'packages': [],
      'package_data': {},
  }
else:
  collaborator_build_dependent_options = {
      'cmdclass': {
          'install_headers': InstallHeaders,
          'install': InstallCommand,
      },
      'distclass': BinaryDistribution,
      'entry_points': {
          'console_scripts': CONSOLE_SCRIPTS,
      },
      'headers': headers,
      'include_package_data': True,
      'packages': find_namespace_packages(),
      'package_data': {
          'tensorflow': [EXTENSION_NAME] + matches,
      },
  }

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
    extras_require=EXTRA_PACKAGES,
    # Add in any packaged data.
    zip_safe=False,
    # Supported Python versions
    python_requires='>=3.9',
    # PyPI package information.
    classifiers=sorted([
        'Development Status :: 5 - Production/Stable',
        # TODO(angerson) Add IFTTT when possible
        'Environment :: GPU :: NVIDIA CUDA :: 12',
        'Environment :: GPU :: NVIDIA CUDA :: 12 :: 12.2',
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
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]),
    license='Apache 2.0',
    keywords='tensorflow tensor machine learning',
    **collaborator_build_dependent_options
)
