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

"""System configuration library.

API docstring: tensorflow.sysconfig
"""
import os.path as _os_path
import platform as _platform

from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework.versions import CXX11_ABI_FLAG as _CXX11_ABI_FLAG
from tensorflow.python.framework.versions import CXX_VERSION as _CXX_VERSION
from tensorflow.python.framework.versions import MONOLITHIC_BUILD as _MONOLITHIC_BUILD
from tensorflow.python.framework.versions import VERSION as _VERSION
from tensorflow.python.platform import build_info
from tensorflow.python.util.tf_export import tf_export


# pylint: disable=g-import-not-at-top
@tf_export('sysconfig.get_include')
def get_include():
  """Get the directory containing the TensorFlow C++ header files.

  Returns:
    The directory as string.
  """
  # Import inside the function.
  # sysconfig is imported from the tensorflow module, so having this
  # import at the top would cause a circular import, resulting in
  # the tensorflow module missing symbols that come after sysconfig.
  import tensorflow as tf
  return _os_path.join(_os_path.dirname(tf.__file__), 'include')


@tf_export('sysconfig.get_lib')
def get_lib():
  """Get the directory containing the TensorFlow framework library.

  Returns:
    The directory as string.
  """
  import tensorflow as tf
  return _os_path.join(_os_path.dirname(tf.__file__))


@tf_export('sysconfig.get_compile_flags')
def get_compile_flags():
  """Returns the compilation flags for compiling with TensorFlow.

  The returned list of arguments can be passed to the compiler for compiling
  against TensorFlow headers. The result is platform dependent.

  For example, on a typical Linux system with Python 3.7 the following command
  prints `['-I/usr/local/lib/python3.7/dist-packages/tensorflow/include',
  '-D_GLIBCXX_USE_CXX11_ABI=1', '-DEIGEN_MAX_ALIGN_BYTES=64']`

  >>> print(tf.sysconfig.get_compile_flags())

  Returns:
    A list of strings for the compiler flags.
  """
  flags = []
  flags.append('-I%s' % get_include())
  flags.append('-D_GLIBCXX_USE_CXX11_ABI=%d' % _CXX11_ABI_FLAG)
  cxx_version_flag = None
  if _CXX_VERSION == 201103:
    cxx_version_flag = '--std=c++11'
  elif _CXX_VERSION == 201402:
    cxx_version_flag = '--std=c++14'
  elif _CXX_VERSION == 201703:
    cxx_version_flag = '--std=c++17'
  elif _CXX_VERSION == 202002:
    cxx_version_flag = '--std=c++20'
  if cxx_version_flag:
    flags.append(cxx_version_flag)
  flags.append('-DEIGEN_MAX_ALIGN_BYTES=%d' %
               pywrap_tf_session.get_eigen_max_align_bytes())
  return flags


@tf_export('sysconfig.get_link_flags')
def get_link_flags():
  """Returns the linker flags for linking with TensorFlow.

  The returned list of arguments can be passed to the linker for linking against
  TensorFlow. The result is platform dependent.

  For example, on a typical Linux system with Python 3.7 the following command
  prints `['-L/usr/local/lib/python3.7/dist-packages/tensorflow',
  '-l:libtensorflow_framework.so.2']`

  >>> print(tf.sysconfig.get_link_flags())

  Returns:
    A list of strings for the linker flags.
  """
  is_mac = _platform.system() == 'Darwin'
  ver = _VERSION.split('.')[0]
  flags = []
  if not _MONOLITHIC_BUILD:
    flags.append('-L%s' % get_lib())
    if is_mac:
      flags.append('-ltensorflow_framework.%s' % ver)
    else:
      flags.append('-l:libtensorflow_framework.so.%s' % ver)
  return flags


@tf_export('sysconfig.get_build_info')
def get_build_info():
  """Get a dictionary describing TensorFlow's build environment.

  Values are generated when TensorFlow is compiled, and are static for each
  TensorFlow package. The return value is a dictionary with string keys such as:

    - cuda_version
    - cudnn_version
    - is_cuda_build
    - is_rocm_build
    - msvcp_dll_names
    - nvcuda_dll_name
    - cudart_dll_name
    - cudnn_dll_name

  Note that the actual keys and values returned by this function is subject to
  change across different versions of TensorFlow or across platforms.

  Returns:
    A Dictionary describing TensorFlow's build environment.
  """
  return build_info.build_info
