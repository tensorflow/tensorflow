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

"""Testing.

See the @{$python/test} guide.

Note: `tf.test.mock` is an alias to the python `mock` or `unittest.mock`
depending on the python version.

@@main
@@TestCase
@@test_src_dir_path
@@assert_equal_graph_def
@@get_temp_dir
@@is_built_with_cuda
@@is_gpu_available
@@gpu_device_name
@@compute_gradient
@@compute_gradient_error
@@create_local_cluster

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# pylint: disable=g-bad-import-order
from tensorflow.python.framework import test_util as _test_util
from tensorflow.python.platform import googletest as _googletest
from tensorflow.python.util.all_util import remove_undocumented

# pylint: disable=unused-import
from tensorflow.python.framework.test_util import assert_equal_graph_def
from tensorflow.python.framework.test_util import create_local_cluster
from tensorflow.python.framework.test_util import TensorFlowTestCase as TestCase
from tensorflow.python.framework.test_util import gpu_device_name
from tensorflow.python.framework.test_util import is_gpu_available

from tensorflow.python.ops.gradient_checker import compute_gradient_error
from tensorflow.python.ops.gradient_checker import compute_gradient
# pylint: enable=unused-import,g-bad-import-order

import sys
from tensorflow.python.util.tf_export import tf_export
if sys.version_info.major == 2:
  import mock                # pylint: disable=g-import-not-at-top,unused-import
else:
  from unittest import mock  # pylint: disable=g-import-not-at-top

tf_export('test.mock')(mock)

# Import Benchmark class
Benchmark = _googletest.Benchmark  # pylint: disable=invalid-name

# Import StubOutForTesting class
StubOutForTesting = _googletest.StubOutForTesting  # pylint: disable=invalid-name


@tf_export('test.main')
def main(argv=None):
  """Runs all unit tests."""
  _test_util.InstallStackTraceHandler()
  return _googletest.main(argv)


@tf_export('test.get_temp_dir')
def get_temp_dir():
  """Returns a temporary directory for use during tests.

  There is no need to delete the directory after the test.

  Returns:
    The temporary directory.
  """
  return _googletest.GetTempDir()


@tf_export('test.test_src_dir_path')
def test_src_dir_path(relative_path):
  """Creates an absolute test srcdir path given a relative path.

  Args:
    relative_path: a path relative to tensorflow root.
      e.g. "core/platform".

  Returns:
    An absolute path to the linked in runfiles.
  """
  return _googletest.test_src_dir_path(relative_path)


@tf_export('test.is_built_with_cuda')
def is_built_with_cuda():
  """Returns whether TensorFlow was built with CUDA (GPU) support."""
  return _test_util.IsGoogleCudaEnabled()


_allowed_symbols = [
    # We piggy-back googletest documentation.
    'Benchmark',
    'mock',
    'StubOutForTesting',
]

remove_undocumented(__name__, _allowed_symbols)
