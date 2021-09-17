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

"""Testing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# pylint: disable=g-bad-import-order
from tensorflow.python.framework import test_util as _test_util
from tensorflow.python.platform import googletest as _googletest

# pylint: disable=unused-import
from tensorflow.python.framework.test_util import assert_equal_graph_def
from tensorflow.python.framework.test_util import create_local_cluster
from tensorflow.python.framework.test_util import TensorFlowTestCase as TestCase
from tensorflow.python.framework.test_util import gpu_device_name
from tensorflow.python.framework.test_util import is_gpu_available

from tensorflow.python.ops.gradient_checker import compute_gradient_error
from tensorflow.python.ops.gradient_checker import compute_gradient
# pylint: enable=unused-import,g-bad-import-order

import functools

import sys
from tensorflow.python.util.tf_export import tf_export
if sys.version_info.major == 2:
  import mock                # pylint: disable=g-import-not-at-top,unused-import
else:
  from unittest import mock  # pylint: disable=g-import-not-at-top,g-importing-member

tf_export(v1=['test.mock'])(mock)

# Import Benchmark class
Benchmark = _googletest.Benchmark  # pylint: disable=invalid-name

# Import StubOutForTesting class
StubOutForTesting = _googletest.StubOutForTesting  # pylint: disable=invalid-name


@tf_export('test.main')
def main(argv=None):
  """Runs all unit tests."""
  _test_util.InstallStackTraceHandler()
  return _googletest.main(argv)


@tf_export(v1=['test.get_temp_dir'])
def get_temp_dir():
  """Returns a temporary directory for use during tests.

  There is no need to delete the directory after the test.

  @compatibility(TF2)
  This function is removed in TF2. Please use `TestCase.get_temp_dir` instead
  in a test case.
  Outside of a unit test, obtain a temporary directory through Python's
  `tempfile` module.
  @end_compatibility

  Returns:
    The temporary directory.
  """
  return _googletest.GetTempDir()


@tf_export(v1=['test.test_src_dir_path'])
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
  """Returns whether TensorFlow was built with CUDA (GPU) support.

  This method should only be used in tests written with `tf.test.TestCase`. A
  typical usage is to skip tests that should only run with CUDA (GPU).

  >>> class MyTest(tf.test.TestCase):
  ...
  ...   def test_add_on_gpu(self):
  ...     if not tf.test.is_built_with_cuda():
  ...       self.skipTest("test is only applicable on GPU")
  ...
  ...     with tf.device("GPU:0"):
  ...       self.assertEqual(tf.math.add(1.0, 2.0), 3.0)

  TensorFlow official binary is built with CUDA.
  """
  return _test_util.IsGoogleCudaEnabled()


@tf_export('test.is_built_with_rocm')
def is_built_with_rocm():
  """Returns whether TensorFlow was built with ROCm (GPU) support.

  This method should only be used in tests written with `tf.test.TestCase`. A
  typical usage is to skip tests that should only run with ROCm (GPU).

  >>> class MyTest(tf.test.TestCase):
  ...
  ...   def test_add_on_gpu(self):
  ...     if not tf.test.is_built_with_rocm():
  ...       self.skipTest("test is only applicable on GPU")
  ...
  ...     with tf.device("GPU:0"):
  ...       self.assertEqual(tf.math.add(1.0, 2.0), 3.0)

  TensorFlow official binary is NOT built with ROCm.
  """
  return _test_util.IsBuiltWithROCm()


@tf_export('test.disable_with_predicate')
def disable_with_predicate(pred, skip_message):
  """Disables the test if pred is true."""

  def decorator_disable_with_predicate(func):

    @functools.wraps(func)
    def wrapper_disable_with_predicate(self, *args, **kwargs):
      if pred():
        self.skipTest(skip_message)
      else:
        return func(self, *args, **kwargs)

    return wrapper_disable_with_predicate

  return decorator_disable_with_predicate


@tf_export('test.is_built_with_gpu_support')
def is_built_with_gpu_support():
  """Returns whether TensorFlow was built with GPU (CUDA or ROCm) support.

  This method should only be used in tests written with `tf.test.TestCase`. A
  typical usage is to skip tests that should only run with GPU.

  >>> class MyTest(tf.test.TestCase):
  ...
  ...   def test_add_on_gpu(self):
  ...     if not tf.test.is_built_with_gpu_support():
  ...       self.skipTest("test is only applicable on GPU")
  ...
  ...     with tf.device("GPU:0"):
  ...       self.assertEqual(tf.math.add(1.0, 2.0), 3.0)

  TensorFlow official binary is built with CUDA GPU support.
  """
  return is_built_with_cuda() or is_built_with_rocm()


@tf_export('test.is_built_with_xla')
def is_built_with_xla():
  """Returns whether TensorFlow was built with XLA support.

  This method should only be used in tests written with `tf.test.TestCase`. A
  typical usage is to skip tests that should only run with XLA.

  >>> class MyTest(tf.test.TestCase):
  ...
  ...   def test_add_on_xla(self):
  ...     if not tf.test.is_built_with_xla():
  ...       self.skipTest("test is only applicable on XLA")

  ...     @tf.function(jit_compile=True)
  ...     def add(x, y):
  ...       return tf.math.add(x, y)
  ...
  ...     self.assertEqual(add(tf.ones(()), tf.ones(())), 2.0)

  TensorFlow official binary is built with XLA.
  """
  return _test_util.IsBuiltWithXLA()
