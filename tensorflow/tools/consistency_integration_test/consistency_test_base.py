# Lint as: python2, python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests to improve the the of tensorflow.

Here we would like to include high level tests that stress tf.function and
autograph in ways users have discovered.  Not everything here has to work,
some things just need to have good error messages.  some things currently
have bugs assigned to them but do not work and do not have sufficient error
messages.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import enum
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow.python.util import tf_inspect


# Example. to run test case against.
#
# arg: Argument tuple to function callable
# out: Expected output.
# failure: List of `RunMode` enums that are expected to fail.
# bugs: List of bugs that are related to this test case.
Example = collections.namedtuple('Example', ['arg', 'out', 'failure', 'bugs'])


class RunMode(enum.Enum):
  RAW = 0
  FUNCTION = 1
  SAVED = 2
  XLA = 3


dashboard_data = {}


class ConsistencyTestBase(tf.test.TestCase):
  """Tests that attempt to use py function's in the 4 use-examples.

  The example kinds are:
  raw, tf.function'ified, tf.function xlaified, and loaded from saved model.
  """

  def recordProperty(self, property_name, property_value):
    """Wrapper to handle recording properties.

    Args:
      property_name: Name of property to record.
      property_value: Value to record associated with `property_name`.

    Open source does not have record property.
    """
    base = super(ConsistencyTestBase, self)
    if hasattr(base, 'recordProperty'):
      getattr(base, 'recordProperty')(property_name, property_value)

  def _deep_equal(self, left, right):
    if isinstance(left, tf.Tensor):
      return self._deep_equal(left.numpy(), right)
    if isinstance(right, tf.Tensor):
      return self._deep_equal(left, right.numpy())
    if isinstance(left, tf.SparseTensor) and isinstance(right, tf.SparseTensor):
      return (self._deep_equal(left.indices, right.indices)
              and self._deep_equal(left.values, right.values)
              and self._deep_equal(left.shape, right.shape))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
      return np.array_equal(left, right)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
      return all(self._deep_equal(l, r) for l, r in zip(left, right))
    return left == right

  def _run_and_check(self, f, mode, examples):
    for arg, out, failure, bugs in examples:
      del bugs
      if mode in failure:
        with self.assertRaisesWithPredicateMatch(BaseException, '.*'):
          self._deep_equal(f(*arg), out)
      else:
        self._deep_equal(f(*arg), out)

  def _generic_test(self, f_raw, examples):
    """Test a function `f_raw` against all tests `examples`.

    Args:
      f_raw: a callable.
      examples: A list of `Example` named tuples.
    """
    self.recordProperty('f', tf_inspect.getsource(f_raw))
    for arg, out, failure, bugs in examples:
      del out
      self.recordProperty('Input "%r"' % arg, {'not-working': failure,
                                               'bugs': bugs})

    # Run the function without tf.function
    self._run_and_check(f_raw, RunMode.RAW, examples)
    # TF Function
    f_tf = tf.function(f_raw)
    self._run_and_check(f_tf, RunMode.FUNCTION, examples)
    # XLA Function
    f_xla = tf.function(f_raw, experimental_compile=True)
    self._run_and_check(f_xla, RunMode.XLA, examples)
    # Write a saved model and try to run it
    module = tf.Module()
    module.f = f_tf
    saved_model_dir = tempfile.gettempdir()
    tf.saved_model.save(module, saved_model_dir)
    module_loaded = tf.saved_model.load(saved_model_dir)
    self._run_and_check(module_loaded.f, RunMode.SAVED, examples)


if __name__ == '__main__':
  tf.test.main()
