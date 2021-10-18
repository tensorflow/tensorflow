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
# failure:
#   List of `RunMode` enums that are expected to fail, or
#   Dict of {'<`RunMode` enum>': 'error mesaage', ...} where keys are `RunMode`
#     enums that are excpected to fail and values are the corresponding
#     'error message' of the failure.
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
    if isinstance(left, tf.TensorArray):
      return self._deep_equal(left.stack(), right)
    if isinstance(right, tf.TensorArray):
      return self._deep_equal(left, right.stack())
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
      err_msg = '.*'
      # `failure` can be a list of `RunMode` enums or a dict of `RunMode` enum
      # and corresponding error message as key-value pairs:
      # `{'<`RunMode` enum>': 'error message', ...}`. If `failure` is a dict,
      # retrieve the error message corresponding to the `RunMode`.
      if isinstance(failure, dict):
        if mode in failure.keys():
          err_msg = failure[mode]

        # Get a list of `RunMode` enums from `failure` (dict) by getting the
        # keys to make it consistent with when `failure` is a list.
        failure = failure.keys()

      if mode in failure:
        with self.assertRaisesWithPredicateMatch(BaseException, err_msg):
          self._deep_equal(f(*arg), out)
      else:
        # Make sure `_deep_equal` returns True. Otherwise, mismatching results
        # (between `f(*arg)` and `out`) will not be caught.
        self.assertTrue(self._deep_equal(f(*arg), out))

  def _generic_test(self,
                    f_raw,
                    examples,
                    input_signature=None,
                    skip_modes=None):
    """Test a function `f_raw` against all tests `examples`.

    Args:
      f_raw: a callable.
      examples: A list of `Example` named tuples.
      input_signature: Input signature to tf.function.
      skip_modes: A list of `RunMode` enums to entirely skip testing in the
        specified `RunMode`s. This is necessary when things fail in a certain
        `RunMode` even before executing the function (e.g. during saving or
        loading in `RunMode.SAVED` mode).
    """
    f_tf = None
    if not skip_modes:
      skip_modes = []

    if tf_inspect.isfunction(f_raw):
      self.recordProperty('f', tf_inspect.getsource(f_raw))
    else:
      self.recordProperty('f', tf_inspect.getdoc(f_raw))

    for arg, out, failure, bugs in examples:
      del out
      self.recordProperty('Input "{}"'.format(arg), {
          'not-working': failure,
          'bugs': bugs
      })

    # Run the function without tf.function
    if RunMode.RAW not in skip_modes:
      self._run_and_check(f_raw, RunMode.RAW, examples)

    # TF Function
    if RunMode.FUNCTION not in skip_modes:
      f_tf = tf.function(f_raw, input_signature=input_signature)
      self._run_and_check(f_tf, RunMode.FUNCTION, examples)

    # XLA Function
    if RunMode.XLA not in skip_modes:
      f_xla = tf.function(
          f_raw, input_signature=input_signature, experimental_compile=True)
      self._run_and_check(f_xla, RunMode.XLA, examples)

    # Write a saved model and try to run it
    if RunMode.SAVED not in skip_modes:
      module = tf.Module()
      if f_tf:
        module.f = f_tf
      else:
        module.f = tf.function(f_raw, input_signature=input_signature)

      saved_model_dir = tempfile.gettempdir()
      tf.saved_model.save(module, saved_model_dir)
      module_loaded = tf.saved_model.load(saved_model_dir)
      self._run_and_check(module_loaded.f, RunMode.SAVED, examples)


if __name__ == '__main__':
  tf.test.main()
