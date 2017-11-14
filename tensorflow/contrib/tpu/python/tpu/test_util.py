# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ===================================================================
"""Utilities to ease testing on TPU devices."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.tpu.python.tpu import tpu

from tensorflow.python.client import session
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import variables


def has_tpu():
  """Check if a TPU device is available.

  Device enumeration via `device_lib` currently fails for TPU systems.
  (http://b/68333779).  To work around this, we determine the existence of a
  TPU by a successful call to `initialize_system`.

  Returns:
    boolean, True if a TPU device is available, otherwise False.
  """
  def _check():
    with session.Session() as sess:
      sess.run(tpu.initialize_system())
      sess.run(tpu.shutdown_system())

  try:
    _check()
    return True
  except errors.OpError as _:
    return False


def _available_devices():
  devices = ["cpu"]
  if not test_util.gpu_device_name():
    devices.append("gpu")

  if has_tpu():
    devices.append("tpu")

  return tuple(devices)


class TPUTestCase(test_util.TensorFlowTestCase):
  """Adds helpers for testing on TPU devices to `TensorFlowTestCase`.

  Example usage:

  ```
  def model_fn(features):
  return tf.reduce_sum(features * 2)

  class ModelTests(test_util.TPUTestCase):
    def test_sum(self):
      v = np.random.randn(10, 10).astype("float32")
      self.assert_device_output(model_fn, [v], (v*2).sum(),
                                devices=("cpu", "tpu"))
  ```
  """

  def __init__(self, methodName="runTest"):  # pylint: disable=invalid-name
    super(TPUTestCase, self).__init__(methodName)
    self._available_devices = _available_devices()

  def run_on_device(self, model_fn, model_inputs, device):
    """Runs `model_fn` on the given device.

    Raises an exception if no such device is available.  `model_fn` should
    return one or more tensors as a list or tuple.

    Args:
      model_fn: Function returning one or more tensors.
      model_inputs: An iterable of Numpy arrays or scalars.
                    These will be passed as arguments to `model_fn`.
      device: Device to run on.  One of ("tpu", "gpu", "cpu").

    Returns:
      Output from the model function.
    """
    def _make_placeholders():
      return dict(
          [(gen_array_ops.placeholder_with_default(v, v.shape), v)
           for v in model_inputs])

    if device == "tpu":
      with self.test_session(graph=ops.Graph()) as sess:
        placeholders = _make_placeholders()
        tpu_computation = tpu.rewrite(model_fn, placeholders.keys())
        sess.run(tpu.initialize_system())
        sess.run(variables.global_variables_initializer())
        result = sess.run(tpu_computation, placeholders)
        sess.run(tpu.shutdown_system())
        # TODO(b/36891278): supports non-flat returns lists in tpu.rewrite().
        if len(result) == 1:
          return result[0]
        return result
    elif device == "gpu":
      with self.test_session(graph=ops.Graph(), use_gpu=True) as sess:
        placeholders = _make_placeholders()
        sess.run(variables.global_variables_initializer())
        return sess.run(model_fn(placeholders.keys()), placeholders)
    elif device == "cpu":
      # TODO(power) -- will this interact poorly with cached GPU sessions?
      with self.test_session(graph=ops.Graph(), use_gpu=False) as sess:
        placeholders = _make_placeholders()
        sess.run(variables.global_variables_initializer())
        return sess.run(model_fn(placeholders.keys()), placeholders)

  def _compare_values(self, actual_outputs, expected_outputs):
    if isinstance(expected_outputs, (list, tuple)):
      for a, b in zip(actual_outputs, expected_outputs):
        self.assertAllCloseAccordingToType(a, b)
    else:
      self.assertAllCloseAccordingToType(actual_outputs, expected_outputs)

  def assert_device_output(self, model_fn, model_inputs, expected_outputs,
                           devices=("cpu", "gpu", "tpu")):
    """Run `model_fn` on the given devices.

    Results are compared via `assertAllCloseAccordingToType`.

    Args:
      model_fn: Function returning one or more tensors
      model_inputs: Numpy arrays or scalars passed as arguments to model_fn
      expected_outputs: Numpy arrays or scalars to compare against.
      devices: Set of devices to run on.  If a device is not available, tests
               will be skipped for that device.
    """
    devices = set(devices).intersection(self._available_devices)

    for device in devices:
      device_out = self.run_on_device(model_fn, model_inputs, device=device)
      self._compare_values(device_out, expected_outputs)
