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

import os.path
import pickle
import tempfile

import numpy as np

from tensorflow.contrib.tpu.python.tpu import tpu
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as tf_session
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as tf_saver


def has_tpu():
  """Check if a TPU device is available.

  Device enumeration via `device_lib` currently fails for TPU systems.
  (http://b/68333779).  To work around this, we determine the existence of a
  TPU by a successful call to `initialize_system`.

  Returns:
    boolean, True if a TPU device is available, otherwise False.
  """

  def _check():
    with tf_session.Session() as sess:
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


def copy_dir(src, tgt):
  """Copy src to tgt."""
  gfile.MakeDirs(tgt)
  seen_dirs = set()
  for dirname, _, files in gfile.Walk(src):
    for f in files:
      src_f = os.path.join(dirname, f)
      tgt_f = src_f.replace(src, tgt)
      tgt_d = os.path.dirname(tgt_f)
      if tgt_d not in seen_dirs:
        gfile.MkDir(tgt_d)
        seen_dirs.add(tgt_d)
      gfile.Copy(src_f, tgt_f, overwrite=True)


def compare_model(model_fn, input_fn, params, master="local", temp_dir=None,
                  tolerance=1e-4):
  """Compare the results of running `model_fn` on the TPU and CPU."""
  if not temp_dir:
    temp_dir = tempfile.mkdtemp()

  cpu_model_dir = "%s/cpu-model" % temp_dir
  tpu_model_dir = "%s/tpu-model" % temp_dir
  initial_model_dir = "%s/initial-model" % temp_dir

  logging.info("Checkpoints and weights will be written to %s", temp_dir)

  num_steps = 1
  num_shards = 8

  def _make_run_config(model_dir):
    return tpu_config.RunConfig(
        master=master,
        model_dir=model_dir,
        save_checkpoints_secs=10000,
        session_config=config_pb2.ConfigProto(
            allow_soft_placement=True, log_device_placement=False),
        tpu_config=tpu_config.TPUConfig(
            iterations_per_loop=num_steps,
            num_shards=num_shards,
        ),
    )

  def _make_estimator(use_tpu, model_dir):
    return tpu_estimator.TPUEstimator(
        model_fn=model_fn,
        use_tpu=use_tpu,
        config=_make_run_config(model_dir),
        train_batch_size=num_shards,
        params=dict(params, use_tpu=use_tpu),
    )

  def _extract_weights(checkpoint):
    """Extract model weights from the given checkpoint file."""
    weights = {}
    graph = ops.Graph()
    with graph.as_default():
      model_fn(
          *input_fn(params),
          params=dict(params, use_tpu=False),
          mode=model_fn_lib.ModeKeys.TRAIN)
      saver = tf_saver.Saver()
      with tf_session.Session(graph=graph) as sess:
        saver.restore(sess, checkpoint)
        all_vars = []
        all_vars.extend(graph.get_collection(ops.GraphKeys.GLOBAL_VARIABLES))
        all_vars.extend(graph.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES))
        all_vars.extend(graph.get_collection(ops.GraphKeys.MODEL_VARIABLES))

        for var in all_vars:
          weights[var.name] = sess.run(var)
    return weights

  def _run_step(use_tpu, model_dir):
    est = _make_estimator(use_tpu=use_tpu, model_dir=model_dir)
    est.train(input_fn=input_fn, steps=num_steps)
    weights = _extract_weights(est.latest_checkpoint())
    with gfile.Open(temp_dir + "tpu-%d.weights" % use_tpu, "wb") as f:
      f.write(pickle.dumps(weights))
    return weights

  # initialize models to the same weights by running a single step on the CPU
  _run_step(use_tpu=False, model_dir=initial_model_dir)

  copy_dir(initial_model_dir, cpu_model_dir)
  cpu_weights = _run_step(use_tpu=False, model_dir=cpu_model_dir)

  copy_dir(initial_model_dir, tpu_model_dir)
  tpu_weights = _run_step(use_tpu=True, model_dir=tpu_model_dir)

  bad_weights = False
  for k in cpu_weights:
    if k not in tpu_weights:
      raise KeyError("Missing weight %s from TPU checkpoint.", k)

    if not np.allclose(
        cpu_weights[k], tpu_weights[k], rtol=tolerance, atol=tolerance):
      bad_weights = True
      logging.error("Weights for layer %s have diverged.", k)

  if bad_weights:
    raise ValueError("Some weights have diverged.  Output pickle files have "
                     "been written to %s for inspection." % temp_dir)


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
      return dict([(gen_array_ops.placeholder_with_default(v, v.shape), v)
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

  def assert_device_output(self,
                           model_fn,
                           model_inputs,
                           expected_outputs,
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
