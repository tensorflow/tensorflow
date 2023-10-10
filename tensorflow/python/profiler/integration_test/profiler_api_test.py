# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf 2.x profiler."""

import glob
import os
import threading

import portpicker

from tensorflow.python.distribute import collective_all_reduce_strategy as collective_strategy
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import profiler_client
from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.python.profiler.integration_test import mnist_testing_utils


def _model_setup():
  """Set up a MNIST Keras model for testing purposes.

  Builds a MNIST Keras model and returns model information.

  Returns:
    A tuple of (batch_size, steps, train_dataset, mode)
  """
  context.set_log_device_placement(True)
  batch_size = 64
  steps = 2
  with collective_strategy.CollectiveAllReduceStrategy().scope():
    # TODO(b/142509827): In rare cases this errors out at C++ level with the
    # "Connect failed" error message.
    train_ds, _ = mnist_testing_utils.mnist_synthetic_dataset(batch_size, steps)
    model = mnist_testing_utils.get_mnist_model((28, 28, 1))
  return batch_size, steps, train_ds, model


def _make_temp_log_dir(test_obj):
  return test_obj.get_temp_dir()


class ProfilerApiTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super().setUp()
    self.worker_start = threading.Event()
    self.profile_done = False

  def _check_xspace_pb_exist(self, logdir):
    path = os.path.join(logdir, 'plugins', 'profile', '*', '*.xplane.pb')
    self.assertEqual(1, len(glob.glob(path)),
                     'Expected one path match: ' + path)

  def test_single_worker_no_profiling(self):
    """Test single worker without profiling."""

    _, steps, train_ds, model = _model_setup()

    model.fit(x=train_ds, epochs=2, steps_per_epoch=steps)

  def test_single_worker_sampling_mode(self, delay_ms=None):
    """Test single worker sampling mode."""

    def on_worker(port, worker_start):
      logging.info('worker starting server on {}'.format(port))
      profiler.start_server(port)
      _, steps, train_ds, model = _model_setup()
      worker_start.set()
      while True:
        model.fit(x=train_ds, epochs=2, steps_per_epoch=steps)
        if self.profile_done:
          break

    def on_profile(port, logdir, worker_start):
      # Request for 30 milliseconds of profile.
      duration_ms = 30

      worker_start.wait()
      options = profiler.ProfilerOptions(
          host_tracer_level=2,
          python_tracer_level=0,
          device_tracer_level=1,
          delay_ms=delay_ms,
      )

      profiler_client.trace('localhost:{}'.format(port), logdir, duration_ms,
                            '', 100, options)

      self.profile_done = True

    logdir = self.get_temp_dir()
    port = portpicker.pick_unused_port()
    thread_profiler = threading.Thread(
        target=on_profile, args=(port, logdir, self.worker_start))
    thread_worker = threading.Thread(
        target=on_worker, args=(port, self.worker_start))
    thread_worker.start()
    thread_profiler.start()
    thread_profiler.join()
    thread_worker.join(120)
    self._check_xspace_pb_exist(logdir)

  def test_single_worker_sampling_mode_short_delay(self):
    """Test single worker sampling mode with a short delay.

    Expect that requested delayed start time will arrive late, and a subsequent
    retry will issue an immediate start.
    """

    self.test_single_worker_sampling_mode(delay_ms=1)

  def test_single_worker_sampling_mode_long_delay(self):
    """Test single worker sampling mode with a long delay."""

    self.test_single_worker_sampling_mode(delay_ms=1000)

  def test_single_worker_programmatic_mode(self):
    """Test single worker programmatic mode."""
    logdir = self.get_temp_dir()

    options = profiler.ProfilerOptions(
        host_tracer_level=2,
        python_tracer_level=0,
        device_tracer_level=1,
    )
    profiler.start(logdir, options)
    _, steps, train_ds, model = _model_setup()
    model.fit(x=train_ds, epochs=2, steps_per_epoch=steps)
    profiler.stop()
    self._check_xspace_pb_exist(logdir)


if __name__ == '__main__':
  multi_process_runner.test_main()
