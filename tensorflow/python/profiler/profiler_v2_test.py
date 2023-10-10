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

import os
import socket

from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import gfile
from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.python.profiler import trace


class ProfilerTest(test_util.TensorFlowTestCase):

  def test_profile_exceptions(self):
    logdir = self.get_temp_dir()
    profiler.start(logdir)
    with self.assertRaises(errors.AlreadyExistsError):
      profiler.start(logdir)

    profiler.stop()
    with self.assertRaises(errors.UnavailableError):
      profiler.stop()

    # Test with a bad logdir, and it correctly raises exception and deletes
    # profiler.
    # pylint: disable=anomalous-backslash-in-string
    profiler.start('/dev/null/\/\/:123')
    # pylint: enable=anomalous-backslash-in-string
    with self.assertRaises(Exception):
      profiler.stop()
    profiler.start(logdir)
    profiler.stop()

  def test_save_profile(self):
    logdir = self.get_temp_dir()
    profiler.start(logdir)
    with trace.Trace('three_times_five'):
      three = constant_op.constant(3)
      five = constant_op.constant(5)
      product = three * five
    self.assertAllEqual(15, product)

    profiler.stop()
    file_list = gfile.ListDirectory(logdir)
    self.assertEqual(len(file_list), 1)
    for file_name in gfile.ListDirectory(logdir):
      if gfile.IsDirectory(os.path.join(logdir, file_name)):
        self.assertEqual(file_name, 'plugins')
    profile_dir = os.path.join(logdir, 'plugins', 'profile')
    run = gfile.ListDirectory(profile_dir)[0]
    hostname = socket.gethostname()
    xplane = os.path.join(profile_dir, run, hostname + '.xplane.pb')
    self.assertTrue(gfile.Exists(xplane))

  def test_profile_with_options(self):
    logdir = self.get_temp_dir()
    options = profiler.ProfilerOptions(
        host_tracer_level=3, python_tracer_level=1)
    profiler.start(logdir, options)
    with trace.Trace('three_times_five'):
      three = constant_op.constant(3)
      five = constant_op.constant(5)
      product = three * five
    self.assertAllEqual(15, product)

    profiler.stop()
    file_list = gfile.ListDirectory(logdir)
    self.assertEqual(len(file_list), 1)

  def test_context_manager_with_options(self):
    logdir = self.get_temp_dir()
    options = profiler.ProfilerOptions(
        host_tracer_level=3, python_tracer_level=1)
    with profiler.Profile(logdir, options):
      with trace.Trace('three_times_five'):
        three = constant_op.constant(3)
        five = constant_op.constant(5)
        product = three * five
      self.assertAllEqual(15, product)

    file_list = gfile.ListDirectory(logdir)
    self.assertEqual(len(file_list), 1)


if __name__ == '__main__':
  test.main()
