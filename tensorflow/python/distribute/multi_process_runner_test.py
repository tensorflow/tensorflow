# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `multi_process_runner`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import time

from six.moves import queue as Queue

from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.eager import test


def proc_func_that_adds_task_type_in_return_data(test_obj, val):
  test_obj.assertEqual(val, 3)
  return multi_worker_test_base.get_task_type()


def proc_func_that_errors():
  raise ValueError('This is an error.')


def proc_func_that_does_nothing():
  pass


def proc_func_that_adds_simple_return_data():
  return 'dummy_data'


def proc_func_that_return_args_and_kwargs(*args, **kwargs):
  return list(args) + list(kwargs.items())


class MultiProcessRunnerTest(test.TestCase):

  def _worker_idx(self):
    config_task = json.loads(os.environ['TF_CONFIG'])['task']
    return config_task['index']

  def test_multi_process_runner(self):
    returned_data, _ = multi_process_runner.run(
        proc_func_that_adds_task_type_in_return_data,
        multi_worker_test_base.create_cluster_spec(
            num_workers=2, num_ps=3, has_eval=1),
        args=(self, 3))

    job_count_dict = {'worker': 2, 'ps': 3, 'evaluator': 1}
    for data in returned_data:
      job_count_dict[data] -= 1

    self.assertEqual(job_count_dict['worker'], 0)
    self.assertEqual(job_count_dict['ps'], 0)
    self.assertEqual(job_count_dict['evaluator'], 0)

  def test_multi_process_runner_error_propagates_from_subprocesses(self):
    runner = multi_process_runner.MultiProcessRunner(
        proc_func_that_errors,
        multi_worker_test_base.create_cluster_spec(num_workers=1, num_ps=1),
        max_run_time=20)
    runner.start()
    with self.assertRaisesRegexp(ValueError, 'This is an error.'):
      runner.join()

  def test_multi_process_runner_queue_emptied_between_runs(self):
    cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
    returned_data, _ = multi_process_runner.run(
        proc_func_that_adds_simple_return_data, cluster_spec)
    self.assertTrue(returned_data)
    self.assertEqual(returned_data[0], 'dummy_data')
    self.assertEqual(returned_data[1], 'dummy_data')
    returned_data, _ = multi_process_runner.run(proc_func_that_does_nothing,
                                                cluster_spec)
    self.assertFalse(returned_data)

  def test_multi_process_runner_args_passed_correctly(self):
    returned_data, _ = multi_process_runner.run(
        proc_func_that_return_args_and_kwargs,
        multi_worker_test_base.create_cluster_spec(num_workers=1),
        args=('a', 'b'),
        kwargs={'c_k': 'c_v'})
    self.assertEqual(returned_data[0][0], 'a')
    self.assertEqual(returned_data[0][1], 'b')
    self.assertEqual(returned_data[0][2], ('c_k', 'c_v'))

  def test_stdout_captured(self):

    def simple_print_func():
      print('This is something printed.')
      return 'This is returned data.'

    returned_data, std_stream_data = multi_process_runner.run(
        simple_print_func,
        multi_worker_test_base.create_cluster_spec(num_workers=2),
        capture_std_stream=True)
    num_string_std_stream = len(
        [d for d in std_stream_data if d == 'This is something printed.'])
    num_string_returned_data = len(
        [d for d in returned_data if d == 'This is returned data.'])
    self.assertEqual(num_string_std_stream, 2)
    self.assertEqual(num_string_returned_data, 2)

  def test_process_that_exits(self):
    def func_to_exit_in_10_sec():
      time.sleep(5)
      mpr._add_return_data('foo')
      time.sleep(20)
      mpr._add_return_data('bar')

    mpr = multi_process_runner.MultiProcessRunner(
        func_to_exit_in_10_sec,
        multi_worker_test_base.create_cluster_spec(num_workers=1),
        max_run_time=10)

    mpr.start()
    returned_data, _ = mpr.join()
    self.assertLen(returned_data, 1)

  def test_signal_doesnt_fire_after_process_exits(self):
    mpr = multi_process_runner.MultiProcessRunner(
        proc_func_that_does_nothing,
        multi_worker_test_base.create_cluster_spec(num_workers=1),
        max_run_time=10)
    mpr.start()
    mpr.join()
    with self.assertRaisesRegexp(Queue.Empty, ''):
      # If the signal was fired, another message would be added to internal
      # queue, so verifying it's empty.
      mpr._get_process_status_queue().get(block=False)

  def test_termination(self):

    def proc_func():
      for i in range(0, 10):
        print('index {}, iteration {}'.format(self._worker_idx(), i))
        time.sleep(1)

    mpr = multi_process_runner.MultiProcessRunner(
        proc_func,
        multi_worker_test_base.create_cluster_spec(num_workers=2),
        capture_std_stream=True)
    mpr.start()
    time.sleep(5)
    mpr.terminate('worker', 0)
    std_stream_result = mpr.join()[1]

    # Worker 0 is terminated in the middle, so it should not have iteration 9
    # printed.
    self.assertIn('index 0, iteration 0', std_stream_result)
    self.assertNotIn('index 0, iteration 9', std_stream_result)
    self.assertIn('index 1, iteration 0', std_stream_result)
    self.assertIn('index 1, iteration 9', std_stream_result)

  def test_termination_and_start_single_process(self):

    def proc_func():
      for i in range(0, 10):
        print('index {}, iteration {}'.format(self._worker_idx(), i))
        time.sleep(1)

    mpr = multi_process_runner.MultiProcessRunner(
        proc_func,
        multi_worker_test_base.create_cluster_spec(num_workers=2),
        capture_std_stream=True)
    mpr.start()
    time.sleep(5)
    mpr.terminate('worker', 0)
    mpr.start_single_process('worker', 0)
    std_stream_result = mpr.join()[1]

    # Worker 0 is terminated in the middle, but a new worker 0 is added, so it
    # should still have iteration 9 printed. Moreover, iteration 0 of worker 0
    # should happen twice.
    self.assertLen(
        [s for s in std_stream_result if s == 'index 0, iteration 0'], 2)
    self.assertIn('index 0, iteration 9', std_stream_result)
    self.assertIn('index 1, iteration 0', std_stream_result)
    self.assertIn('index 1, iteration 9', std_stream_result)


if __name__ == '__main__':
  multi_process_runner.test_main()
