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
import threading
import time
from absl import logging
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
    mpr_result = multi_process_runner.run(
        proc_func_that_adds_task_type_in_return_data,
        multi_worker_test_base.create_cluster_spec(
            num_workers=2, num_ps=3, has_eval=1),
        args=(self, 3))

    job_count_dict = {'worker': 2, 'ps': 3, 'evaluator': 1}
    for data in mpr_result.return_value:
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
    return_value = multi_process_runner.run(
        proc_func_that_adds_simple_return_data, cluster_spec).return_value
    self.assertTrue(return_value)
    self.assertEqual(return_value[0], 'dummy_data')
    self.assertEqual(return_value[1], 'dummy_data')
    return_value = multi_process_runner.run(proc_func_that_does_nothing,
                                            cluster_spec).return_value
    self.assertFalse(return_value)

  def test_multi_process_runner_args_passed_correctly(self):
    return_value = multi_process_runner.run(
        proc_func_that_return_args_and_kwargs,
        multi_worker_test_base.create_cluster_spec(num_workers=1),
        args=('a', 'b'),
        kwargs={
            'c_k': 'c_v'
        }).return_value
    self.assertEqual(return_value[0][0], 'a')
    self.assertEqual(return_value[0][1], 'b')
    self.assertEqual(return_value[0][2], ('c_k', 'c_v'))

  def test_stdout_captured(self):

    def simple_print_func():
      print('This is something printed.', flush=True)
      return 'This is returned data.'

    mpr_result = multi_process_runner.run(
        simple_print_func,
        multi_worker_test_base.create_cluster_spec(num_workers=2),
        list_stdout=True)
    std_stream_results = mpr_result.stdout
    return_value = mpr_result.return_value
    self.assertIn('[worker-0]:    This is something printed.\n',
                  std_stream_results)
    self.assertIn('[worker-1]:    This is something printed.\n',
                  std_stream_results)
    self.assertIn('This is returned data.', return_value)

  def test_process_that_exits(self):

    def func_to_exit_in_15_sec():
      time.sleep(5)
      print('foo', flush=True)
      time.sleep(20)
      print('bar', flush=True)

    mpr = multi_process_runner.MultiProcessRunner(
        func_to_exit_in_15_sec,
        multi_worker_test_base.create_cluster_spec(num_workers=1),
        list_stdout=True,
        max_run_time=15)

    mpr.start()
    stdout = mpr.join().stdout
    self.assertLen([msg for msg in stdout if 'foo' in msg], 1)
    self.assertLen([msg for msg in stdout if 'bar' in msg], 0)

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
      multi_process_runner._resource(
          multi_process_runner.PROCESS_STATUS_QUEUE).get(block=False)

  def test_termination(self):

    def proc_func():
      for i in range(0, 10):
        print(
            'index {}, iteration {}'.format(self._worker_idx(), i), flush=True)
        time.sleep(5)

    mpr = multi_process_runner.MultiProcessRunner(
        proc_func,
        multi_worker_test_base.create_cluster_spec(num_workers=2),
        list_stdout=True)
    mpr.start()
    time.sleep(5)
    mpr.terminate('worker', 0)
    std_stream_results = mpr.join().stdout

    # Worker 0 is terminated in the middle, so it should not have iteration 9
    # printed.
    self.assertIn('[worker-0]:    index 0, iteration 0\n', std_stream_results)
    self.assertNotIn('[worker-0]:    index 0, iteration 9\n',
                     std_stream_results)
    self.assertIn('[worker-1]:    index 1, iteration 0\n', std_stream_results)
    self.assertIn('[worker-1]:    index 1, iteration 9\n', std_stream_results)

  def test_termination_and_start_single_process(self):

    def proc_func():
      for i in range(0, 10):
        print(
            'index {}, iteration {}'.format(self._worker_idx(), i), flush=True)
        time.sleep(1)

    mpr = multi_process_runner.MultiProcessRunner(
        proc_func,
        multi_worker_test_base.create_cluster_spec(num_workers=2),
        list_stdout=True)
    mpr.start()
    time.sleep(5)
    mpr.terminate('worker', 0)
    mpr.start_single_process('worker', 0)
    std_stream_results = mpr.join().stdout

    # Worker 0 is terminated in the middle, but a new worker 0 is added, so it
    # should still have iteration 9 printed. Moreover, iteration 0 of worker 0
    # should happen twice.
    self.assertLen(
        [s for s in std_stream_results if 'index 0, iteration 0' in s], 2)
    self.assertIn('[worker-0]:    index 0, iteration 9\n', std_stream_results)
    self.assertIn('[worker-1]:    index 1, iteration 0\n', std_stream_results)
    self.assertIn('[worker-1]:    index 1, iteration 9\n', std_stream_results)

  def test_streaming(self):

    def proc_func():
      for i in range(5):
        logging.info('(logging) %s-%d, i: %d',
                     multi_worker_test_base.get_task_type(), self._worker_idx(),
                     i)
        print(
            '(print) {}-{}, i: {}'.format(
                multi_worker_test_base.get_task_type(), self._worker_idx(), i),
            flush=True)
        time.sleep(1)

    mpr = multi_process_runner.MultiProcessRunner(
        proc_func,
        multi_worker_test_base.create_cluster_spec(
            has_chief=True, num_workers=2, num_ps=2, has_eval=True),
        list_stdout=True)
    mpr._dependence_on_chief = False

    mpr.start()
    mpr.start_single_process('worker', 2)
    mpr.start_single_process('ps', 2)
    mpr_result = mpr.join()

    list_to_assert = mpr_result.stdout

    for job in ['chief', 'evaluator']:
      for iteration in range(5):
        self.assertTrue(
            any('(logging) {}-0, i: {}'.format(job, iteration) in line
                for line in list_to_assert))
        self.assertTrue(
            any('(print) {}-0, i: {}'.format(job, iteration) in line
                for line in list_to_assert))

    for job in ['worker', 'ps']:
      for iteration in range(5):
        for task in range(3):
          self.assertTrue(
              any('(logging) {}-{}, i: {}'.format(job, task, iteration) in line
                  for line in list_to_assert))
          self.assertTrue(
              any('(print) {}-{}, i: {}'.format(job, task, iteration) in line
                  for line in list_to_assert))
        task = 3
        self.assertFalse(
            any('(logging) {}-{}, i: {}'.format(job, task, iteration) in line
                for line in list_to_assert))
        self.assertFalse(
            any('(print) {}-{}, i: {}'.format(job, task, iteration) in line
                for line in list_to_assert))

  def test_start_in_process_as(self):

    def proc_func():
      for i in range(5):
        logging.info('%s-%d, i: %d', multi_worker_test_base.get_task_type(),
                     self._worker_idx(), i)
        time.sleep(1)

    mpr = multi_process_runner.MultiProcessRunner(
        proc_func,
        multi_worker_test_base.create_cluster_spec(
            has_chief=True, num_workers=1),
        list_stdout=True)

    def follow_ups():
      mpr.start_single_process(task_type='evaluator', task_id=0)

    threading.Thread(target=follow_ups).start()
    mpr.start_in_process_as(as_task_type='chief', as_task_id=0)
    list_to_assert = mpr.join().stdout
    for job in ['worker', 'evaluator']:
      for iteration in range(5):
        self.assertTrue(
            any('{}-0, i: {}'.format(job, iteration) in line
                for line in list_to_assert))

  def test_terminate_all_does_not_ignore_error(self):
    mpr = multi_process_runner.MultiProcessRunner(
        proc_func_that_errors,
        multi_worker_test_base.create_cluster_spec(num_workers=2),
        list_stdout=True)
    mpr.start()
    mpr.terminate_all()
    with self.assertRaisesRegexp(ValueError, 'This is an error.'):
      mpr.join()

if __name__ == '__main__':
  multi_process_runner.test_main()
