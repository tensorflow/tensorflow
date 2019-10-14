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

import time

from absl import flags
from six.moves import queue as Queue

from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.eager import test

flags.DEFINE_boolean(name='test_flag', default=0, help='Test flag')


def proc_func_that_adds_task_type_in_return_data(test_obj):
  test_obj.assertTrue(flags.FLAGS.test_flag == 3)
  return multi_worker_test_base.get_task_type()


def proc_func_that_errors():
  raise ValueError('This is an error.')


def proc_func_that_does_nothing():
  pass


def proc_func_that_adds_simple_return_data():
  return 'dummy_data'


def proc_func_that_verifies_args(*args, **kwargs):
  for arg in args:
    multi_process_runner.add_return_data(arg)
  for kwarg in kwargs.items():
    multi_process_runner.add_return_data(kwarg)


class MultiProcessRunnerTest(test.TestCase):

  def test_multi_process_runner(self):
    job_count_dict = {'worker': 2, 'ps': 3, 'evaluator': 2}
    proc_flags = {
        'test_flag': 3,
    }
    returned_data = multi_process_runner.run(
        proc_func_that_adds_task_type_in_return_data,
        multi_process_runner.job_count_to_cluster_spec(job_count_dict),
        proc_flags=proc_flags,
        args=(self,))

    for data in returned_data:
      job_count_dict[data] -= 1

    self.assertEqual(job_count_dict['worker'], 0)
    self.assertEqual(job_count_dict['ps'], 0)
    self.assertEqual(job_count_dict['evaluator'], 0)

  def test_multi_process_runner_error_propagates_from_subprocesses(self):
    job_count_dict = {'worker': 1, 'ps': 1}
    with self.assertRaisesRegexp(ValueError, 'This is an error.'):
      multi_process_runner.run(
          proc_func_that_errors,
          multi_process_runner.job_count_to_cluster_spec(job_count_dict),
          timeout=20)

  def test_multi_process_runner_queue_emptied_between_runs(self):
    job_count_dict = {'worker': 2}
    cluster_spec = multi_process_runner.job_count_to_cluster_spec(
        job_count_dict)
    returned_data = multi_process_runner.run(
        proc_func_that_adds_simple_return_data, cluster_spec)
    self.assertTrue(returned_data)
    self.assertEqual(returned_data[0], 'dummy_data')
    self.assertEqual(returned_data[1], 'dummy_data')
    returned_data = multi_process_runner.run(proc_func_that_does_nothing,
                                             cluster_spec)
    self.assertFalse(returned_data)

  def test_multi_process_runner_args_passed_correctly(self):
    job_count_dict = {'worker': 1}
    returned_data = multi_process_runner.run(
        proc_func_that_verifies_args,
        multi_process_runner.job_count_to_cluster_spec(job_count_dict),
        args=('a', 'b'),
        kwargs={'c_k': 'c_v'})
    self.assertEqual(returned_data[0], 'a')
    self.assertEqual(returned_data[1], 'b')
    self.assertEqual(returned_data[2], ('c_k', 'c_v'))

  def test_stdout_captured(self):

    def simple_print_func():
      print('This is something printed.')
      return 'This is returned data.'

    job_count_dict = {'worker': 2}
    returned_data, std_stream_data = multi_process_runner.run(
        simple_print_func,
        multi_process_runner.job_count_to_cluster_spec(job_count_dict),
        return_std_stream=True)
    num_string_std_stream = len(
        [d for d in std_stream_data if d == 'This is something printed.'])
    num_string_returned_data = len(
        [d for d in returned_data if d == 'This is returned data.'])
    self.assertEqual(num_string_std_stream, 2)
    self.assertEqual(num_string_returned_data, 2)

  def test_process_that_exits(self):

    def func_to_exit_in_10_sec():
      time.sleep(5)
      multi_process_runner.add_return_data('foo')
      time.sleep(20)
      multi_process_runner.add_return_data('bar')

    job_count_dict = {'worker': 1}
    returned_data = multi_process_runner.run(
        func_to_exit_in_10_sec,
        multi_process_runner.job_count_to_cluster_spec(job_count_dict),
        time_to_exit=10)
    self.assertLen(returned_data, 1)

  def test_signal_doesnt_fire_after_process_exits(self):
    job_count_dict = {'worker': 1}
    multi_process_runner.run(
        proc_func_that_does_nothing,
        multi_process_runner.job_count_to_cluster_spec(job_count_dict),
        time_to_exit=10)
    time.sleep(15)
    with self.assertRaisesRegexp(Queue.Empty, ''):
      # If the signal was fired, another message would be added to internal
      # queue, so verifying it's empty.
      multi_process_runner._get_internal_queue().get(block=False)


if __name__ == '__main__':
  multi_process_runner.test_main()
