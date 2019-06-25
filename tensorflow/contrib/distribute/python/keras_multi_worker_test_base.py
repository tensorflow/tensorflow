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
"""Test base for tf.keras Models in multi-worker mode."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

# pylint: disable=g-direct-tensorflow-import
from tensorflow.contrib.distribute.python import collective_all_reduce_strategy as collective_strategy
from tensorflow.contrib.distribute.python import parameter_server_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_coordinator as dc
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.eager import context
from tensorflow.python.platform import test

_original_run_std_server = dc._run_std_server  # pylint: disable=protected-access

# Used as a decorator on test methods.
run_sync_strategies = combinations.generate(
    combinations.combine(
        mode=['graph'],
        strategy_cls=[
            collective_strategy.CollectiveAllReduceStrategy,
        ],
        required_gpus=[0, 1]))

# Used as a decorator on test methods.
run_async_strategies = combinations.generate(
    combinations.combine(
        mode=['graph'],
        strategy_cls=[parameter_server_strategy.ParameterServerStrategy],
        required_gpus=[0, 1]))


def get_strategy_object(strategy_cls):
  return strategy_cls(num_gpus_per_worker=context.num_gpus())


# TODO(omalleyt): Merge with keras_multiworker_callback_test
class KerasIndependentWorkerTestBase(
    multi_worker_test_base.IndependentWorkerTestBase):
  """Test base for simulating Keras Multi-Worker in threads."""

  def _make_mock_run_std_server(self):
    thread_local = threading.local()

    def _mock_run_std_server(*args, **kwargs):
      ret = _original_run_std_server(*args, **kwargs)
      # Wait for all std servers to be brought up in order to reduce the chance
      # of remote sessions taking local ports that have been assigned to std
      # servers. Only call this barrier the first time this function is run for
      # each thread.
      if not getattr(thread_local, 'server_started', False):
        self._barrier.wait()
      thread_local.server_started = True
      return ret

    return _mock_run_std_server

  def run_independent_workers(self,
                              worker_fn,
                              strategy_cls,
                              num_workers,
                              num_ps=None,
                              **kwargs):
    cluster_spec = multi_worker_test_base.create_cluster_spec(
        num_workers=num_workers, num_ps=num_ps)
    self._barrier = dc._Barrier(num_workers + (num_ps or 0))  # pylint: disable=protected-access

    def _worker_fn(**kwargs):
      """Runs the worker function in a thread."""
      with test.mock.patch.object(dc, '_run_std_server',
                                  self._make_mock_run_std_server()):
        strategy = get_strategy_object(strategy_cls)
        with strategy.scope():
          return worker_fn(**kwargs)

    threads = self.run_multiple_tasks_in_threads(_worker_fn, cluster_spec,
                                                 **kwargs)
    strategy = get_strategy_object(strategy_cls)
    if strategy.extended.experimental_between_graph:
      threads_to_join = threads.get('chief', []) + threads.get('worker', [])
    else:
      threads_to_join = [
          threads['chief'][0] if 'chief' in threads else threads['worker'][0]
      ]
    self.join_independent_workers(threads_to_join)
