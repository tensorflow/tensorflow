# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Base testing class for strategies that require multiple nodes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import copy

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import test
from tensorflow.python.framework import test_util


class MultiWorkerTestBase(test.TestCase):
  """Base class for testing multi node strategy and dataset."""

  @classmethod
  def setUpClass(cls):
    """Create a local cluster with 2 workers."""
    num_workers = 2
    # Leave some memory for cuda runtime.
    gpu_mem_frac = 0.7 / num_workers
    default_config = config_pb2.ConfigProto()
    default_config.gpu_options.per_process_gpu_memory_fraction = gpu_mem_frac

    # The local cluster takes some portion of the local GPUs and there is no way
    # for the cluster to terminate unless using multiple processes. Therefore,
    # we have to only create only one cluster throughout a test process.
    workers, _ = test_util.create_local_cluster(
        num_workers, num_ps=0, worker_config=default_config)
    cls._master_target = workers[0].target

  @contextlib.contextmanager
  def test_session(self, graph=None, config=None):
    """Create a test session with master target set to the testing cluster.

    This overrides the base class' method, removes arguments that are not needed
    by the multi-node case and creates a test session that connects to the local
    testing cluster.

    Args:
      graph: Optional graph to use during the returned session.
      config: An optional config_pb2.ConfigProto to use to configure the
        session.

    Yields:
      A Session object that should be used as a context manager to surround
      the graph building and execution code in a test case.
    """
    if self.id().endswith('.test_session'):
      self.skipTest('Not a test.')

    if config is None:
      config = config_pb2.ConfigProto(allow_soft_placement=True)
    else:
      config = copy.deepcopy(config)
    # Don't perform optimizations for tests so we don't inadvertently run
    # gpu ops on cpu
    config.graph_options.optimizer_options.opt_level = -1
    config.graph_options.rewrite_options.constant_folding = (
        rewriter_config_pb2.RewriterConfig.OFF)

    if graph is None:
      if self._cached_session is None:  # pylint: disable=access-member-before-definition
        self._cached_session = session.Session(
            graph=None, config=config, target=self._master_target)
      sess = self._cached_session
      with sess.graph.as_default(), sess.as_default():
        yield sess
    else:
      with session.Session(
          graph=graph, config=config, target=self._master_target) as sess:
        yield sess
