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
"""A configure tuple for high-level APIs for running distribution strategies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


class DistributeConfig(
    collections.namedtuple(
        'DistributeConfig',
        ['train_distribute', 'eval_distribute', 'remote_cluster'])):
  """A config tuple for distribution strategies.

  Attributes:
    train_distribute: a `DistributionStrategy` object for training.
    eval_distribute: an optional `DistributionStrategy` object for
      evaluation.
    remote_cluster: a dict, `ClusterDef` or `ClusterSpec` object specifying
      the cluster configurations. If this is given, the `train_and_evaluate`
      method will be running as a standalone client which connects to the
      cluster for training.
  """

  def __new__(cls,
              train_distribute=None,
              eval_distribute=None,
              remote_cluster=None):
    return super(DistributeConfig, cls).__new__(cls, train_distribute,
                                                eval_distribute, remote_cluster)
