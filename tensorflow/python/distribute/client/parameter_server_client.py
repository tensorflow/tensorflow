# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Parameter server client module.

This is currently under development and the API is subject to change.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute.client import client


class ParameterServerClient(client.Client):
  """A client that uses `ParameterServerStrategy` to distribute tasks.

  Parameter server training refers to the distributed training architecture
  that requires two jobs in the cluster: workers and parameter servers. The
  variables and updates to those variables are assigned on the parameter
  servers' tasks, and the actual computation intensive operations are assigned
  on worker tasks. In TF2, parameter server training only starts up one
  client process, to drive and coordinate the workers and parameter servers.
  This is referred to as single-client architecture, as opposed to multi-client
  approach which is seen more often in traditional TensorFlow distributed
  training, including `tf.estimator.Estimator` and `tf.keras` with
  `tf.distribute.experimental.MultiWorkerMirroredStrategy`.

  `ParameterServerClient` is a `Client` that uses `ParameterServerStrategy` as
  the underlying strategy to distribute, and is the starting point of parameter
  server training/evaluation.

  If 'TF_CONFIG' environment variable is used, provide a
  `TFConfigClusterResolver` to detect configurations for multi-worker training.

  """

  def __init__(self, cluster_resolver):
    super(ParameterServerClient, self).__init__(
        parameter_server_strategy_v2.ParameterServerStrategyV2(
            cluster_resolver))
