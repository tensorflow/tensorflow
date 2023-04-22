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
"""Utilities related to distributed training."""
# pylint:disable=protected-access

from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.keras import backend
from tensorflow.python.ops import variables


# TODO(b/118776054): Currently we support global batch size for TPUStrategy and
# core MirroredStrategy only. Remove this check when contrib MirroredStrategy is
# no longer needed.
def global_batch_size_supported(distribution_strategy):
  return distribution_strategy.extended._global_batch_size  # pylint: disable=protected-access


def call_replica_local_fn(fn, *args, **kwargs):
  """Call a function that uses replica-local variables.

  This function correctly handles calling `fn` in a cross-replica
  context.

  Args:
    fn: The function to call.
    *args: Positional arguments to the `fn`.
    **kwargs: Keyword argument to `fn`.

  Returns:
    The result of calling `fn`.
  """
  # TODO(b/132666209): Remove this function when we support assign_*
  # for replica-local variables.
  strategy = None
  if 'strategy' in kwargs:
    strategy = kwargs.pop('strategy')
  else:
    if ds_context.has_strategy():
      strategy = ds_context.get_strategy()

  # TODO(b/120571621): TPUStrategy does not implement replica-local variables.
  is_tpu = backend.is_tpu_strategy(strategy)
  if ((not is_tpu) and strategy and ds_context.in_cross_replica_context()):
    with strategy.scope():
      return strategy.extended.call_for_each_replica(fn, args, kwargs)
  return fn(*args, **kwargs)


def is_distributed_variable(v):
  """Returns whether `v` is a distributed variable."""
  return (isinstance(v, values_lib.DistributedValues) and
          isinstance(v, variables.Variable))
