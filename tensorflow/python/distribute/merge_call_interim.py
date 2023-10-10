# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""A module for interm merge-call related internal APIs."""
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.util.tf_export import tf_export


@tf_export("__internal__.distribute.strategy_supports_no_merge_call", v1=[])
def strategy_supports_no_merge_call():
  """Returns if the current `Strategy` can operate in pure replica context."""
  if not distribute_lib.has_strategy():
    return True
  strategy = distribute_lib.get_strategy()
  return not strategy.extended._use_merge_call()  # pylint: disable=protected-access


@tf_export("__internal__.distribute.interim.maybe_merge_call", v1=[])
def maybe_merge_call(fn, strategy, *args, **kwargs):
  """Maybe invoke `fn` via `merge_call` which may or may not be fulfilled.

  The caller of this utility function requests to invoke `fn` via `merge_call`
  at `tf.distribute.Strategy`'s best efforts. It is `tf.distribute`'s internal
  whether the request is honored, depending on the `Strategy`. See
  `tf.distribute.ReplicaContext.merge_call()` for more information.

  This is an interim API which is subject to removal and does not guarantee
  backward-compatibility.

  Args:
    fn: the function to be invoked.
    strategy: the `tf.distribute.Strategy` to call `fn` with.
    *args: the positional arguments to be passed in to `fn`.
    **kwargs: the keyword arguments to be passed in to `fn`.

  Returns:
    The return value of the `fn` call.
  """
  if strategy_supports_no_merge_call():
    return fn(strategy, *args, **kwargs)
  else:
    return distribute_lib.get_replica_context().merge_call(
        fn, args=args, kwargs=kwargs)
