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
"""Utility to get tf.distribute.Strategy related contexts."""

from tensorflow.python.util.lazy_loader import LazyLoader


# There is a circular dependency between this and the `distribute_lib` module.
# So we load it lazily to work around this.
distribute_lib = LazyLoader(
    "distribute_lib", globals(),
    "tensorflow.python.distribute.distribute_lib")


# Refactor currently in progress to combine this file with distribute_lib.
# The below references must be left until all references have been migrated.
_ThreadMode = distribute_lib._ThreadMode  # pylint: disable=protected-access
_CrossReplicaThreadMode = distribute_lib._CrossReplicaThreadMode  # pylint: disable=protected-access
_InReplicaThreadMode = distribute_lib._InReplicaThreadMode  # pylint: disable=protected-access
_DefaultReplicaThreadMode = distribute_lib._DefaultReplicaThreadMode  # pylint: disable=protected-access
_get_per_thread_mode = distribute_lib._get_per_thread_mode  # pylint: disable=protected-access
_variable_sync_on_read_context = distribute_lib._variable_sync_on_read_context  # pylint: disable=protected-access
variable_sync_on_read_context = distribute_lib.variable_sync_on_read_context
_push_per_thread_mode = distribute_lib._push_per_thread_mode  # pylint: disable=protected-access
_pop_per_thread_mode = distribute_lib._pop_per_thread_mode  # pylint: disable=protected-access
in_variable_sync_on_read_context = distribute_lib.in_variable_sync_on_read_context  # pylint: disable=protected-access
get_replica_context = distribute_lib.get_replica_context
get_cross_replica_context = distribute_lib.get_cross_replica_context
in_cross_replica_context = distribute_lib.in_cross_replica_context
get_strategy = distribute_lib.get_strategy
has_strategy = distribute_lib.has_strategy
get_strategy_and_replica_context = distribute_lib.get_strategy_and_replica_context  # pylint: disable=protected-access
experimental_set_strategy = distribute_lib.experimental_set_strategy
enter_or_assert_strategy = distribute_lib.enter_or_assert_strategy
_defaults = distribute_lib._defaults  # pylint: disable=protected-access
_default_strategy_lock = distribute_lib._default_strategy_lock  # pylint: disable=protected-access
_default_replica_context_lock = distribute_lib._default_replica_context_lock  # pylint: disable=protected-access
_default_replica_mode_lock = distribute_lib._default_replica_mode_lock  # pylint: disable=protected-access
_assert_strategy = distribute_lib._assert_strategy  # pylint: disable=protected-access
_get_default_strategy = distribute_lib._get_default_strategy  # pylint: disable=protected-access
_get_default_replica_context = distribute_lib._get_default_replica_context  # pylint: disable=protected-access
_get_default_replica_mode = distribute_lib._get_default_replica_mode  # pylint: disable=protected-access
get_distribution_strategy = distribute_lib.get_distribution_strategy
has_distribution_strategy = distribute_lib.has_distribution_strategy
