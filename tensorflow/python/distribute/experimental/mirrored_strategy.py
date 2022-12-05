# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Implement a MirroredStrategy based on the DTensor low level API.

This is an experiment to validate the viability of the DTensor API, and expose
any potential feature gaps between the current API and the need.
"""

from tensorflow.dtensor.python import api as dtensor
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import layout
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import ops


class MirroredStrategy(distribute_lib.Strategy):
  """Synchronous training across multiple replicas on one machine.

  This strategy is typically used for training on one machine with multiple
  accelerators (GPUs/TPUs).

  For example, a variable created under a `MirroredStrategy` is a distributed
  variable with layout replicated on each dimension. The variables will be
  placed on the `mesh` that is specified in the __init__.
  """

  def __init__(self, mesh):
    extended = MirroredExtended(container_strategy=self, mesh=mesh)
    super().__init__(extended)
    self._mesh = mesh


class MirroredExtended(distribute_lib.StrategyExtendedV2):
  """Strategy extension contains the concrete logic for variable creation."""

  def __init__(self, container_strategy, mesh):
    super().__init__(container_strategy)
    self._mesh = mesh

  def _create_variable(self, next_creator, **kwargs):
    # Make sure the pop the `use_resource` which is not supported by the
    # base tf.Variable. The `use_resource` is added by
    # creator_with_resource_vars in distribute_lib.py
    kwargs.pop('use_resource')

    # Make sure to call DVariable initializer under the scope so that it will
    # have the proper replicated layout. The initial_value is multi-typed,
    # eg it can be a tensor, or a python/numpy type, or a callable that
    # produce tensor/python/numpy types. In all those cases, we need to wrap
    # them invoke convert_to_tensor() under the scope so that the proper
    # layout can be assigned.

    # TODO(scottzhu): The layout information should be injected via kwargs, or
    # lazily set later.
    initial_value = kwargs.pop('initial_value')
    def new_initial_value():
      if callable(initial_value):
        init_var = ops.convert_to_tensor(initial_value())
      else:
        init_var = ops.convert_to_tensor(initial_value)
      rank = init_var.shape.rank
      return dtensor.copy_to_mesh(
          init_var, layout.Layout.replicated(self._mesh, rank))

    return d_variable.DVariable(new_initial_value, **kwargs)
