# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Various classes representing distributed values for PS."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import weakref

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import values
from tensorflow.python.distribute import values_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.types import core


# Variable used in PSStrategy TF 1 and CentralStorageStrategy.
class AggregatingVariable(variables_lib.Variable, core.Tensor):
  """A wrapper around a variable that aggregates updates across replicas."""

  def __init__(self, strategy, v, aggregation):
    self._distribute_strategy = strategy
    self._v = v
    # NOTE: We don't use "_distributed_container" here because we don't want
    # to trigger that code path in regroup().
    v._aggregating_container = weakref.ref(self)  # pylint: disable=protected-access
    self._aggregation = aggregation

  def __deepcopy__(self, memo):
    """Perform a deepcopy of the `AggregatingVariable`.

    Unlike the deepcopy of a regular tf.Variable, this keeps the original
    strategy and devices of the `AggregatingVariable`.  To avoid confusion
    with the behavior of deepcopy on a regular `Variable` (which does
    copy into new devices), we only allow a deepcopy of a `AggregatingVariable`
    within its originating strategy scope.

    Args:
      memo: The memoization object for `deepcopy`.

    Returns:
      A deep copy of the current `AggregatingVariable`.

    Raises:
      RuntimeError: If trying to deepcopy into a different strategy.
    """
    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      v = copy.deepcopy(self._v, memo)

    copied_variable = type(self)(
        strategy=self._distribute_strategy,
        v=v,
        aggregation=self._aggregation)

    memo[id(self)] = copied_variable

    return copied_variable

  def get(self):
    return self._v

  @property
  def distribute_strategy(self):
    return self._distribute_strategy

  def __getattr__(self, name):
    return getattr(self._v, name)

  def _assign_func(self, *args, **kwargs):
    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      f = kwargs.pop("f")
      if ds_context.in_cross_replica_context():
        if distribute_lib.get_update_replica_id() is not None:
          # We are calling an assign function in an update context.
          return f(self._v, *args, **kwargs)

        # We are calling an assign function in cross replica context, wrap it in
        # an update call.
        return self._distribute_strategy.extended.update(
            self, f, args=args, kwargs=kwargs)
      else:
        replica_context = ds_context.get_replica_context()
        assert replica_context
        # We are calling an assign function in replica context.
        # We reduce the value we want to assign/add/sub. More details about how
        # we handle the different use cases can be found in the _reduce method.
        # We call the function with the reduced value.
        if self._aggregation == vs.VariableAggregation.NONE:
          raise ValueError(
              values_util.aggregation_error_msg.format(
                  variable_type="AggregatingVariable"))

        def merge_fn(strategy,
                     value,
                     use_locking=False,
                     name=None,
                     read_value=True):
          v = values_util.apply_aggregation(strategy, value, self._aggregation,
                                            self)
          if name and isinstance(name, values.PerReplica):
            name = name.values[0]
          return strategy.extended.update(
              self,
              f,
              args=(v,),
              kwargs={
                  "use_locking": use_locking,
                  "name": name,
                  "read_value": read_value
              })
        return replica_context.merge_call(merge_fn, args=args, kwargs=kwargs)

  def assign_sub(self, *args, **kwargs):
    assign_sub_fn = lambda var, *a, **kw: var.assign_sub(*a, **kw)
    return self._assign_func(f=assign_sub_fn, *args, **kwargs)

  def assign_add(self, *args, **kwargs):
    assign_add_fn = lambda var, *a, **kw: var.assign_add(*a, **kw)
    return self._assign_func(f=assign_add_fn, *args, **kwargs)

  def assign(self, *args, **kwargs):
    assign_fn = lambda var, *a, **kw: var.assign(*a, **kw)
    return self._assign_func(f=assign_fn, *args, **kwargs)

  @property
  def initializer(self):
    return self._v.initializer

  def initialized_value(self):
    return self._v.initialized_value()

  @property
  def initial_value(self):
    return self._v.initial_value

  @property
  def op(self):
    return self._v.op

  def read_value(self):
    return self._v.read_value()

  def eval(self, session=None):
    return self._v.eval(session)

  @property
  def graph(self):
    return self._v.graph

  @property
  def device(self):
    return self._v.device

  @property
  def shape(self):
    return self._v.shape

  @property
  def aggregation(self):
    return self._aggregation

  @property
  def synchronization(self):
    return self._v.synchronization

  @property
  def name(self):
    return self._v.name

  @property
  def trainable(self):
    return self._v.trainable

  @property
  def dtype(self):
    return self._v.dtype

  # TODO(josh11b): Test saving & restoring.
  def _gather_saveables_for_checkpoint(self):
    return {trackable.VARIABLE_VALUE_KEY: self._v}

  def _map_resources(self, save_options):
    """For implementing `Trackable`."""
    # By delegating this method to the wrapped variable, SavedModel with
    # AggregatingVariable are identical to SavedModel with normal variables.
    obj_map, resource_map = self._v._map_resources(save_options)  # pylint:disable=protected-access
    obj_map[self] = obj_map[self._v]
    return obj_map, resource_map

  # pylint: disable=multiple-statements
  def __add__(self, o):
    return self._v + o

  def __radd__(self, o):
    return o + self._v

  def __sub__(self, o):
    return self._v - o

  def __rsub__(self, o):
    return o - self._v

  def __mul__(self, o):
    return self._v * o

  def __rmul__(self, o):
    return o * self._v

  def __truediv__(self, o):
    return self._v / o

  def __rtruediv__(self, o):
    return o / self._v

  def __floordiv__(self, o):
    return self._v // o

  def __rfloordiv__(self, o):
    return o // self._v

  def __mod__(self, o):
    return self._v % o

  def __rmod__(self, o):
    return o % self._v

  def __lt__(self, o):
    return self._v < o

  def __le__(self, o):
    return self._v <= o

  def __gt__(self, o):
    return self._v > o

  def __ge__(self, o):
    return self._v >= o

  def __and__(self, o):
    return self._v & o

  def __rand__(self, o):
    return o & self._v

  def __or__(self, o):
    return self._v | o

  def __ror__(self, o):
    return o | self._v

  def __xor__(self, o):
    return self._v ^ o

  def __rxor__(self, o):
    return o ^ self._v

  def __getitem__(self, o):
    return self._v[o]

  def __pow__(self, o, modulo=None):
    return pow(self._v, o, modulo)

  def __rpow__(self, o):
    return pow(o, self._v)

  def __invert__(self):
    return ~self._v

  def __neg__(self):
    return -self._v

  def __abs__(self):
    return abs(self._v)

  def __div__(self, o):
    try:
      return self._v.__div__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __rdiv__(self, o):
    try:
      return self._v.__rdiv__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __matmul__(self, o):
    try:
      return self._v.__matmul__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __rmatmul__(self, o):
    try:
      return self._v.__rmatmul__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __str__(self):
    return str(self._v)

  def __repr__(self):
    return repr(self._v)

  def _should_act_as_resource_variable(self):
    """Pass resource_variable_ops.is_resource_variable check."""
    pass

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    return ops.convert_to_tensor(self.get(), dtype=dtype, name=name,
                                 as_ref=as_ref)


# Register a conversion function which reads the value of the variable,
# allowing instances of the class to be used as tensors.
def _tensor_conversion_aggregate(var, dtype=None, name=None, as_ref=False):
  return var._dense_var_to_tensor(dtype, name, as_ref)  # pylint: disable=protected-access


ops.register_tensor_conversion_function(AggregatingVariable,
                                        _tensor_conversion_aggregate)
