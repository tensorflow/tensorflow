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
"""Core TensorFlow types."""

import sys
import textwrap

from typing import Union

import numpy as np

from tensorflow.python.types import doc_typealias
from tensorflow.python.util.tf_export import tf_export

# pylint:disable=g-import-not-at-top
if sys.version_info >= (3, 8):
  from typing import Protocol
  from typing import runtime_checkable
else:
  from typing_extensions import Protocol
  from typing_extensions import runtime_checkable
# pylint:enable=g-import-not-at-top

# TODO(mdan): Consider adding ABC once the dependence on isinstance is reduced.
# TODO(mdan): Add type annotations.


# TODO(b/178822082): Revisit this API when tf.types gets more resource.
@tf_export("__internal__.types.Tensor", v1=[])
class Tensor(object):
  """The base class of all dense Tensor objects.

  A dense tensor has a static data type (dtype), and may have a static rank and
  shape. Tensor objects are immutable. Mutable objects may be backed by a Tensor
  which holds the unique handle that identifies the mutable object.
  """

  @property
  def dtype(self):
    pass

  @property
  def shape(self):
    pass


# `ops.EagerTensor` subclasses `Symbol` by way of subclassing `ops.Tensor`;
# care should be taken when performing `isinstance` checks on `Value`, e.g.:
#
# ```
# if isinstance(core.Symbol) and not isinstance(core.Value):
#   ...
# ```
class Symbol(Tensor):
  """Symbolic "graph" Tensor.

  These objects represent the output of an op definition and do not carry a
  value.
  """
  pass


class Value(Tensor):
  """Tensor that can be associated with a value (aka "eager tensor").

  These objects represent the (usually future) output of executing an op
  immediately.
  """

  def numpy(self):
    pass


@tf_export("types.experimental.Callable", v1=[])
class Callable:
  """Base class for TF callables like those created by tf.function.

  Note: Callables are conceptually very similar to `tf.Operation`: a
  `tf.Operation` is a kind of callable.
  """

  def __call__(self, *args, **kwargs):
    """Executes this callable.

    This behaves like a regular op - in eager mode, it immediately starts
    execution, returning results. In graph mode, it creates ops which return
    symbolic TensorFlow values (like `tf.Tensor`, `tf.data.Dataset`,
    etc.). For example, `tf.function` callables typically generate a
    `tf.raw_ops.PartitionedCall` op, but not always - the
    exact operations being generated are an internal implementation detail.

    Args:
      *args: positional argument for this call
      **kwargs: keyword arguments for this call
    Returns:
      The execution results.
    """


@tf_export("types.experimental.ConcreteFunction", v1=[])
class ConcreteFunction(Callable):
  """Base class for graph functions.

  A `ConcreteFunction` encapsulates a single graph function definition and
  is differentiable under `tf.GradientTape` contexts.
  """


# TODO(mdan): Name just `types.Function`, for historic continuity?
@tf_export("types.experimental.GenericFunction", v1=[])
class GenericFunction(Callable):
  """Base class for polymorphic graph functions.

  Graph functions are Python callable objects that dispatch calls to a
  TensorFlow graph. Polymorphic graph functions can be backed by multiple TF
  graphs, and automatically select the appropriate specialization based on the
  type of input they were called with. They may also create specializations on
  the fly if necessary, for example by tracing.

  Also see `tf.function`.
  """

  def get_concrete_function(self, *args, **kwargs) -> ConcreteFunction:
    """Returns a `ConcreteFunction` specialized to input types.

    The arguments specified by `args` and `kwargs` follow normal function call
    rules. The returned `ConcreteFunction` has the same set of positional and
    keyword arguments as `self`, but their types are compatible to the types
    specified by `args` and `kwargs` (though not neccessarily equal).

    >>> @tf.function
    ... def f(x):
    ...   return x
    >>> f_concrete = f.get_concrete_function(tf.constant(1.0))
    >>> f_concrete = f.get_concrete_function(x=tf.constant(1.0))

    Unlike normal calls, `get_concrete_function` allow type specifiers instead
    of TensorFlow objects, so for example `tf.Tensor`s may be replaced with
    `tf.TensorSpec`s.

    >>> @tf.function
    ... def f(x):
    ...   return x
    >>> f_concrete = f.get_concrete_function(tf.TensorSpec([], tf.float64))

    If the function definition allows only one specialization, `args` and
    `kwargs` may be omitted altogether.

    >>> @tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
    ... def f(x):
    ...   return x
    >>> f_concrete = f.get_concrete_function()

    The returned `ConcreteFunction` can be called normally:

    >>> f_concrete(tf.constant(1.0))
    <tf.Tensor: shape=(), dtype=float32, numpy=1.0>
    >>> f_concrete(x=tf.constant(1.0))
    <tf.Tensor: shape=(), dtype=float32, numpy=1.0>

    Args:
      *args: inputs to specialize on.
      **kwargs: inputs to specialize on.

    Returns:
      A `ConcreteFunction`.
    """
    pass

  def experimental_get_compiler_ir(self, *args, **kwargs):
    """Returns compiler IR for the compiled function.

    This API is intended *only* for debugging as there are no guarantees on
    backwards compatibility of returned IR or the allowed values of `stage`.

    Args:
      *args: compilation args supports inputs either: (1) all inputs are
        TensorSpec or (2) all inputs are tf.Tensor/Python variables.
      **kwargs: Keyword arguments used for compilation. Same requirement as
        compiliation args.

    Returns:
      Function callable with the following kwargs:
        - `stage` at which the compiler IR should be serialized. Allowed values
          are:
           - `hlo`: HLO output after conversion from TF
            (https://www.tensorflow.org/xla/operation_semantics).
           - `hlo_serialized`: Like stage=`hlo`, but the output is a serialized
             HLO module proto (a bytes object).
           - `optimized_hlo`: HLO after compiler optimizations.
           - `optimized_hlo_serialized`: Like stage=`optimized_hlo`, but the
             output is a serialized HLO module proto (a bytes object).
           - `optimized_hlo_dot`: optimized HLO in DOT format suitable for
             Graphviz.
        - `device_name` can be either None, in which case the preferred device
          is used for compilation, or a device name. It can be a full device
          name, or a partial one, e.g., `/device:CPU:0`.

      For example, for

      ```python
      @tf.function(jit_compile=True)
      def f(x):
        return x + 1

      f.experimental_get_compiler_ir(tf.random.normal([10, 10])(stage='hlo')
      ```

      the output is:

      ```
      HloModule a_inference_f_13__.9

      ENTRY %a_inference_f_13__.9 (arg0.1: f32[10,10]) -> f32[10,10] {
        %arg0.1 = f32[10,10]{1,0} parameter(0), parameter_replication={false}
        %reshape.2 = f32[10,10]{1,0} reshape(f32[10,10]{1,0} %arg0.1)
        %constant.3 = f32[] constant(1)
        %broadcast.4 = f32[10,10]{1,0} broadcast(f32[] %constant.3)
        %add.5 = f32[10,10]{1,0} add(f32[10,10]{1,0} %reshape.2,
                                     f32[10,10]{1,0} %broadcast.4)
        %reshape.6 = f32[10,10]{1,0} reshape(f32[10,10]{1,0} %add.5)
        %tuple.7 = (f32[10,10]{1,0}) tuple(f32[10,10]{1,0} %reshape.6)
        ROOT %get-tuple-element.8 = f32[10,10]{1,0}
          get-tuple-element((f32[10,10]{1,0}) %tuple.7), index=0
      }
      ```

    Raises:
      ValueError:
        (1) If an invalid `stage` is selected
        (2) or if applied to a function which is not compiled
        (`jit_compile=True` is not set).
        (3) or if input shapes are not fully defined for tf.TensorSpec inputs
      TypeError: When called with input in graph mode.
    """
    pass


@runtime_checkable
class TensorProtocol(Protocol):
  """Protocol type for objects that can be converted to Tensor."""

  def __tf_tensor__(self, dtype=None, name=None):
    """Converts this object to a Tensor.

    Args:
      dtype: data type for the returned Tensor
      name: a name for the operations which create the Tensor
    Returns:
      A Tensor.
    """
    pass


# TODO(rahulkamat): Add missing types that are convertible to Tensor.
TensorLike = Union[Tensor, TensorProtocol, int, float, bool, str, bytes,
                   complex, tuple, list, np.ndarray, np.generic]
doc_typealias.document(
    obj=TensorLike,
    doc=textwrap.dedent("""\
      Union of all types that can be converted to a `tf.Tensor` by `tf.convert_to_tensor`.

      This definition may be used in user code. Additional types may be added
      in the future as more input types are supported.

      Example:

      ```
      def foo(x: TensorLike):
        pass
      ```

      This definition passes static type verification for:

      ```
      foo(tf.constant([1, 2, 3]))
      foo([1, 2, 3])
      foo(np.array([1, 2, 3]))
      ```
      """),
)
tf_export("types.experimental.TensorLike").export_constant(
    __name__, "TensorLike")
