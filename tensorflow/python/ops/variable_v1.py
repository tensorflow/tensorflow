# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""VariableV1 class."""

from tensorflow.python.framework import ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.util import tf_should_use
from tensorflow.python.util.tf_export import tf_export


_variable_from_proto_fn = None


def set_variable_from_proto_fn(variable_from_proto_fn):
  """Set the variable class that variable proto defs will be converted to."""
  global _variable_from_proto_fn
  _variable_from_proto_fn = variable_from_proto_fn


@tf_export(v1=["is_variable_initialized"])
@tf_should_use.should_use_result
def is_variable_initialized(variable):
  """Tests if a variable has been initialized.

  Args:
    variable: A `Variable`.

  Returns:
    Returns a scalar boolean Tensor, `True` if the variable has been
    initialized, `False` otherwise.
  """
  return state_ops.is_variable_initialized(variable)


@tf_export(v1=["Variable"])
class VariableV1(variables.Variable):
  """See the [Variables Guide](https://tensorflow.org/guide/variables).

  A variable maintains state in the graph across calls to `run()`. You add a
  variable to the graph by constructing an instance of the class `Variable`.

  The `Variable()` constructor requires an initial value for the variable,
  which can be a `Tensor` of any type and shape. The initial value defines the
  type and shape of the variable. After construction, the type and shape of
  the variable are fixed. The value can be changed using one of the assign
  methods.

  If you want to change the shape of a variable later you have to use an
  `assign` Op with `validate_shape=False`.

  Just like any `Tensor`, variables created with `Variable()` can be used as
  inputs for other Ops in the graph. Additionally, all the operators
  overloaded for the `Tensor` class are carried over to variables, so you can
  also add nodes to the graph by just doing arithmetic on variables.

  ```python
  import tensorflow as tf

  # Create a variable.
  w = tf.Variable(<initial-value>, name=<optional-name>)

  # Use the variable in the graph like any Tensor.
  y = tf.matmul(w, ...another variable or tensor...)

  # The overloaded operators are available too.
  z = tf.sigmoid(w + y)

  # Assign a new value to the variable with `assign()` or a related method.
  w.assign(w + 1.0)
  w.assign_add(1.0)
  ```

  When you launch the graph, variables have to be explicitly initialized before
  you can run Ops that use their value. You can initialize a variable by
  running its *initializer op*, restoring the variable from a save file, or
  simply running an `assign` Op that assigns a value to the variable. In fact,
  the variable *initializer op* is just an `assign` Op that assigns the
  variable's initial value to the variable itself.

  ```python
  # Launch the graph in a session.
  with tf.compat.v1.Session() as sess:
      # Run the variable initializer.
      sess.run(w.initializer)
      # ...you now can run ops that use the value of 'w'...
  ```

  The most common initialization pattern is to use the convenience function
  `global_variables_initializer()` to add an Op to the graph that initializes
  all the variables. You then run that Op after launching the graph.

  ```python
  # Add an Op to initialize global variables.
  init_op = tf.compat.v1.global_variables_initializer()

  # Launch the graph in a session.
  with tf.compat.v1.Session() as sess:
      # Run the Op that initializes global variables.
      sess.run(init_op)
      # ...you can now run any Op that uses variable values...
  ```

  If you need to create a variable with an initial value dependent on another
  variable, use the other variable's `initialized_value()`. This ensures that
  variables are initialized in the right order.

  All variables are automatically collected in the graph where they are
  created. By default, the constructor adds the new variable to the graph
  collection `GraphKeys.GLOBAL_VARIABLES`. The convenience function
  `global_variables()` returns the contents of that collection.

  When building a machine learning model it is often convenient to distinguish
  between variables holding the trainable model parameters and other variables
  such as a `global step` variable used to count training steps. To make this
  easier, the variable constructor supports a `trainable=<bool>` parameter. If
  `True`, the new variable is also added to the graph collection
  `GraphKeys.TRAINABLE_VARIABLES`. The convenience function
  `trainable_variables()` returns the contents of this collection. The
  various `Optimizer` classes use this collection as the default list of
  variables to optimize.

  WARNING: tf.Variable objects by default have a non-intuitive memory model. A
  Variable is represented internally as a mutable Tensor which can
  non-deterministically alias other Tensors in a graph. The set of operations
  which consume a Variable and can lead to aliasing is undetermined and can
  change across TensorFlow versions. Avoid writing code which relies on the
  value of a Variable either changing or not changing as other operations
  happen. For example, using Variable objects or simple functions thereof as
  predicates in a `tf.cond` is dangerous and error-prone:

  ```
  v = tf.Variable(True)
  tf.cond(v, lambda: v.assign(False), my_false_fn)  # Note: this is broken.
  ```

  Here, adding `use_resource=True` when constructing the variable will
  fix any nondeterminism issues:
  ```
  v = tf.Variable(True, use_resource=True)
  tf.cond(v, lambda: v.assign(False), my_false_fn)
  ```

  To use the replacement for variables which does
  not have these issues:

  * Add `use_resource=True` when constructing `tf.Variable`;
  * Call `tf.compat.v1.get_variable_scope().set_use_resource(True)` inside a
    `tf.compat.v1.variable_scope` before the `tf.compat.v1.get_variable()` call.
  """

  def __init__(
      self,  # pylint: disable=super-init-not-called
      initial_value=None,
      trainable=None,
      collections=None,
      validate_shape=True,
      caching_device=None,
      name=None,
      variable_def=None,
      dtype=None,
      expected_shape=None,
      import_scope=None,
      constraint=None,
      use_resource=None,
      synchronization=variables.VariableSynchronization.AUTO,
      aggregation=variables.VariableAggregation.NONE,
      shape=None):
    """Creates a new variable with value `initial_value`.

    The new variable is added to the graph collections listed in `collections`,
    which defaults to `[GraphKeys.GLOBAL_VARIABLES]`.

    If `trainable` is `True` the variable is also added to the graph collection
    `GraphKeys.TRAINABLE_VARIABLES`.

    This constructor creates both a `variable` Op and an `assign` Op to set the
    variable to its initial value.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called. In
        that case, `dtype` must be specified. (Note that initializer functions
        from init_ops.py must first be bound to a shape before being used here.)
      trainable: If `True`, also adds the variable to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as the default
        list of variables to use by the `Optimizer` classes. Defaults to `True`,
        unless `synchronization` is set to `ON_READ`, in which case it defaults
        to `False`.
      collections: List of graph collections keys. The new variable is added to
        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
      validate_shape: If `False`, allows the variable to be initialized with a
        value of unknown shape. If `True`, the default, the shape of
        `initial_value` must be known.
      caching_device: Optional device string describing where the Variable
        should be cached for reading.  Defaults to the Variable's device. If not
        `None`, caches on another device.  Typical use is to cache on the device
        where the Ops using the Variable reside, to deduplicate copying through
        `Switch` and other conditional statements.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      variable_def: `VariableDef` protocol buffer. If not `None`, recreates the
        Variable object with its contents, referencing the variable's nodes in
        the graph, which must already exist. The graph is not changed.
        `variable_def` and the other arguments are mutually exclusive.
      dtype: If set, initial_value will be converted to the given type. If
        `None`, either the datatype will be kept (if `initial_value` is a
        Tensor), or `convert_to_tensor` will decide.
      expected_shape: A TensorShape. If set, initial_value is expected to have
        this shape.
      import_scope: Optional `string`. Name scope to add to the `Variable.` Only
        used when initializing from protocol buffer.
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      use_resource: whether to use resource variables.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses when to
        synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.
      shape: (optional) The shape of this variable. If None, the shape of
        `initial_value` will be used. When setting this argument to
        `tf.TensorShape(None)` (representing an unspecified shape), the variable
        can be assigned with values of different shapes.

    Raises:
      ValueError: If both `variable_def` and initial_value are specified.
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.
      RuntimeError: If eager execution is enabled.
    """

  SaveSliceInfo = variables.Variable.SaveSliceInfo

  def initialized_value(self):
    with ops.init_scope():
      return cond.cond(
          is_variable_initialized(self), self.read_value,
          lambda: self.initial_value)

  @staticmethod
  def from_proto(variable_def, import_scope=None):
    return _variable_from_proto_fn(
        variable_def=variable_def, import_scope=import_scope)

  @classmethod
  def _variable_call(
      cls,
      initial_value=None,
      trainable=None,
      validate_shape=True,
      caching_device=None,
      name=None,
      variable_def=None,
      dtype=None,
      import_scope=None,
      constraint=None,
      synchronization=variables.VariableSynchronization.AUTO,
      aggregation=variables.VariableAggregation.NONE,
      shape=None,
      experimental_enable_variable_lifting=None,
      expected_shape=None,
      collections=None,
      use_resource=None,
      **kwargs,
    ):
    """VariableV1 class getter. Useful to force the signature."""
    if cls is not VariableV1:
      return None
    previous_getter = lambda **kwargs: variables.default_variable_creator(
        None, **kwargs)
    for _, getter in ops.get_default_graph()._variable_creator_stack:  # pylint: disable=protected-access
      previous_getter = variables._make_getter(getter, previous_getter)  # pylint: disable=protected-access

    # Reset `aggregation` that is explicitly set as `None` to the enum NONE.
    if aggregation is None:
      aggregation = variables.VariableAggregation.NONE
    return previous_getter(
        initial_value=initial_value,
        trainable=trainable,
        validate_shape=validate_shape,
        caching_device=caching_device,
        name=name,
        variable_def=variable_def,
        dtype=dtype,
        import_scope=import_scope,
        constraint=constraint,
        synchronization=synchronization,
        aggregation=aggregation,
        shape=shape,
        experimental_enable_variable_lifting=experimental_enable_variable_lifting,
        expected_shape=expected_shape,
        collections=collections,
        use_resource=use_resource,
    )
