# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Utility functions for training."""
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import cond
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export

# Picked a long key value to minimize the chance of collision with user defined
# collection keys.
GLOBAL_STEP_READ_KEY = 'global_step_read_op_cache'

# TODO(drpng): remove this after legacy uses are resolved.
write_graph = graph_io.write_graph


@tf_export(v1=['train.global_step'])
def global_step(sess, global_step_tensor):
  """Small helper to get the global step.

  ```python
  # Create a variable to hold the global_step.
  global_step_tensor = tf.Variable(10, trainable=False, name='global_step')
  # Create a session.
  sess = tf.compat.v1.Session()
  # Initialize the variable
  sess.run(global_step_tensor.initializer)
  # Get the variable value.
  print('global_step: %s' % tf.compat.v1.train.global_step(sess,
  global_step_tensor))

  global_step: 10
  ```

  Args:
    sess: A TensorFlow `Session` object.
    global_step_tensor:  `Tensor` or the `name` of the operation that contains
      the global step.

  Returns:
    The global step value.
  """
  if context.executing_eagerly():
    return int(global_step_tensor.numpy())
  return int(sess.run(global_step_tensor))


@tf_export(v1=['train.get_global_step'])
def get_global_step(graph=None):
  """Get the global step tensor.

  The global step tensor must be an integer variable. We first try to find it
  in the collection `GLOBAL_STEP`, or by name `global_step:0`.

  Args:
    graph: The graph to find the global step in. If missing, use default graph.

  Returns:
    The global step variable, or `None` if none was found.

  Raises:
    TypeError: If the global step tensor has a non-integer type, or if it is not
      a `Variable`.

  @compatibility(TF2)
  With the deprecation of global graphs, TF no longer tracks variables in
  collections. In other words, there are no global variables in TF2. Thus, the
  global step functions have been removed  (`get_or_create_global_step`,
  `create_global_step`, `get_global_step`) . You have two options for migrating:

  1. Create a Keras optimizer, which generates an `iterations` variable. This
     variable is automatically incremented when calling `apply_gradients`.
  2. Manually create and increment a `tf.Variable`.

  Below is an example of migrating away from using a global step to using a
  Keras optimizer:

  Define a dummy model and loss:

  >>> def compute_loss(x):
  ...   v = tf.Variable(3.0)
  ...   y = x * v
  ...   loss = x * 5 - x * v
  ...   return loss, [v]

  Before migrating:

  >>> g = tf.Graph()
  >>> with g.as_default():
  ...   x = tf.compat.v1.placeholder(tf.float32, [])
  ...   loss, var_list = compute_loss(x)
  ...   global_step = tf.compat.v1.train.get_or_create_global_step()
  ...   global_init = tf.compat.v1.global_variables_initializer()
  ...   optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
  ...   train_op = optimizer.minimize(loss, global_step, var_list)
  >>> sess = tf.compat.v1.Session(graph=g)
  >>> sess.run(global_init)
  >>> print("before training:", sess.run(global_step))
  before training: 0
  >>> sess.run(train_op, feed_dict={x: 3})
  >>> print("after training:", sess.run(global_step))
  after training: 1

  Using `get_global_step`:

  >>> with g.as_default():
  ...   print(sess.run(tf.compat.v1.train.get_global_step()))
  1

  Migrating to a Keras optimizer:

  >>> optimizer = tf.keras.optimizers.SGD(.01)
  >>> print("before training:", optimizer.iterations.numpy())
  before training: 0
  >>> with tf.GradientTape() as tape:
  ...   loss, var_list = compute_loss(3)
  ...   grads = tape.gradient(loss, var_list)
  ...   optimizer.apply_gradients(zip(grads, var_list))
  >>> print("after training:", optimizer.iterations.numpy())
  after training: 1

  @end_compatibility
  """
  graph = graph or ops.get_default_graph()
  global_step_tensor = None
  global_step_tensors = graph.get_collection(ops.GraphKeys.GLOBAL_STEP)
  if len(global_step_tensors) == 1:
    global_step_tensor = global_step_tensors[0]
  elif not global_step_tensors:
    try:
      global_step_tensor = graph.get_tensor_by_name('global_step:0')
    except KeyError:
      return None
  else:
    logging.error('Multiple tensors in global_step collection.')
    return None

  assert_global_step(global_step_tensor)
  return global_step_tensor


@tf_export(v1=['train.create_global_step'])
def create_global_step(graph=None):
  """Create global step tensor in graph.

  Args:
    graph: The graph in which to create the global step tensor. If missing, use
      default graph.

  Returns:
    Global step tensor.

  Raises:
    ValueError: if global step tensor is already defined.

  @compatibility(TF2)
  With the deprecation of global graphs, TF no longer tracks variables in
  collections. In other words, there are no global variables in TF2. Thus, the
  global step functions have been removed  (`get_or_create_global_step`,
  `create_global_step`, `get_global_step`) . You have two options for migrating:

  1. Create a Keras optimizer, which generates an `iterations` variable. This
     variable is automatically incremented when calling `apply_gradients`.
  2. Manually create and increment a `tf.Variable`.

  Below is an example of migrating away from using a global step to using a
  Keras optimizer:

  Define a dummy model and loss:

  >>> def compute_loss(x):
  ...   v = tf.Variable(3.0)
  ...   y = x * v
  ...   loss = x * 5 - x * v
  ...   return loss, [v]

  Before migrating:

  >>> g = tf.Graph()
  >>> with g.as_default():
  ...   x = tf.compat.v1.placeholder(tf.float32, [])
  ...   loss, var_list = compute_loss(x)
  ...   global_step = tf.compat.v1.train.create_global_step()
  ...   global_init = tf.compat.v1.global_variables_initializer()
  ...   optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
  ...   train_op = optimizer.minimize(loss, global_step, var_list)
  >>> sess = tf.compat.v1.Session(graph=g)
  >>> sess.run(global_init)
  >>> print("before training:", sess.run(global_step))
  before training: 0
  >>> sess.run(train_op, feed_dict={x: 3})
  >>> print("after training:", sess.run(global_step))
  after training: 1

  Migrating to a Keras optimizer:

  >>> optimizer = tf.keras.optimizers.SGD(.01)
  >>> print("before training:", optimizer.iterations.numpy())
  before training: 0
  >>> with tf.GradientTape() as tape:
  ...   loss, var_list = compute_loss(3)
  ...   grads = tape.gradient(loss, var_list)
  ...   optimizer.apply_gradients(zip(grads, var_list))
  >>> print("after training:", optimizer.iterations.numpy())
  after training: 1

  @end_compatibility
  """
  graph = graph or ops.get_default_graph()
  if get_global_step(graph) is not None:
    raise ValueError('"global_step" already exists.')
  if context.executing_eagerly():
    with ops.device('cpu:0'):
      return variable_scope.get_variable(
          ops.GraphKeys.GLOBAL_STEP,
          shape=[],
          dtype=dtypes.int64,
          initializer=init_ops.zeros_initializer(),
          trainable=False,
          aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA,
          collections=[
              ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.GLOBAL_STEP
          ])
  # Create in proper graph and base name_scope.
  with graph.as_default() as g, g.name_scope(None):
    return variable_scope.get_variable(
        ops.GraphKeys.GLOBAL_STEP,
        shape=[],
        dtype=dtypes.int64,
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.GLOBAL_STEP])


@tf_export(v1=['train.get_or_create_global_step'])
def get_or_create_global_step(graph=None):
  """Returns and create (if necessary) the global step tensor.

  Args:
    graph: The graph in which to create the global step tensor. If missing, use
      default graph.

  Returns:
    The global step tensor.

  @compatibility(TF2)
  With the deprecation of global graphs, TF no longer tracks variables in
  collections. In other words, there are no global variables in TF2. Thus, the
  global step functions have been removed  (`get_or_create_global_step`,
  `create_global_step`, `get_global_step`) . You have two options for migrating:

  1. Create a Keras optimizer, which generates an `iterations` variable. This
     variable is automatically incremented when calling `apply_gradients`.
  2. Manually create and increment a `tf.Variable`.

  Below is an example of migrating away from using a global step to using a
  Keras optimizer:

  Define a dummy model and loss:

  >>> def compute_loss(x):
  ...   v = tf.Variable(3.0)
  ...   y = x * v
  ...   loss = x * 5 - x * v
  ...   return loss, [v]

  Before migrating:

  >>> g = tf.Graph()
  >>> with g.as_default():
  ...   x = tf.compat.v1.placeholder(tf.float32, [])
  ...   loss, var_list = compute_loss(x)
  ...   global_step = tf.compat.v1.train.get_or_create_global_step()
  ...   global_init = tf.compat.v1.global_variables_initializer()
  ...   optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
  ...   train_op = optimizer.minimize(loss, global_step, var_list)
  >>> sess = tf.compat.v1.Session(graph=g)
  >>> sess.run(global_init)
  >>> print("before training:", sess.run(global_step))
  before training: 0
  >>> sess.run(train_op, feed_dict={x: 3})
  >>> print("after training:", sess.run(global_step))
  after training: 1

  Migrating to a Keras optimizer:

  >>> optimizer = tf.keras.optimizers.SGD(.01)
  >>> print("before training:", optimizer.iterations.numpy())
  before training: 0
  >>> with tf.GradientTape() as tape:
  ...   loss, var_list = compute_loss(3)
  ...   grads = tape.gradient(loss, var_list)
  ...   optimizer.apply_gradients(zip(grads, var_list))
  >>> print("after training:", optimizer.iterations.numpy())
  after training: 1

  @end_compatibility
  """
  graph = graph or ops.get_default_graph()
  global_step_tensor = get_global_step(graph)
  if global_step_tensor is None:
    global_step_tensor = create_global_step(graph)
  return global_step_tensor


@tf_export(v1=['train.assert_global_step'])
def assert_global_step(global_step_tensor):
  """Asserts `global_step_tensor` is a scalar int `Variable` or `Tensor`.

  Args:
    global_step_tensor: `Tensor` to test.
  """
  if not (isinstance(global_step_tensor, variables.Variable) or
          isinstance(global_step_tensor, tensor.Tensor) or
          resource_variable_ops.is_resource_variable(global_step_tensor)):
    raise TypeError('Existing "global_step" must be a Variable or Tensor: %s.' %
                    global_step_tensor)

  if not global_step_tensor.dtype.base_dtype.is_integer:
    raise TypeError('Existing "global_step" does not have integer type: %s' %
                    global_step_tensor.dtype)

  if (global_step_tensor.get_shape().ndims != 0 and
      global_step_tensor.get_shape().is_fully_defined()):
    raise TypeError('Existing "global_step" is not scalar: %s' %
                    global_step_tensor.get_shape())


def _get_global_step_read(graph=None):
  """Gets global step read tensor in graph.

  Args:
    graph: The graph in which to create the global step read tensor. If missing,
      use default graph.

  Returns:
    Global step read tensor.

  Raises:
    RuntimeError: if multiple items found in collection GLOBAL_STEP_READ_KEY.
  """
  graph = graph or ops.get_default_graph()
  global_step_read_tensors = graph.get_collection(GLOBAL_STEP_READ_KEY)
  if len(global_step_read_tensors) > 1:
    raise RuntimeError('There are multiple items in collection {}. '
                       'There should be only one.'.format(GLOBAL_STEP_READ_KEY))

  if len(global_step_read_tensors) == 1:
    return global_step_read_tensors[0]
  return None


def _get_or_create_global_step_read(graph=None):
  """Gets or creates global step read tensor in graph.

  Args:
    graph: The graph in which to create the global step read tensor. If missing,
      use default graph.

  Returns:
    Global step read tensor if there is global_step_tensor else return None.
  """
  graph = graph or ops.get_default_graph()
  global_step_read_tensor = _get_global_step_read(graph)
  if global_step_read_tensor is not None:
    return global_step_read_tensor
  global_step_tensor = get_global_step(graph)
  if global_step_tensor is None:
    return None
  # add 'zero' so that it will create a copy of variable as Tensor.
  with graph.as_default() as g, g.name_scope(None):
    with g.name_scope(global_step_tensor.op.name + '/'):
      # must ensure that global_step is initialized before
      # this run. This is needed for example Estimator makes all model_fn build
      # under global_step_read_tensor dependency.
      if isinstance(global_step_tensor, variables.Variable):
        global_step_value = cond.cond(
            variable_v1.is_variable_initialized(global_step_tensor),
            global_step_tensor.read_value,
            lambda: global_step_tensor.initial_value)
      else:
        global_step_value = global_step_tensor

      global_step_read_tensor = global_step_value + 0
      ops.add_to_collection(GLOBAL_STEP_READ_KEY, global_step_read_tensor)
  return _get_global_step_read(graph)


def _increment_global_step(increment, graph=None):
  graph = graph or ops.get_default_graph()
  global_step_tensor = get_global_step(graph)
  if global_step_tensor is None:
    raise ValueError(
        'Global step tensor should be created by '
        'tf.train.get_or_create_global_step before calling increment.')
  global_step_read_tensor = _get_or_create_global_step_read(graph)
  with graph.as_default() as g, g.name_scope(None):
    with g.name_scope(global_step_tensor.op.name + '/'):
      with ops.control_dependencies([global_step_read_tensor]):
        return state_ops.assign_add(global_step_tensor, increment)
