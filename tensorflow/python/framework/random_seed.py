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

"""For seeding individual ops based on a graph-level seed.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


DEFAULT_GRAPH_SEED = 87654321
_MAXINT32 = 2**31 - 1


def _truncate_seed(seed):
  return seed % _MAXINT32  # Truncate to fit into 32-bit integer


@tf_export(v1=['random.get_seed', 'get_seed'])
@deprecation.deprecated_endpoints('get_seed')
def get_seed(op_seed):
  """Returns the local seeds an operation should use given an op-specific seed.

  Given operation-specific seed, `op_seed`, this helper function returns two
  seeds derived from graph-level and op-level seeds. Many random operations
  internally use the two seeds to allow user to change the seed globally for a
  graph, or for only specific operations.

  For details on how the graph-level seed interacts with op seeds, see
  `tf.compat.v1.random.set_random_seed`.

  Args:
    op_seed: integer.

  Returns:
    A tuple of two integers that should be used for the local seed of this
    operation.
  """
  eager = context.executing_eagerly()

  if eager:
    global_seed = context.global_seed()
  else:
    global_seed = ops.get_default_graph().seed

  if global_seed is not None:
    if op_seed is None:
      # pylint: disable=protected-access
      if hasattr(ops.get_default_graph(), '_seed_used'):
        ops.get_default_graph()._seed_used = True
      if eager:
        op_seed = context.internal_operation_seed()
      else:
        op_seed = ops.get_default_graph()._last_id

    seeds = _truncate_seed(global_seed), _truncate_seed(op_seed)
  else:
    if op_seed is not None:
      seeds = DEFAULT_GRAPH_SEED, _truncate_seed(op_seed)
    else:
      seeds = None, None
  # Avoid (0, 0) as the C++ ops interpret it as nondeterminism, which would
  # be unexpected since Python docs say nondeterminism is (None, None).
  if seeds == (0, 0):
    return (0, _MAXINT32)
  return seeds


@tf_export(v1=['random.set_random_seed', 'set_random_seed'])
def set_random_seed(seed):
  """Sets the graph-level random seed for the default graph.

  Operations that rely on a random seed actually derive it from two seeds:
  the graph-level and operation-level seeds. This sets the graph-level seed.

  Its interactions with operation-level seeds is as follows:

    1. If neither the graph-level nor the operation seed is set:
      A random seed is used for this op.
    2. If the graph-level seed is set, but the operation seed is not:
      The system deterministically picks an operation seed in conjunction with
      the graph-level seed so that it gets a unique random sequence. Within the
      same version of tensorflow and user code, this sequence is deterministic.
      However across different versions, this sequence might change. If the
      code depends on particular seeds to work, specify both graph-level
      and operation-level seeds explicitly.
    3. If the graph-level seed is not set, but the operation seed is set:
      A default graph-level seed and the specified operation seed are used to
      determine the random sequence.
    4. If both the graph-level and the operation seed are set:
      Both seeds are used in conjunction to determine the random sequence.

  To illustrate the user-visible effects, consider these examples:

  To generate different sequences across sessions, set neither
  graph-level nor op-level seeds:

  ```python
  a = tf.random.uniform([1])
  b = tf.random.normal([1])

  print("Session 1")
  with tf.compat.v1.Session() as sess1:
    print(sess1.run(a))  # generates 'A1'
    print(sess1.run(a))  # generates 'A2'
    print(sess1.run(b))  # generates 'B1'
    print(sess1.run(b))  # generates 'B2'

  print("Session 2")
  with tf.compat.v1.Session() as sess2:
    print(sess2.run(a))  # generates 'A3'
    print(sess2.run(a))  # generates 'A4'
    print(sess2.run(b))  # generates 'B3'
    print(sess2.run(b))  # generates 'B4'
  ```

  To generate the same repeatable sequence for an op across sessions, set the
  seed for the op:

  ```python
  a = tf.random.uniform([1], seed=1)
  b = tf.random.normal([1])

  # Repeatedly running this block with the same graph will generate the same
  # sequence of values for 'a', but different sequences of values for 'b'.
  print("Session 1")
  with tf.compat.v1.Session() as sess1:
    print(sess1.run(a))  # generates 'A1'
    print(sess1.run(a))  # generates 'A2'
    print(sess1.run(b))  # generates 'B1'
    print(sess1.run(b))  # generates 'B2'

  print("Session 2")
  with tf.compat.v1.Session() as sess2:
    print(sess2.run(a))  # generates 'A1'
    print(sess2.run(a))  # generates 'A2'
    print(sess2.run(b))  # generates 'B3'
    print(sess2.run(b))  # generates 'B4'
  ```

  To make the random sequences generated by all ops be repeatable across
  sessions, set a graph-level seed:

  ```python
  tf.compat.v1.random.set_random_seed(1234)
  a = tf.random.uniform([1])
  b = tf.random.normal([1])

  # Repeatedly running this block with the same graph will generate the same
  # sequences of 'a' and 'b'.
  print("Session 1")
  with tf.compat.v1.Session() as sess1:
    print(sess1.run(a))  # generates 'A1'
    print(sess1.run(a))  # generates 'A2'
    print(sess1.run(b))  # generates 'B1'
    print(sess1.run(b))  # generates 'B2'

  print("Session 2")
  with tf.compat.v1.Session() as sess2:
    print(sess2.run(a))  # generates 'A1'
    print(sess2.run(a))  # generates 'A2'
    print(sess2.run(b))  # generates 'B1'
    print(sess2.run(b))  # generates 'B2'
  ```

  Args:
    seed: integer.
  """
  if context.executing_eagerly():
    context.set_global_seed(seed)
  else:
    ops.get_default_graph().seed = seed


@tf_export('random.set_seed', v1=[])
def set_seed(seed):
  """Sets the global random seed.

  Operations that rely on a random seed actually derive it from two seeds:
  the global and operation-level seeds. This sets the global seed.

  Its interactions with operation-level seeds is as follows:

    1. If neither the global seed nor the operation seed is set: A randomly
      picked seed is used for this op.
    2. If the graph-level seed is set, but the operation seed is not:
      The system deterministically picks an operation seed in conjunction with
      the graph-level seed so that it gets a unique random sequence. Within the
      same version of tensorflow and user code, this sequence is deterministic.
      However across different versions, this sequence might change. If the
      code depends on particular seeds to work, specify both graph-level
      and operation-level seeds explicitly.
    3. If the operation seed is set, but the global seed is not set:
      A default global seed and the specified operation seed are used to
      determine the random sequence.
    4. If both the global and the operation seed are set:
      Both seeds are used in conjunction to determine the random sequence.

  To illustrate the user-visible effects, consider these examples:

  If neither the global seed nor the operation seed is set, we get different
  results for every call to the random op and every re-run of the program:

  ```python
  print(tf.random.uniform([1]))  # generates 'A1'
  print(tf.random.uniform([1]))  # generates 'A2'
  ```

  (now close the program and run it again)

  ```python
  print(tf.random.uniform([1]))  # generates 'A3'
  print(tf.random.uniform([1]))  # generates 'A4'
  ```

  If the global seed is set but the operation seed is not set, we get different
  results for every call to the random op, but the same sequence for every
  re-run of the program:

  ```python
  tf.random.set_seed(1234)
  print(tf.random.uniform([1]))  # generates 'A1'
  print(tf.random.uniform([1]))  # generates 'A2'
  ```

  (now close the program and run it again)

  ```python
  tf.random.set_seed(1234)
  print(tf.random.uniform([1]))  # generates 'A1'
  print(tf.random.uniform([1]))  # generates 'A2'
  ```

  The reason we get 'A2' instead 'A1' on the second call of `tf.random.uniform`
  above is because the secand call uses a different operation seed.

  Note that `tf.function` acts like a re-run of a program in this case. When
  the global seed is set but operation seeds are not set, the sequence of random
  numbers are the same for each `tf.function`. For example:

  ```python
  tf.random.set_seed(1234)

  @tf.function
  def f():
    a = tf.random.uniform([1])
    b = tf.random.uniform([1])
    return a, b

  @tf.function
  def g():
    a = tf.random.uniform([1])
    b = tf.random.uniform([1])
    return a, b

  print(f())  # prints '(A1, A2)'
  print(g())  # prints '(A1, A2)'
  ```

  If the operation seed is set, we get different results for every call to the
  random op, but the same sequence for every re-run of the program:

  ```python
  print(tf.random.uniform([1], seed=1))  # generates 'A1'
  print(tf.random.uniform([1], seed=1))  # generates 'A2'
  ```

  (now close the program and run it again)

  ```python
  print(tf.random.uniform([1], seed=1))  # generates 'A1'
  print(tf.random.uniform([1], seed=1))  # generates 'A2'
  ```

  The reason we get 'A2' instead 'A1' on the second call of `tf.random.uniform`
  above is because the same `tf.random.uniform` kernel (i.e. internel
  representation) is used by TensorFlow for all calls of it with the same
  arguments, and the kernel maintains an internal counter which is incremented
  every time it is executed, generating different results.

  Calling `tf.random.set_seed` will reset any such counters:

  ```python
  tf.random.set_seed(1234)
  print(tf.random.uniform([1], seed=1))  # generates 'A1'
  print(tf.random.uniform([1], seed=1))  # generates 'A2'
  tf.random.set_seed(1234)
  print(tf.random.uniform([1], seed=1))  # generates 'A1'
  print(tf.random.uniform([1], seed=1))  # generates 'A2'
  ```

  When multiple identical random ops are wrapped in a `tf.function`, their
  behaviors change because the ops no long share the same counter. For example:

  ```python
  @tf.function
  def foo():
    a = tf.random.uniform([1], seed=1)
    b = tf.random.uniform([1], seed=1)
    return a, b
  print(foo())  # prints '(A1, A1)'
  print(foo())  # prints '(A2, A2)'

  @tf.function
  def bar():
    a = tf.random.uniform([1])
    b = tf.random.uniform([1])
    return a, b
  print(bar())  # prints '(A1, A2)'
  print(bar())  # prints '(A3, A4)'
  ```

  The second call of `foo` returns '(A2, A2)' instead of '(A1, A1)' because
  `tf.random.uniform` maintains an internal counter. If you want `foo` to return
  '(A1, A1)' every time, use the stateless random ops such as
  `tf.random.stateless_uniform`. Also see `tf.random.experimental.Generator` for
  a new set of stateful random ops that use external variables to manage their
  states.

  Args:
    seed: integer.
  """
  set_random_seed(seed)
