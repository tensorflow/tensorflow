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
# pylint: disable=unidiomatic-typecheck
"""Eager semantics for polymorphic function."""

from tensorflow.python.util.tf_export import tf_export


RUN_FUNCTIONS_EAGERLY = False


@tf_export("config.functions_run_eagerly")
def functions_run_eagerly():
  """Returns the value of the `run_functions_eagerly` setting."""
  return RUN_FUNCTIONS_EAGERLY


@tf_export("config.run_functions_eagerly")
def run_functions_eagerly(run_eagerly):
  """Enables / disables eager execution of `tf.function`s.

  Calling `tf.config.run_functions_eagerly(True)` will make all
  invocations of `tf.function` run eagerly instead of running as a traced graph
  function. This can be useful for debugging. As the code now runs line-by-line,
  you can add arbitrary `print` messages or pdb breakpoints to monitor the
  inputs/outputs of each Tensorflow operation. However, you should avoid using
  this for actual production because it significantly slows down execution.

  >>> def my_func(a):
  ...  print(f'a: {a}')
  ...  return a + a
  >>> a_fn = tf.function(my_func)

  >>> # A side effect the first time the function is traced
  >>> # In tracing time, `a` is printed with shape and dtype only
  >>> a_fn(tf.constant(1))
  a: Tensor("a:0", shape=(), dtype=int32)
  <tf.Tensor: shape=(), dtype=int32, numpy=2>

  >>> # `print` is a python side effect, it won't execute as the traced function
  >>> # is called
  >>> a_fn(tf.constant(2))
  <tf.Tensor: shape=(), dtype=int32, numpy=4>

  >>> # Now, switch to eager running
  >>> tf.config.run_functions_eagerly(True)
  >>> # The code now runs eagerly and the actual value of `a` is printed
  >>> a_fn(tf.constant(2))
  a: 2
  <tf.Tensor: shape=(), dtype=int32, numpy=4>

  >>> # Turn this back off
  >>> tf.config.run_functions_eagerly(False)

  Note: This flag has no effect on functions passed into tf.data transformations
  as arguments. tf.data functions are never executed eagerly and are always
  executed as a compiled Tensorflow Graph.

  Args:
    run_eagerly: Boolean. Whether to run functions eagerly.
  """
  global RUN_FUNCTIONS_EAGERLY
  RUN_FUNCTIONS_EAGERLY = bool(run_eagerly)
