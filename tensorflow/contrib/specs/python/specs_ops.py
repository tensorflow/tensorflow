# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Operators for concise TensorFlow network models.

This module is used as an environment for evaluating expressions
in the "specs" DSL.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow.contrib.ndlstm.python import lstm1d
from tensorflow.contrib.ndlstm.python import lstm2d
from tensorflow.contrib.specs.python import specs_lib


slim = tf.contrib.slim


# The following assignments don't appear to follow Google naming
# conventions, but that's because these are functions defined by
# higher-order function application, not "constants" and because they
# are the commands of the DSL.
# pylint: disable=invalid-name


class Idx(specs_lib.Composable):
  """Implements the identity function in network specifications."""

  def funcall(self, x):
    return x


class Conc(specs_lib.Composable):
  """Implements tensor concatenation in network specifications."""

  def __init__(self, dim, *args):
    """Concatenates tensors along the given dimension.

    Args:
        dim: dimension along which concatenation takes place
        *args: argument tensor functions to be concatenated
    """
    self.dim = dim
    self.funs = args

  def funcall(self, x):
    outputs = [f.funcall(x) for f in self.funs]
    return tf.concat_v2(outputs, self.dim)


External = specs_lib.External
Import = specs_lib.Import
Fun = specs_lib.Function
debug = specs_lib.debug
Print = Fun(tf.Print)
Id = Fun(tf.identity)

# TODO(tmb) add Assert

# Two letter names for the most common layers.

# 2D Convolutional layers with nonlinearities (s/t/r/m/l)
# TODO(tmb) add Cbs, Fbs etc. for batch norms

Cx = Fun(slim.conv2d)
Cs = Fun(slim.conv2d, activation_fn=tf.nn.sigmoid)
Ct = Fun(slim.conv2d, activation_fn=tf.nn.tanh)
Cr = Fun(slim.conv2d, activation_fn=tf.nn.relu)
Cm = Fun(slim.conv2d, activation_fn=tf.nn.softmax)
Cl = Fun(slim.conv2d, activation_fn=None)

# Fully connected slim with nonlinearities (s/t/r/m/l)

Fx = Fun(slim.fully_connected)
Fs = Fun(slim.fully_connected, activation_fn=tf.nn.sigmoid)
Ft = Fun(slim.fully_connected, activation_fn=tf.nn.tanh)
Fr = Fun(slim.fully_connected, activation_fn=tf.nn.relu)
Fm = Fun(slim.fully_connected, activation_fn=tf.nn.softmax)
Fl = Fun(slim.fully_connected, activation_fn=None)

# Pooling

Mp = Fun(slim.max_pool2d)
Ap = Fun(slim.avg_pool2d)

# Batch manipulations

Do = Fun(slim.dropout)
Bn = Fun(slim.batch_norm)
Lrn = Fun(tf.nn.local_response_normalization)
Unit = Fun(slim.unit_norm)

# Shape changes

Flat = Fun(slim.flatten)
Reshape = Fun(tf.reshape)
Transpose = Fun(tf.transpose)
Squeeze = Fun(tf.squeeze)
Expand = Fun(tf.expand_dims)

# Nonlinearities (rarely needed on their own)

Relu = Fun(tf.nn.relu)
Sig = Fun(tf.nn.sigmoid)
Tanh = Fun(tf.nn.tanh)
Smax = Fun(tf.nn.softmax)

# 2D LSTM

Lstm2 = Fun(lstm2d.separable_lstm)
Lstm2to1 = Fun(lstm2d.reduce_to_sequence)  # 2D to 1D
Lstm2to0 = Fun(lstm2d.reduce_to_final)  # 2D to depth-only


def Clstm2(n, *args, **kw):
  """2D LSTM with 3x3 pre-convolution."""
  return Cl(n, [3, 3]) | Lstm2(*args, **kw)


def Dws(n):
  """Depth-wise convolution + sigmoid (used after LSTM)."""
  return Cs(n, [1, 1])


def Dwm(n):
  """Depth-wise convolution + softmax (used after LSTM)."""
  return Cm(n, [1, 1])

# 1D LSTM

Lstm1 = Fun(lstm1d.ndlstm_base)
Lstm1to0 = Fun(lstm1d.sequence_to_final)  # 1D to depth-only
Ssm = Fun(lstm1d.sequence_softmax)

# Sharing of Variables


def Var(name, *args, **kw):
  """Implements an operator that generates a variable.

  This function is still experimental. Use it only
  for generating a single variable instance for
  each name.

  Args:
      name: Name of the variable.
      *args: Other arguments to get_variable.
      **kw: Other keywords for get_variable.

  Returns:
      A specs object for generating a variable.
  """
  def var(_):
    return tf.get_variable(name, *args, **kw)
  return specs_lib.Callable(var)


class Shared(specs_lib.Composable):
  """Wraps a scope with variable reuse around the subnetwork.

  This function is still experimental.

  Attributes:
      f: The shared subnetwork.
      name: A name for the shared scope.
      used: A flag indicating whether the scope has already been used.
  """

  shared_number = 1

  def __init__(self, subnet, name=None, scope=None):
    """Create the Shared operator.

    Use this as:

        f = Shared(Cr(100, 3))
        g = f | f | f

    Ordinarily, you do not need to provide either a name or a scope.
    Providing a name is useful if you want a well-defined namespace
    for the variables (e.g., for saving a subnet).

    Args:
        subnet: Definition of the shared network.
        name: Optional name for the shared context.
        scope: Optional shared scope (must be a Scope, not a string).

    Raises:
        ValueError: Scope is not of type tf.Scope, name is not
        of type string, or both scope and name are given together.
    """
    if scope is not None and not isinstance(scope, tf.VariableScope):
      raise ValueError("scope must be None or a VariableScope")
    if name is not None and not isinstance(scope, str):
      raise ValueError("name must be None or a string")
    if scope is not None and name is not None:
      raise ValueError("cannot provide both a name and a scope")
    if name is None:
      name = "Shared_%d" % Shared.shared_number
      Shared.shared_number += 1
    self.subnet = subnet
    self.name = name
    self.scope = scope

  def funcall(self, x):
    """Apply the shared operator to an input.

    This wraps a variable scope around the creation of the subnet.

    Args:
        x: The input argument on which the subnet is invoked.

    Returns:
        The output tensor from invoking the subnet constructor.
    """
    if self.scope is None:
      with tf.variable_scope(self.name, values=[x]) as scope:
        self.scope = scope
        return self.subnet.funcall(x)
    else:
      with tf.variable_scope(self.scope, values=[x], reuse=True):
        return self.subnet.funcall(x)

# AutoFunction bindings of some existing modules

TF = specs_lib.AutoFunction(tf)
NN = specs_lib.AutoFunction(tf.nn)
SL = specs_lib.AutoFunction(slim)

# pylint: enable=invalid-name
