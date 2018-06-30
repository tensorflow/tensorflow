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

from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.specs.python import specs_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

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
    return array_ops.concat(outputs, self.dim)


External = specs_lib.External
Import = specs_lib.Import
Fun = specs_lib.Function
debug = specs_lib.debug
Print = Fun(logging_ops.Print)
Id = Fun(array_ops.identity)

# TODO(tmb) add Assert

# Two letter names for the most common layers.

# 2D Convolutional layers with nonlinearities (s/t/r/m/l)
# TODO(tmb) add Cbs, Fbs etc. for batch norms

Cx = Fun(layers.conv2d)
Cs = Fun(layers.conv2d, activation_fn=math_ops.sigmoid)
Ct = Fun(layers.conv2d, activation_fn=math_ops.tanh)
Cr = Fun(layers.conv2d, activation_fn=nn_ops.relu)
Cm = Fun(layers.conv2d, activation_fn=nn_ops.softmax)
Cl = Fun(layers.conv2d, activation_fn=None)

# Fully connected slim with nonlinearities (s/t/r/m/l)

Fx = Fun(layers.fully_connected)
Fs = Fun(layers.fully_connected, activation_fn=math_ops.sigmoid)
Ft = Fun(layers.fully_connected, activation_fn=math_ops.tanh)
Fr = Fun(layers.fully_connected, activation_fn=nn_ops.relu)
Fm = Fun(layers.fully_connected, activation_fn=nn_ops.softmax)
Fl = Fun(layers.fully_connected, activation_fn=None)

# Pooling

Mp = Fun(layers.max_pool2d)
Ap = Fun(layers.avg_pool2d)

# Batch manipulations

Do = Fun(layers.dropout)
Bn = Fun(layers.batch_norm)
Lrn = Fun(nn.local_response_normalization)
Unit = Fun(layers.unit_norm)

# Shape changes

Flat = Fun(layers.flatten)
Reshape = Fun(array_ops.reshape)
Transpose = Fun(array_ops.transpose)
Squeeze = Fun(array_ops.squeeze)
Expand = Fun(array_ops.expand_dims)

# Nonlinearities (rarely needed on their own)

Relu = Fun(nn_ops.relu)
Sig = Fun(math_ops.sigmoid)
Tanh = Fun(math_ops.tanh)
Smax = Fun(nn_ops.softmax)


def Dws(n):
  """Depth-wise convolution + sigmoid (used after LSTM)."""
  return Cs(n, [1, 1])


def Dwm(n):
  """Depth-wise convolution + softmax (used after LSTM)."""
  return Cm(n, [1, 1])

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
    return variable_scope.get_variable(name, *args, **kw)

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
    if scope is not None and not isinstance(scope,
                                            variable_scope.VariableScope):
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
      with variable_scope.variable_scope(self.name, values=[x]) as scope:
        self.scope = scope
        return self.subnet.funcall(x)
    else:
      with variable_scope.variable_scope(self.scope, values=[x], reuse=True):
        return self.subnet.funcall(x)
