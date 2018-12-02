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
"""Defines abstract class for `ConstrainedMinimizationProblem`s.

A ConstrainedMinimizationProblem consists of an objective function to minimize,
and a set of constraint functions that are constrained to be nonpositive.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six


@six.add_metaclass(abc.ABCMeta)
class ConstrainedMinimizationProblem(object):
  """Abstract class representing a `ConstrainedMinimizationProblem`.

  A ConstrainedMinimizationProblem consists of an objective function to
  minimize, and a set of constraint functions that are constrained to be
  nonpositive.

  In addition to the constraint functions, there may (optionally) be proxy
  constraint functions: a ConstrainedOptimizer will attempt to penalize these
  proxy constraint functions so as to satisfy the (non-proxy) constraints. Proxy
  constraints could be used if the constraints functions are difficult or
  impossible to optimize (e.g. if they're piecewise constant), in which case the
  proxy constraints should be some approximation of the original constraints
  that is well-enough behaved to permit successful optimization.
  """

  @abc.abstractproperty
  def objective(self):
    """Returns the objective function.

    Returns:
      A 0d tensor that should be minimized.
    """
    pass

  @property
  def num_constraints(self):
    """Returns the number of constraints.

    Returns:
      An int containing the number of constraints.

    Raises:
      ValueError: If the constraints (or proxy_constraints, if present) do not
        have fully-known shapes, OR if proxy_constraints are present, and the
        shapes of constraints and proxy_constraints are fully-known, but they're
        different.
    """
    constraints_shape = self.constraints.get_shape()
    if self.proxy_constraints is None:
      proxy_constraints_shape = constraints_shape
    else:
      proxy_constraints_shape = self.proxy_constraints.get_shape()

    if (constraints_shape.ndims is None or
        proxy_constraints_shape.ndims is None or
        any(ii is None for ii in constraints_shape.as_list()) or
        any(ii is None for ii in proxy_constraints_shape.as_list())):
      raise ValueError(
          "constraints and proxy_constraints must have fully-known shapes")
    if constraints_shape != proxy_constraints_shape:
      raise ValueError(
          "constraints and proxy_constraints must have the same shape")

    size = 1
    for ii in constraints_shape.as_list():
      size *= ii
    return int(size)

  @abc.abstractproperty
  def constraints(self):
    """Returns the vector of constraint functions.

    Letting g_i be the ith element of the constraints vector, the ith constraint
    will be g_i <= 0.

    Returns:
      A tensor of constraint functions.
    """
    pass

  # This is a property, instead of an abstract property, since it doesn't need
  # to be overridden: if proxy_constraints returns None, then there are no
  # proxy constraints.
  @property
  def proxy_constraints(self):
    """Returns the optional vector of proxy constraint functions.

    The difference between `constraints` and `proxy_constraints` is that, when
    proxy constraints are present, the `constraints` are merely EVALUATED during
    optimization, whereas the `proxy_constraints` are DIFFERENTIATED. If there
    are no proxy constraints, then the `constraints` are both evaluated and
    differentiated.

    For example, if we want to impose constraints on step functions, then we
    could use these functions for `constraints`. However, because a step
    function has zero gradient almost everywhere, we can't differentiate these
    functions, so we would take `proxy_constraints` to be some differentiable
    approximation of `constraints`.

    Returns:
      A tensor of proxy constraint functions.
    """
    return None

  # This is a property, instead of an abstract property, since it doesn't need
  # to be overridden: if pre_train_ops returns None, then there are no ops to
  # run before train_op.
  @property
  def pre_train_ops(self):
    """Returns a list of `Operation`s to run before the train_op.

    When a `ConstrainedOptimizer` creates a train_op (in `minimize`
    `minimize_unconstrained`, or `minimize_constrained`), it will include these
    ops before the main training step.

    Returns:
      A list of `Operation`s.
    """
    return None
