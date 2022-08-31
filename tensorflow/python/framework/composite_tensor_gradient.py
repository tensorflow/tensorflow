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
"""Gradient support for Composite Tensors."""

import abc
import sys

from tensorflow.python.framework import composite_tensor
from tensorflow.python.util import nest


# pylint:disable=g-import-not-at-top
if sys.version_info >= (3, 8):
  from typing import Protocol
  from typing import runtime_checkable
else:
  from typing_extensions import Protocol
  from typing_extensions import runtime_checkable
# pylint:enable=g-import-not-at-top


# TODO(xjun): Add CompositeTensorGradient support for SparseTensor,
# StructuredTensor, and MaskedTensor.
class CompositeTensorGradient(object, metaclass=abc.ABCMeta):
  """Class used to help compute gradients for CompositeTensors.

  This abstract base class defines two methods: `get_gradient_components`, which
  returns the components of a value that should be included in gradients; and
  `replace_gradient_components`, which replaces the gradient components in a
  value.  These methods can be used to compute the gradient of a `y` with
  respect to `x` (`grad(y, x)`) as follows:

  * If `y` is a `CompositeTensor` with `CompositeTensorGradient` `cg` =
    `y.__composite_gradient__`, then `grad(y, x)` =
    `grad(cg.get_gradient_components(y), x)`.

  * If `x` is a `CompositeTensor` with `CompositeTensorGradient` `cg` =
    'x.__composite_gradient__', then `grad(y, x)` =
    `cg.replace_gradient_components(x, grad(y, cg.get_gradient_components(x))`.
  """

  @abc.abstractmethod
  def get_gradient_components(self, value):
    """Returns the components of `value` that should be included in gradients.

    This method may not call TensorFlow ops, since any new ops added to the
    graph would not be propertly tracked by the gradient mechanisms.

    Args:
      value: A `CompositeTensor` value.

    Returns:
      A nested structure of `Tensor` or `IndexedSlices`.
    """
    raise NotImplementedError(
        f"{type(self).__name__}.get_gradient_components()")

  @abc.abstractmethod
  def replace_gradient_components(self, value, component_grads):
    """Replaces the gradient components in `value` with `component_grads`.

    Args:
      value: A value with its gradient components compatible with
        `component_grads`.
      component_grads: A nested structure of `Tensor` or `IndexedSlices` or
        `None` (for unconnected gradients).

    Returns:
      A copy of `value`, where the components that should be included in
      gradients have been replaced by `component_grads`; or `None` (if
      `component_grads` includes `None`).
    """
    raise NotImplementedError(
        f"{type(self).__name__}.replace_gradient_components()")


@runtime_checkable
class CompositeTensorGradientProtocol(Protocol):
  """Protocol for adding gradient support to CompositeTensors."""
  __composite_gradient__: CompositeTensorGradient


class WithValuesCompositeTensorGradient(CompositeTensorGradient):
  """CompositeTensorGradient based on `T.values` and `T.with_values`."""

  def get_gradient_components(self, value):
    return value.values

  def replace_gradient_components(self, value, component_grads):
    return value.with_values(component_grads)


def _get_tensors_for_gradient(x):
  """Returns the Tensors in `x` that should be differentiated.

  Args:
    x: A `Tensor` or `CompositeTensor`.

  Returns:
    A `Tensor` or a nested structure of `Tensor`.
  """
  if not isinstance(x, composite_tensor.CompositeTensor):
    return x

  if not isinstance(x, CompositeTensorGradientProtocol):
    raise ValueError(
        f"Type {type(x).__name__} is not supported as a gradient source or "
        "gradient target.")
  composite_gradient = x.__composite_gradient__
  gradient_components = composite_gradient.get_gradient_components(x)
  if gradient_components is x:
    return x
  return nest.map_structure(_get_tensors_for_gradient, gradient_components)


def _replace_tensors_for_gradient(x, grad):
  """Replaces the tensors in `x` that should be differentiated with `grad`.

  Args:
    x: A `Tensor` or `CompositeTensor`.
    grad: A nested structure of `Tensor`, with the same structure as the value
      returned by `_get_tensors_for_gradient(x)`.

  Returns:
    A `Tensor` or `CompositeTensor`.
  """
  if not isinstance(x, composite_tensor.CompositeTensor):
    return grad

  if not isinstance(x, CompositeTensorGradientProtocol):
    raise ValueError(
        f"Type {type(x).__name__} is not supported as a gradient source.")

  composite_gradient = x.__composite_gradient__
  x_components = composite_gradient.get_gradient_components(x)
  if x_components is x:
    grad_components = grad
  else:
    grad_components = nest.map_structure_up_to(x_components,
                                               _replace_tensors_for_gradient,
                                               x_components, grad)
  if grad_components is None:
    return None
  return composite_gradient.replace_gradient_components(x, grad_components)


def get_flat_tensors_for_gradients(xs):
  """Returns a flat list of Tensors that should be differentiated for `xs`.

  Args:
    xs: A list of `Tensor`s or `CompositeTensor`s.

  Returns:
    A flat list of `Tensor`s constructed from `xs`, where `Tensor` values are
    left as-is, and `CompositeTensor`s are replaced with
    `_get_tensors_for_gradient(x)`.
  """
  return nest.flatten([_get_tensors_for_gradient(x) for x in xs])


def replace_flat_tensors_for_gradients(xs, flat_grads):
  """Replaces Tensors that should be differentiated in `xs` with `flat_grads`.

  Args:
    xs: A list of `Tensor`s or `CompositeTensor`s.
    flat_grads: A list of `Tensor`.

  Returns:
    A list of `Tensor` or `CompositeTensor`.
  """
  xs_structure = [_get_tensors_for_gradient(x) for x in xs]
  grads = nest.pack_sequence_as(xs_structure, flat_grads)
  return [_replace_tensors_for_gradient(x, grad) for x, grad in zip(xs, grads)]
