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
"""Autograph specifc overrides for tf.data.ops."""
import functools

import numpy as np

from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import nest


# TODO(mdan): These checks should be easier. Fix the nest API.
def _verify_spec_compatible(input_name, spec_name, input_, spec):
  """Verifies that a symbol has a type compatible vith a given spec.

  Here, compatibility is viewed in the general TensorFlow sense: that the dtypes
  are the same after implicit conversion, if both are tensors.

  This verifier ensures consistent treatment of types across AutoGraph.

  Args:
    input_name: A name to use for `input_` in error messages.
    spec_name: A name to use for `spec` in error messages.
    input_: Any, value to verify.
    spec: TypeSpec that `input_` must be compatible with.

  Raises:
    ValueError if the two types have been determined not to be compatible.
  """
  assert isinstance(spec, tensor_spec.TensorSpec)
  if input is None:
    # TODO(mdan): raise from None when switching to Py3.
    raise ValueError("{} cannot be None".format(input_name))

  # TODO(mdan): Use TensorCompatible when ready.
  if isinstance(input_, (bool, int, float, str, np.ndarray)):
    input_ = ops.convert_to_tensor_v2(input_)

  input_dtype = getattr(input_, "dtype", None)

  if input_dtype != spec.dtype:
    input_dtype_str = "no dtype" if input_dtype is None else str(input_dtype)

    raise TypeError(
        "{} must have the same dtype as {}. Expected {}, got {}".format(
            input_name, spec_name, spec.dtype, input_dtype_str
        )
    )


def _verify_structure_compatible(input_name, spec_name, input_, spec):
  """Verifies that possibly-structured symbol has types compatible vith another.

  See _verify_spec_compatible for a more concrete meaning of "compatible".
  Unspec _verify_spec_compatible, which handles singular Tensor-spec objects,
  verify_structures_compatible can process structures recognized by tf.nest.

  Args:
    input_name: A name to use for `input_` in error messages.
    spec_name: A name to use for `spec` in error messages.
    input_: Any, value to verify. May, but doesn't need to, be a structure.
    spec: Any, value that `input_` must be compatible with. May, but doesn't
      need to, be a structure.

  Raises:
    ValueError if the two types have been determined not to be compatible.
  """
  try:
    nest.assert_same_structure(input_, spec, expand_composites=True)
  except (ValueError, TypeError) as e:
    raise TypeError(
        "{} must have the same element structure as {}.\n\n{}".format(
            input_name, spec_name, str(e)
        )
    ) from e

  nest.map_structure(
      functools.partial(_verify_spec_compatible, input_name, spec_name), input_,
      spec)


def _next_tf_iterator(iterator, default=py_builtins.UNSPECIFIED):
  if default is py_builtins.UNSPECIFIED:
    # Without a default, fall back to the "normal" behavior which raises
    # a runtime exception.
    return next(iterator)
  opt_iterate = iterator.get_next_as_optional()
  _verify_structure_compatible(
      "the default argument", "the iterate", default, iterator.element_spec
  )
  return control_flow_ops.cond(
      opt_iterate.has_value(), opt_iterate.get_value, lambda: default
  )


def register_overrides():
  py_builtins.next_registry.register(
      iterator_ops.OwnedIterator, _next_tf_iterator
  )
  control_flow.for_loop_registry.register(
      iterator_ops.OwnedIterator, control_flow._tf_iterator_for_stmt  # pylint: disable=protected-access
  )
