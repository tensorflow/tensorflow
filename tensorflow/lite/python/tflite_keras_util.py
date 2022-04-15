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

"""Keras functions required by TensorFlow Lite.

The functions defined in this library have been copied over from Keras in order
to remove the dependency from TensorFlow Lite to Keras. The functions which
could not be copied over are accessed using the dependency inversion principle.
(for details, refer to tensorflow/python/util/keras_deps.py).
"""

import copy

from tensorflow.python.eager import def_function
from tensorflow.python.util import keras_deps
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc


def _enforce_names_consistency(specs):
  """Enforces that either all specs have names or none do."""

  def _has_name(spec):
    return hasattr(spec, 'name') and spec.name is not None

  def _clear_name(spec):
    spec = copy.deepcopy(spec)
    if hasattr(spec, 'name'):
      spec._name = None  # pylint:disable=protected-access
    return spec

  flat_specs = nest.flatten(specs)
  name_inconsistency = (
      any(_has_name(s) for s in flat_specs) and
      not all(_has_name(s) for s in flat_specs))

  if name_inconsistency:
    specs = nest.map_structure(_clear_name, specs)
  return specs


def model_input_signature(model, keep_original_batch_size=False):
  """Inspect model to get its input signature.

  The model's input signature is a list with a single (possibly-nested) object.
  This is due to the Keras-enforced restriction that tensor inputs must be
  passed in as the first argument.

  For example, a model with input {'feature1': <Tensor>, 'feature2': <Tensor>}
  will have input signature: [{'feature1': TensorSpec, 'feature2': TensorSpec}]

  Args:
    model: Keras Model object.
    keep_original_batch_size: A boolean indicating whether we want to keep using
      the original batch size or set it to None. Default is `False`, which means
      that the batch dim of the returned input signature will always be set to
      `None`.

  Returns:
    A list containing either a single TensorSpec or an object with nested
    TensorSpecs. This list does not contain the `training` argument.
  """
  if hasattr(model, 'save_spec'):
    input_specs = model.save_spec(dynamic_batch=not keep_original_batch_size)
    if input_specs is None:
      return None
    # The model's save spec returns (args, kwargs). Extract the first input arg
    # to use as the input spec.
    # TODO(b/188105669): Add support for multiple tensor arguments.
    input_specs = input_specs[0][0]
  else:
    input_specs = model._get_save_spec(  # pylint: disable=protected-access
        dynamic_batch=not keep_original_batch_size)
    if input_specs is None:
      return None
  input_specs = _enforce_names_consistency(input_specs)
  # Return a list with a single element as the model's input signature.
  if isinstance(input_specs,
                collections_abc.Sequence) and len(input_specs) == 1:
    # Note that the isinstance check filters out single-element dictionaries,
    # which should also be wrapped as a single-element list.
    return input_specs
  else:
    return [input_specs]


def raise_model_input_error(model):
  raise ValueError(
      'Model {} cannot be saved because the input shapes have not been '
      'set. Usually, input shapes are automatically determined from calling'
      ' `.fit()` or `.predict()`. To manually set the shapes, call '
      '`model.build(input_shape)`.'.format(model))


def _create_pseudo_names(tensors, prefix):
  """Creates pseudo {input | output} names for subclassed Models.

  Warning: this function should only be used to define default
  names for `Metics` and `SavedModel`. No other use cases should
  rely on a `Model`'s input or output names.

  Example with dict:

  `{'a': [x1, x2], 'b': x3}` becomes:
  `['a_1', 'a_2', 'b']`

  Example with list:

  `[x, y]` becomes:
  `['output_1', 'output_2']`

  Args:
    tensors: `Model`'s outputs or inputs.
    prefix: 'output_' for outputs, 'input_' for inputs.

  Returns:
    Flattened list of pseudo names.
  """

  def one_index(ele):
    # Start with "output_1" instead of "output_0".
    if isinstance(ele, int):
      return ele + 1
    return ele

  flat_paths = list(nest.yield_flat_paths(tensors))
  flat_paths = nest.map_structure(one_index, flat_paths)
  names = []
  for path in flat_paths:
    if not path:
      name = prefix + '1'  # Single output.
    else:
      name = '_'.join(str(p) for p in path)
      if isinstance(path[0], int):
        name = prefix + name
    names.append(name)
  return names


def create_pseudo_output_names(outputs):
  """Create pseudo output names for a subclassed Model."""
  return _create_pseudo_names(outputs, prefix='output_')


def trace_model_call(model, input_signature=None):
  """Trace the model call to create a tf.function for exporting a Keras model.

  Args:
    model: A Keras model.
    input_signature: optional, a list of tf.TensorSpec objects specifying the
      inputs to the model.

  Returns:
    A tf.function wrapping the model's call function with input signatures set.

  Raises:
    ValueError: if input signature cannot be inferred from the model.
  """
  if input_signature is None:
    if isinstance(model.call, def_function.Function):
      input_signature = model.call.input_signature

  if input_signature is None:
    input_signature = model_input_signature(model)

  if input_signature is None:
    raise_model_input_error(model)

  @def_function.function(input_signature=input_signature, autograph=False)
  def _wrapped_model(*args):
    """A concrete tf.function that wraps the model's call function."""
    # When given a single input, Keras models will call the model on the tensor
    # rather than a list consisting of the single tensor.
    inputs = args[0] if len(input_signature) == 1 else list(args)

    with keras_deps.get_call_context_function()().enter(
        model, inputs=inputs, build_graph=False, training=False, saving=True):
      outputs = model(inputs, training=False)

    return outputs

  return _wrapped_model
