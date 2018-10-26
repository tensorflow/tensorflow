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
"""Exports a SavedModel from a Checkpointable Python object."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import utils
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.util import compat
from tensorflow.python.util import nest


def _canonicalize_signatures(signatures):
  """Converts `signatures` into a dictionary of concrete functions."""
  if signatures is None:
    signatures = {}
  elif not isinstance(signatures, collections.Mapping):
    signatures = {
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signatures}
  concrete_signatures = {}
  for serving_key, signature_function in signatures.items():
    if isinstance(signature_function, (function.PolymorphicFunction,
                                       def_function.PolymorphicFunction)):
      input_signature = signature_function._input_signature  # pylint: disable=protected-access
      if input_signature is None:
        raise ValueError(
            ("Unable to use the function {} as a signature directly. Functions "
             "used to generate serving signatures must either have an "
             "`input_signature=` specified when constructed, or must be "
             "converted to concrete functions using "
             "`f.get_concrete_function(...)`.").format(signature_function))
      signature_function = signature_function.get_concrete_function()
    elif not isinstance(signature_function, function.Function):
      raise ValueError(
          ("Expected a TensorFlow function to generate a signature for, but "
           "got {}. Python functions may be decorated with "
           "`@tf.function(input_signature=...)` and passed as signatures "
           "directly, or created without a signature using `@tf.function` "
           "and then converted to a concrete TensorFlow function using "
           "`f.get_concrete_function(...)`.").format(signature_function))
    concrete_signatures[serving_key] = signature_function
  return concrete_signatures


def _is_flat(sequence):
  sequence_flat = nest.flatten(sequence)
  try:
    nest.assert_same_structure(sequence_flat, sequence)
    return True
  except ValueError:
    return False
  except TypeError:
    return False


def _normalize_outputs(outputs, function_name, signature_key):
  """Construct an output dictionary from unnormalized function outputs."""
  if isinstance(outputs, collections.Mapping):
    for key, value in outputs.items():
      if not isinstance(value, ops.Tensor):
        raise ValueError(
            ("Got a dictionary containing non-Tensor value {} for key {} "
             "in the output of the function {} used to generate a SavedModel "
             "signature. Dictionaries outputs for functions used as signatures "
             "should have one Tensor output per string key.")
            .format(value, key, compat.as_str_any(function_name)))
    return outputs
  else:
    original_outputs = outputs
    if not isinstance(outputs, collections.Sequence):
      outputs = [outputs]
    if not _is_flat(outputs):
      raise ValueError(
          ("Got non-flat outputs '{}' from '{}' for SavedModel "
           "signature '{}'. Signatures have one Tensor per output, so "
           "to have predictable names Python functions used to generate "
           "these signatures should avoid outputting Tensors in nested "
           "structures.")
          .format(original_outputs, function_name, signature_key))
    return {("output_{}".format(output_index)): output
            for output_index, output
            in enumerate(outputs)}


def _tensor_dict_to_tensorinfo(tensor_dict):
  return {key: utils.build_tensor_info(value)
          for key, value in tensor_dict.items()}


def _generate_signatures(signature_functions):
  """Validates and calls `signature_functions` in the default graph.

  Args:
    signature_functions: A dictionary mapping string keys to concrete TensorFlow
      functions (e.g. from `_canonicalize_signatures`) which will be used to
      generate SignatureDefs.

  Returns:
    Each function in the `signature_functions` dictionary is called with
    placeholder Tensors, generating a function call operation and output
    Tensors. The placeholder Tensors, the function call operation, and the
    output Tensors from the function call are part of the default Graph.

    This function then returns a dictionary with the same structure as
    `signature_functions`, with the concrete functions replaced by SignatureDefs
    implicitly containing information about how to call each function from a
    TensorFlow 1.x Session / the C++ Loader API. These SignatureDefs reference
    the generated placeholders and Tensor outputs by name.

    The caller is expected to include the default Graph set while calling this
    function as a MetaGraph in a SavedModel, including the returned
    SignatureDefs as part of that MetaGraph.
  """
  signatures = {}
  for signature_key, func in sorted(signature_functions.items()):
    function.register_concrete(func)
    # `exterior_placeholders` holds placeholders which are outside the function
    # body, directly contained in a MetaGraph of the SavedModel. The function
    # body itself contains nearly identical placeholders used when running the
    # function, but these exterior placeholders allow Session-based APIs to call
    # the function using feeds and fetches which name Tensors in the MetaGraph.
    exterior_placeholders = {}
    kwargs = {}
    for placeholder in func.inputs:
      user_input_name = compat.as_str_any(
          placeholder.op.get_attr("_user_specified_name"))
      # If the internal placeholders for a function have names which were
      # uniquified by TensorFlow, then a single user-specified argument name
      # must refer to multiple Tensors. The resulting signatures would be
      # confusing to call. Instead, we throw an exception telling the user to
      # specify explicit names.
      if user_input_name != placeholder.op.name:
        # This should be unreachable, since concrete functions may not be
        # generated with non-unique argument names.
        raise ValueError(
            ("Got non-flat/non-unique argument names for SavedModel "
             "signature '{}': more than one argument to '{}' was named '{}'. "
             "Signatures have one Tensor per named input, so to have "
             "predictable names Python functions used to generate these "
             "signatures should avoid *args and Tensors in nested "
             "structures unless unique names are specified for each. Use "
             "tf.TensorSpec(..., name=...) to provide a name for a Tensor "
             "input.")
            .format(signature_key, compat.as_str_any(func.name),
                    user_input_name))
      arg_placeholder = array_ops.placeholder(
          shape=placeholder.shape,
          dtype=placeholder.dtype,
          name="{}_{}".format(signature_key, user_input_name))
      exterior_placeholders[user_input_name] = arg_placeholder
      kwargs[user_input_name] = arg_placeholder
    outputs = _normalize_outputs(
        func(**kwargs), func.name, signature_key)
    signatures[signature_key] = signature_def_utils.build_signature_def(
        _tensor_dict_to_tensorinfo(exterior_placeholders),
        _tensor_dict_to_tensorinfo(outputs))
  return signatures


def _make_graph_def(signature_functions):
  """Generates and exports call ops for `signature_functions`."""
  # TODO(allenl): Handle variables
  signatures = {}
  exported_graph = ops.Graph()
  with exported_graph.as_default():
    signatures = _generate_signatures(signature_functions)
  graph_def = exported_graph.as_graph_def(add_shapes=True)
  return graph_def, signatures


def export(obj, export_dir, signatures=None):
  # pylint: disable=line-too-long
  """Exports the Checkpointable object `obj` to [SavedModel format](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md).

  The `signatures` argument indicates TensorFlow functions which will be
  available to programs which consume `SavedModel`s, for example serving
  APIs. Python functions may be decorated with
  `@tf.function(input_signature=...)` and passed as signatures directly, or
  created without a signature using `@tf.function` and then converted to a
  concrete TensorFlow function using `f.get_concrete_function(...)`.

  In either case, `Tensor` inputs to `signatures` functions which are not
  associated with a unique Python argument name must have names explicitly
  specified in their `tf.TensorSpec` objects. Cases where this is necessary
  include positional arguments passed through variadic `*args` and multiple
  `Tensor` inputs which are part of the same nested structure.

  The outputs of functions used as `signatures` must either be flat lists, in
  which case outputs will be numbered, or a dictionary mapping string keys to
  Tensors, in which case the string keys will be used to name outputs.

  Exporting with a signature specified:

  ```python
  class Model(tf.keras.Model):

    @tf.function(input_signature=tf.TensorSpec(shape=[None], dtype=tf.string))
    def serve(serialized):
      ...

  m = Model()
  tf.saved_model.export(m, '/tmp/saved_model/', signatures=m.serve)
  ```

  Exporting from a function without a fixed signature:

  ```python
  class Model(tf.keras.Model):

    @tf.function
    def compute(x):
      ...

  m = Model()
  tf.saved_model.export(
      m, '/tmp/saved_model/',
      signatures=m.compute.get_concrete_function(
          tf.TensorSpec(shape=[None, 3], dtype=tf.float32, name="inp")))
  ```

  Args:
    obj: A checkpointable object to export.
    export_dir: A directory in which to write the SavedModel.
    signatures: Optional, either a `tf.function` with an input signature
      specified or the result of `f.get_concrete_function` on a
      `tf.function`-decorated function `f`, in which case `f` will be used to
      generate a signature for the SavedModel under the default serving
      signature key. `signatures` may also be a dictionary, in which case it
      maps from signature keys to either `tf.function` instances with input
      signatures or concrete functions. The keys of such a dictionary may be
      arbitrary strings, but will typically be from the
      `tf.saved_model.signature_constants` module.

  Raises:
    ValueError: If `obj` is not checkpointable.
  """
  # pylint: enable=line-too-long
  if not isinstance(obj, checkpointable.CheckpointableBase):
    raise ValueError(
        "Expected a Checkpointable object for export, got {}.".format(obj))
  signatures = _canonicalize_signatures(signatures)
  graph_def, signatures = _make_graph_def(signatures)
  saved_model = saved_model_pb2.SavedModel()
  saved_model.saved_model_schema_version = (
      constants.SAVED_MODEL_SCHEMA_VERSION)
  meta_graph_def = saved_model.meta_graphs.add()
  # TODO(allenl): Factor out some subset of SavedModelBuilder which is 2.x
  # compatible (no sessions) and share it with this export API rather than
  # making a SavedModel proto and writing it directly.
  meta_graph_def.graph_def.MergeFrom(graph_def)
  for signature_key, signature in signatures.items():
    meta_graph_def.signature_def[signature_key].MergeFrom(signature)
  file_io.recursive_create_dir(export_dir)
  path = os.path.join(
      compat.as_bytes(export_dir),
      compat.as_bytes(constants.SAVED_MODEL_FILENAME_PB))
  file_io.write_string_to_file(path, saved_model.SerializeToString())
