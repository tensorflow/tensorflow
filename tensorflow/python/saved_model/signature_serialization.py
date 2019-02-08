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
"""Helpers for working with signatures in tf.saved_model.save."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training.checkpointable import base
from tensorflow.python.util import compat
from tensorflow.python.util import nest


DEFAULT_SIGNATURE_ATTR = "_default_save_signature"
SIGNATURE_ATTRIBUTE_NAME = "signatures"


def _get_signature(function):
  if (isinstance(function, (defun.Function, def_function.Function)) and
      function._input_signature is not None):  # pylint: disable=protected-access
    function = function.get_concrete_function()
  if not isinstance(function, defun.ConcreteFunction):
    return None
  return function


def _valid_signature(concrete_function):
  """Returns whether concrete function can be converted to a signature."""
  if not concrete_function.outputs:
    # Functions without outputs don't make sense as signatures. We just don't
    # have any way to run an Operation with no outputs as a SignatureDef in the
    # 1.x style.
    return False
  try:
    _normalize_outputs(concrete_function.structured_outputs, "unused", "unused")
  except ValueError:
    return False
  return True


def find_function_to_export(saveable_view):
  """Function to export, None if no suitable function was found."""
  # If the user did not specify signatures, check the root object for a function
  # that can be made into a signature.
  functions = saveable_view.list_functions(saveable_view.root)
  signature = functions.get(DEFAULT_SIGNATURE_ATTR, None)
  if signature is not None:
    return signature

  # TODO(andresp): Discuss removing this behaviour. It can lead to WTFs when a
  # user decides to annotate more functions with tf.function and suddenly
  # serving that model way later in the process stops working.
  possible_signatures = []
  for function in functions.values():
    concrete = _get_signature(function)
    if concrete is not None and _valid_signature(concrete):
      possible_signatures.append(concrete)
  if len(possible_signatures) == 1:
    single_function = possible_signatures[0]
    signature = _get_signature(single_function)
    if signature and  _valid_signature(signature):
      return signature
  return None


def canonicalize_signatures(signatures):
  """Converts `signatures` into a dictionary of concrete functions."""
  if signatures is None:
    return {}
  if not isinstance(signatures, collections.Mapping):
    signatures = {
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signatures}
  concrete_signatures = {}
  for signature_key, function in signatures.items():
    signature_function = _get_signature(function)
    if signature_function is None:
      raise ValueError(
          ("Expected a TensorFlow function to generate a signature for, but "
           "got {}. Only `tf.functions` with an input signature or "
           "concrete functions can be used as a signature.").format(function))

    # Re-wrap the function so that it only takes keyword arguments and it
    # returns a dictionary of Tensors. This matches the format of 1.x-style
    # signatures.
    # pylint: disable=cell-var-from-loop
    @def_function.function
    def signature_wrapper(**kwargs):
      structured_outputs = signature_function(**kwargs)
      return _normalize_outputs(
          structured_outputs, signature_function.name, signature_key)
    # TODO(b/123902469): Use ConcreteFunction.structured_inputs once their names
    # always match keyword arguments.
    tensor_spec_signature = {}
    for keyword, tensor in zip(
        signature_function._arg_keywords,  # pylint: disable=protected-access
        signature_function.inputs):
      keyword = compat.as_str(keyword)
      tensor_spec_signature[keyword] = tensor_spec.TensorSpec.from_tensor(
          tensor, name=keyword)
    concrete_signatures[signature_key] = (
        signature_wrapper.get_concrete_function(**tensor_spec_signature))
    # pylint: enable=cell-var-from-loop
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


# _SignatureMap is immutable to ensure that users do not expect changes to be
# reflected in the SavedModel. Using public APIs, tf.saved_model.load() is the
# only way to create a _SignatureMap and there is no way to modify it. So we can
# safely ignore/overwrite ".signatures" attributes attached to objects being
# saved if they contain a _SignatureMap. A ".signatures" attribute containing
# any other type (e.g. a regular dict) will raise an exception asking the user
# to first "del obj.signatures" if they want it overwritten.
class _SignatureMap(collections.Mapping, base.Checkpointable):
  """A collection of SavedModel signatures."""

  def __init__(self):
    self._signatures = {}

  def _add_signature(self, name, concrete_function):
    """Adds a signature to the _SignatureMap."""
    # Ideally this object would be immutable, but restore is streaming so we do
    # need a private API for adding new signatures to an existing object.
    self._signatures[name] = concrete_function

  def __getitem__(self, key):
    return self._signatures[key]

  def __iter__(self):
    return iter(self._signatures)

  def __len__(self):
    return len(self._signatures)

  def __repr__(self):
    return "_SignatureMap({})".format(self._signatures)

  def _list_functions_for_serialization(self):
    return {
        key: value for key, value in self.items()
        if isinstance(value, (def_function.Function, defun.ConcreteFunction))
    }


revived_types.register_revived_type(
    "signature_map",
    lambda obj: isinstance(obj, _SignatureMap),
    versions=[revived_types.VersionedTypeRegistration(
        # Standard dependencies are enough to reconstruct the checkpointable
        # items in dictionaries, so we don't need to save any extra information.
        object_factory=lambda proto: _SignatureMap(),
        version=1,
        min_producer_version=1,
        min_consumer_version=1,
        setter=_SignatureMap._add_signature  # pylint: disable=protected-access
    )])


def create_signature_map(signatures, saveable_view):
  """Performs sanity checks and creates an object containing `signatures`."""
  for name, dep in saveable_view.list_dependencies(
      saveable_view.root):
    if name == SIGNATURE_ATTRIBUTE_NAME:
      if not isinstance(dep, _SignatureMap):
        raise ValueError(
            ("Exporting an object {} which has an attribute named "
             "'{signatures}'. This is a reserved attribute used to store "
             "SavedModel signatures in objects which come from "
             "`tf.saved_model.load`. Delete this attribute "
             "(e.g. 'del obj.{signatures}') before saving if this shadowing is "
             "acceptable.").format(
                 saveable_view.root,
                 signatures=SIGNATURE_ATTRIBUTE_NAME))
      break
  signature_map = _SignatureMap()
  for name, func in signatures.items():
    # This true of any signature that came from canonicalize_signatures. Here as
    # a sanity check on saving; crashing on load (e.g. in _add_signature) would
    # be more problematic in case future export changes violated these
    # assertions.
    assert isinstance(func, defun.ConcreteFunction)
    assert isinstance(func.structured_outputs, collections.Mapping)
    assert 0 == func._num_positional_args  # pylint: disable=protected-access
    signature_map._add_signature(name, func)  # pylint: disable=protected-access
  return signature_map
