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

from absl import logging

from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import function_serialization
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.trackable import base
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc

DEFAULT_SIGNATURE_ATTR = "_default_save_signature"
SIGNATURE_ATTRIBUTE_NAME = "signatures"
# Max number of warnings to show if signature contains normalized input names.
_NUM_DISPLAY_NORMALIZED_SIGNATURES = 5


def _get_signature(function):
  if (
      isinstance(function, (defun.Function, def_function.Function))
      and function.input_signature is not None
  ):
    function = function._get_concrete_function_garbage_collected()  # pylint: disable=protected-access
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
    _validate_inputs(concrete_function)
    _normalize_outputs(concrete_function.structured_outputs, "unused", "unused")
  except ValueError:
    return False
  return True


def _validate_inputs(concrete_function):
  """Raises error if input type is tf.Variable."""
  if any(
      isinstance(inp, resource_variable_ops.VariableSpec)
      for inp in nest.flatten(concrete_function.structured_input_signature)
  ):
    raise ValueError(
        f"Unable to serialize concrete_function '{concrete_function.name}'"
        "with tf.Variable input. Functions that expect tf.Variable "
        "inputs cannot be exported as signatures."
    )


def _get_signature_name_changes(concrete_function):
  """Checks for user-specified signature input names that are normalized."""
  # Map of {user-given name: normalized name} if the names are un-identical.
  name_changes = {}
  for signature_input_name, graph_input in zip(
      concrete_function.function_def.signature.input_arg,
      concrete_function.graph.inputs,
  ):
    try:
      user_specified_name = compat.as_str(
          graph_input.op.get_attr("_user_specified_name")
      )
      if signature_input_name.name != user_specified_name:
        name_changes[user_specified_name] = signature_input_name.name
    except ValueError:
      # Signature input does not have a user-specified name.
      pass
  return name_changes


def find_function_to_export(saveable_view):
  """Function to export, None if no suitable function was found."""
  # If the user did not specify signatures, check the root object for a function
  # that can be made into a signature.
  children = saveable_view.list_children(saveable_view.root)

  # TODO(b/205014194): Discuss removing this behaviour. It can lead to WTFs when
  # a user decides to annotate more functions with tf.function and suddenly
  # serving that model way later in the process stops working.
  possible_signatures = []
  for name, child in children:
    if not isinstance(child, (def_function.Function, defun.ConcreteFunction)):
      continue
    if name == DEFAULT_SIGNATURE_ATTR:
      return child
    concrete = _get_signature(child)
    if concrete is not None and _valid_signature(concrete):
      possible_signatures.append(concrete)

  if len(possible_signatures) == 1:
    single_function = possible_signatures[0]
    signature = _get_signature(single_function)
    if signature and _valid_signature(signature):
      return signature
  return None


def canonicalize_signatures(signatures):
  """Converts `signatures` into a dictionary of concrete functions."""
  if signatures is None:
    return {}, {}, {}
  if not isinstance(signatures, collections_abc.Mapping):
    signatures = {
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signatures
    }
  num_normalized_signatures_counter = 0
  concrete_signatures = {}
  wrapped_functions = {}
  defaults = {}
  for signature_key, function in signatures.items():
    original_function = signature_function = _get_signature(function)
    if signature_function is None:
      raise ValueError(
          "Expected a TensorFlow function for which to generate a signature, "
          f"but got {function}. Only `tf.functions` with an input signature or "
          "concrete functions can be used as a signature."
      )

    wrapped_functions[original_function] = signature_function = (
        wrapped_functions.get(original_function)
        or function_serialization.wrap_cached_variables(original_function)
    )
    _validate_inputs(signature_function)
    if num_normalized_signatures_counter < _NUM_DISPLAY_NORMALIZED_SIGNATURES:
      signature_name_changes = _get_signature_name_changes(signature_function)
      if signature_name_changes:
        num_normalized_signatures_counter += 1
        logging.info(
            "Function `%s` contains input name(s) %s with unsupported "
            "characters which will be renamed to %s in the SavedModel.",
            compat.as_str(signature_function.graph.name),
            ", ".join(signature_name_changes.keys()),
            ", ".join(signature_name_changes.values()),
        )

    # Re-wrap the function so that it returns a dictionary of Tensors. This
    # matches the format of 1.x-style signatures.
    # pylint: disable=cell-var-from-loop
    def signature_wrapper(**kwargs):
      structured_outputs = signature_function(**kwargs)
      return _normalize_outputs(
          structured_outputs, signature_function.name, signature_key
      )

    if hasattr(function, "__name__"):
      signature_wrapper.__name__ = "signature_wrapper_" + function.__name__
    wrapped_function = def_function.function(signature_wrapper)
    tensor_spec_signature = {}
    if signature_function.structured_input_signature is not None:
      # The structured input signature may contain other non-tensor arguments.
      inputs = filter(
          lambda x: isinstance(x, tensor_spec.TensorSpec),
          nest.flatten(
              signature_function.structured_input_signature,
              expand_composites=True,
          ),
      )
    else:
      # Structured input signature isn't always defined for some functions.
      inputs = signature_function.inputs

    for keyword, inp in zip(
        signature_function._arg_keywords,  # pylint: disable=protected-access
        inputs,
    ):
      keyword = compat.as_str(keyword)
      if isinstance(inp, tensor_spec.TensorSpec):
        spec = tensor_spec.TensorSpec(inp.shape, inp.dtype, name=keyword)
      else:
        spec = tensor_spec.TensorSpec.from_tensor(inp, name=keyword)
      tensor_spec_signature[keyword] = spec
    final_concrete = wrapped_function._get_concrete_function_garbage_collected(  # pylint: disable=protected-access
        **tensor_spec_signature
    )
    # pylint: disable=protected-access
    if len(final_concrete._arg_keywords) == 1:
      # If there is only one input to the signature, a very common case, then
      # ordering is unambiguous and we can let people pass a positional
      # argument. Since SignatureDefs are unordered (protobuf "map") multiple
      # arguments means we need to be keyword-only.
      final_concrete._num_positional_args = 1
    else:
      final_concrete._num_positional_args = 0
    # pylint: enable=protected-access
    concrete_signatures[signature_key] = final_concrete
    # pylint: enable=cell-var-from-loop
    if isinstance(function, core.GenericFunction):
      flattened_defaults = nest.flatten(
          function._function_spec.fullargspec.defaults  # pylint: disable=protected-access
      )
      len_default = len(flattened_defaults or [])
      arg_names = list(tensor_spec_signature.keys())
      if len_default > 0:
        # tensor_spec_signature uses the same nest.flatten() as
        # flattened_defaults.
        for arg, default in zip(
            arg_names[-len_default:],  # pylint: disable=protected-access
            flattened_defaults or [],
        ):
          if not isinstance(default, ops.Tensor):
            continue
          defaults.setdefault(signature_key, {})[arg] = default
  return concrete_signatures, wrapped_functions, defaults


def _normalize_outputs(outputs, function_name, signature_key):
  """Normalize outputs if necessary and check that they are tensors."""
  # Convert `outputs` to a dictionary (if it's not one already).
  if not isinstance(outputs, collections_abc.Mapping):
    # Check if `outputs` is a namedtuple.
    if hasattr(outputs, "_asdict"):
      outputs = outputs._asdict()
    else:
      if not isinstance(outputs, collections_abc.Sequence):
        outputs = [outputs]
      outputs = {
          "output_{}".format(output_index): output
          for output_index, output in enumerate(outputs)
      }

  # Check that the keys of `outputs` are strings and the values are Tensors.
  for key, value in outputs.items():
    if not isinstance(key, compat.bytes_or_text_types):
      raise ValueError(
          f"Got a dictionary with a non-string key {key!r} in the output of "
          f"the function {compat.as_str_any(function_name)} used to generate "
          f"the SavedModel signature {signature_key!r}."
      )
    if not isinstance(value, (ops.Tensor, composite_tensor.CompositeTensor)):
      raise ValueError(
          f"Got a non-Tensor value {value!r} for key {key!r} in the output of "
          f"the function {compat.as_str_any(function_name)} used to generate "
          f"the SavedModel signature {signature_key!r}. "
          "Outputs for functions used as signatures must be a single Tensor, "
          "a sequence of Tensors, or a dictionary from string to Tensor."
      )
  return outputs


# _SignatureMap is immutable to ensure that users do not expect changes to be
# reflected in the SavedModel. Using public APIs, tf.saved_model.load() is the
# only way to create a _SignatureMap and there is no way to modify it. So we can
# safely ignore/overwrite ".signatures" attributes attached to objects being
# saved if they contain a _SignatureMap. A ".signatures" attribute containing
# any other type (e.g. a regular dict) will raise an exception asking the user
# to first "del obj.signatures" if they want it overwritten.
class _SignatureMap(collections_abc.Mapping, base.Trackable):
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

  def _trackable_children(self, save_type=base.SaveType.CHECKPOINT, **kwargs):
    if save_type != base.SaveType.SAVEDMODEL:
      return {}

    return {
        key: value
        for key, value in self.items()
        if isinstance(value, (def_function.Function, defun.ConcreteFunction))
    }


revived_types.register_revived_type(
    "signature_map",
    lambda obj: isinstance(obj, _SignatureMap),
    versions=[
        revived_types.VersionedTypeRegistration(
            # Standard dependencies are enough to reconstruct the trackable
            # items in dictionaries, so we don't need to save any extra
            # information.
            object_factory=lambda proto: _SignatureMap(),
            version=1,
            min_producer_version=1,
            min_consumer_version=1,
            setter=_SignatureMap._add_signature,  # pylint: disable=protected-access
        )
    ],
)


def create_signature_map(signatures):
  """Creates an object containing `signatures`."""
  signature_map = _SignatureMap()
  for name, func in signatures.items():
    # This true of any signature that came from canonicalize_signatures. Here as
    # a sanity check on saving; crashing on load (e.g. in _add_signature) would
    # be more problematic in case future export changes violated these
    # assertions.
    assert isinstance(func, defun.ConcreteFunction)
    assert isinstance(func.structured_outputs, collections_abc.Mapping)
    # pylint: disable=protected-access
    if len(func._arg_keywords) == 1:
      assert 1 == func._num_positional_args
    else:
      assert 0 == func._num_positional_args
    signature_map._add_signature(name, func)
    # pylint: enable=protected-access
  return signature_map


def validate_augmented_graph_view(augmented_graph_view):
  """Performs signature-related sanity checks on `augmented_graph_view`."""
  for name, dep in augmented_graph_view.list_children(
      augmented_graph_view.root
  ):
    if name == SIGNATURE_ATTRIBUTE_NAME:
      if not isinstance(dep, _SignatureMap):
        raise ValueError(
            f"Exporting an object {augmented_graph_view.root} which has an"
            f" attribute named '{SIGNATURE_ATTRIBUTE_NAME}'. This is a reserved"
            " attribute used to store SavedModel signatures in objects which"
            " come from `tf.saved_model.load`. Delete this attribute (e.g."
            f" `del obj.{SIGNATURE_ATTRIBUTE_NAME}`) before saving if this"
            " shadowing is acceptable."
        )
      break
