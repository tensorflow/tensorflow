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
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import saved_object_graph_pb2
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils_impl
from tensorflow.python.training.checkpointable import base
from tensorflow.python.training.checkpointable import util
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


def _check_for_functional_keras_model(root):
  """Makes an export signature for `root` if it's a functional Keras Model."""
  # If nothing is decorated yet but this is a functional Keras Model (duck
  # typed), we'll try to make a signature ourselves.
  try:
    inputs = root.inputs
    input_names = root.input_names
  except AttributeError:
    return None
  input_signature = []
  for input_tensor, input_name in zip(inputs, input_names):
    input_signature.append(tensor_spec.TensorSpec(
        shape=input_tensor.shape, dtype=input_tensor.dtype,
        name=input_name))

  @def_function.function(input_signature=input_signature)
  def _wrapped_model(*args):
    outputs_list = nest.flatten(root(inputs=list(args)))
    return {name: output for name, output
            in zip(root.output_names, outputs_list)}
  return _wrapped_model


def _find_function_to_export(root):
  """Iterate over `root`'s attributes, finding traced functions."""
  exported_function = None
  previous_attribute_name = None
  for attribute_name in dir(root):
    attribute_value = getattr(root, attribute_name, None)
    if isinstance(attribute_value, def_function.PolymorphicFunction):
      if exported_function is not None:
        raise ValueError(
            ("Exporting an object with no "
             "tf.saved_model.save(..., signatures=...) "
             "argument specified, and with more than one "
             "@tf.function-decorated method attached to it: {}. The signature "
             "keys for these functions are ambiguous. Specify signature "
             "functions explicitly.").format(
                 [previous_attribute_name, attribute_name]))
      exported_function = attribute_value
      previous_attribute_name = attribute_name
  if exported_function is None:
    exported_function = _check_for_functional_keras_model(root)
  if exported_function is None:
    raise ValueError(
        ("Exporting an object with no tf.saved_model.save(..., signatures=...) "
         "argument specified, and with no @tf.function-decorated methods "
         "attached to it. In the future this will be a supported use-case for "
         "Python re-import, but at the moment saving a SavedModel without "
         "signatures does not make sense, as the only consumers will expect "
         "signatures. Either decorate a method or specify a signature function "
         "explicitly."))
  return exported_function


def _canonicalize_signatures(signatures):
  """Converts `signatures` into a dictionary of concrete functions."""
  if not isinstance(signatures, collections.Mapping):
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
  return {key: utils_impl.build_tensor_info(value)
          for key, value in tensor_dict.items()}


def _map_captured_resources_to_created_resources(
    original_captures, resource_map):
  """Maps eager resources captured by a function to Graph resources for export.

  Args:
    original_captures: A dictionary mapping from resource tensors captured by
      the function to interior placeholders for those resources (inside the
      function body).
    resource_map: A dictionary mapping from resource tensors owned by the eager
      context to resource tensors in the exported graph.

  Returns:
    A dictionary mapping from interior placeholders in the function body to
    exterior stand-in resource tensors which belong to the exported graph.

  Raises:
    AssertionError: If the function references a resource which is not part of
      `resource_map`.
  """
  export_captures = {}
  for exterior, interior in original_captures.items():
    mapped_resource = resource_map.get(exterior, None)
    if mapped_resource is None:
      raise AssertionError(
          ("Tried to export a function which references untracked stateful "
           "object {}. Stateful TensorFlow objects (e.g. tf.Variable) must "
           "be tracked by the main object. Objects may be tracked by "
           "assigning them to an attribute of another tracked object, or to "
           "an attribute of the main object directly.")
          .format(interior))
    export_captures[interior] = mapped_resource
  return export_captures


def _map_function_inputs_to_created_inputs(
    function_inputs, export_captures, signature_key, function_name):
  """Creates exterior placeholders in the exported graph for function inputs.

  Functions have two types of inputs: tensors captured from the outside (eager)
  context, and arguments to the function which we expect to receive from the
  user at each call. `_map_captured_resources_to_created_resources` replaces
  captured tensors with stand-ins (typically these are resource dtype tensors
  associated with variables). `_map_function_inputs_to_created_inputs` runs over
  every input, either captured or argument. For captures, it uses the mapped
  resource from `export_captures`. For arguments, it creates a new placeholder
  which will belong to the exported graph rather than the function body.

  Args:
    function_inputs: A list of all placeholders in the function body.
    export_captures: A dictionary mapping from interior placeholders in the
      function body to exterior stand-in resource tensors which belong to the
      exported graph (see `_map_captured_resources_to_created_resources`).
    signature_key: The name of the signature being exported, for error messages.
    function_name: The name of the function, for error messages.

  Returns:
    A tuple of (mapped_inputs, exterior_placeholders)
      mapped_inputs: A list with entries corresponding to `function_inputs`
        containing all of the inputs of the function gathered from the exported
        graph (both captured resources and arguments).
      exterior_argument_placeholders: A dictionary mapping from argument names
        to placeholders in the exported graph, containing the explicit arguments
        to the function which a user is expected to provide.

  Raises:
    ValueError: If argument names are not unique.
  """
  # `exterior_argument_placeholders` holds placeholders which are outside the
  # function body, directly contained in a MetaGraph of the SavedModel. The
  # function body itself contains nearly identical placeholders used when
  # running the function, but these exterior placeholders allow Session-based
  # APIs to call the function using feeds and fetches which name Tensors in the
  # MetaGraph.
  exterior_argument_placeholders = {}
  mapped_inputs = []
  for placeholder in function_inputs:
    mapped_resource_tensor = export_captures.get(placeholder, None)
    if mapped_resource_tensor is not None:
      # This is a captured resource.
      mapped_inputs.append(mapped_resource_tensor)
      continue
    # `export_captures` contains an exhaustive set of captures, so if we don't
    # find the input there then we now know we have an argument.
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
          .format(signature_key, compat.as_str_any(function_name),
                  user_input_name))
    arg_placeholder = array_ops.placeholder(
        shape=placeholder.shape,
        dtype=placeholder.dtype,
        name="{}_{}".format(signature_key, user_input_name))
    exterior_argument_placeholders[user_input_name] = arg_placeholder
    mapped_inputs.append(arg_placeholder)
  return mapped_inputs, exterior_argument_placeholders


def _generate_signatures(signature_functions, resource_map):
  """Validates and calls `signature_functions` in the default graph.

  Args:
    signature_functions: A dictionary mapping string keys to concrete TensorFlow
      functions (e.g. from `_canonicalize_signatures`) which will be used to
      generate SignatureDefs.
    resource_map: A dictionary mapping from resource tensors in the eager
      context to resource tensors in the Graph being exported. This dictionary
      is used to re-bind resources captured by functions to tensors which will
      exist in the SavedModel.

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
    # Register the inference function for this signature in the exported
    # graph. There is no direct use for the gradient of this function, so we
    # don't generate/register a gradient function here (but may end up with one
    # if another function relies on it). Users can still take symbolic gradients
    # of the function on import, the gradient just won't be in the saved
    # graph. When exporting a signature which already computes gradients, this
    # stops us from taking needless second-order gradients.
    func.add_to_graph(register_gradient_functions=False)
    export_captures = _map_captured_resources_to_created_resources(
        func.graph.captures, resource_map)
    mapped_inputs, exterior_argument_placeholders = (
        _map_function_inputs_to_created_inputs(
            func.inputs, export_captures, signature_key, func.name))
    # Calls the function quite directly, since we have new captured resource
    # tensors we need to feed in which weren't part of the original function
    # definition.
    # pylint: disable=protected-access
    outputs = _normalize_outputs(
        func._build_call_outputs(
            func._inference_function.call(context.context(), mapped_inputs)),
        func.name, signature_key)
    # pylint: enable=protected-access
    signatures[signature_key] = signature_def_utils.build_signature_def(
        _tensor_dict_to_tensorinfo(exterior_argument_placeholders),
        _tensor_dict_to_tensorinfo(outputs))
  return signatures


def _map_resources(accessible_objects):
  """Makes new resource handle ops corresponding to existing resource tensors.

  Creates resource handle ops in the current default graph, whereas
  `accessible_objects` will be from an eager context. Resource mapping adds
  resource handle ops to the main GraphDef of a SavedModel, which allows the C++
  loader API to interact with variables.

  Args:
    accessible_objects: A list of objects, some of which may contain resources,
      to create replacements for.

  Returns:
    A tuple of (object_map, resource_map):
      object_map: A dictionary mapping from object in `accessible_objects` to
        replacement objects created to hold the new resource tensors.
      resource_map: A dictionary mapping from resource tensors extracted from
        `accessible_objects` to newly created resource tensors.
  """
  # TODO(allenl, rohanj): Map generic resources rather than just variables.
  # TODO(allenl): Handle MirroredVariables and other types of variables which
  # may need special casing.
  object_map = {}
  resource_map = {}
  for obj in accessible_objects:
    if resource_variable_ops.is_resource_variable(obj):
      new_variable = resource_variable_ops.copy_to_graph_uninitialized(obj)
      object_map[obj] = new_variable
      resource_map[obj.handle] = new_variable.handle
  return object_map, resource_map


def _make_graph_def(root, signature_functions, object_saver):
  """Generates and exports call ops for `signature_functions`."""
  signatures = {}
  # List objects from the eager context to make sure Optimizers give us the
  # right Graph-dependent variables.
  accessible_objects = util.list_objects(root)
  exported_graph = ops.Graph()
  with exported_graph.as_default():
    object_map, resource_map = _map_resources(accessible_objects)
  # Saving an object-based checkpoint again gathers variables. We need to do the
  # gathering from the eager context so Optimizers save the right set of
  # variables, but want any operations associated with the save/restore to be in
  # the exported graph (thus the `to_graph` argument).
  saver = object_saver.freeze(object_map=object_map, to_graph=exported_graph)
  with exported_graph.as_default():
    signatures = _generate_signatures(signature_functions, resource_map)
    saver_def = saver.to_proto()
  graph_def = exported_graph.as_graph_def(add_shapes=True)
  # Clean reference cycles so repeated export()s don't make work for the garbage
  # collector.
  ops.dismantle_graph(exported_graph)
  return graph_def, signatures, saver_def


def _write_object_graph(obj, export_dir):
  """Save a SavedObjectGraph proto for `obj`."""
  # SavedObjectGraph is similar to the CheckpointableObjectGraph proto in the
  # checkpoint. It will eventually go into the SavedModel.
  object_proto = util.make_object_graph_without_attributes(
      obj, proto=saved_object_graph_pb2.SavedObjectGraph())
  extra_asset_dir = os.path.join(
      compat.as_bytes(export_dir),
      compat.as_bytes(constants.EXTRA_ASSETS_DIRECTORY))
  file_io.recursive_create_dir(extra_asset_dir)
  object_graph_filename = os.path.join(
      extra_asset_dir, compat.as_bytes("object_graph.pb"))
  file_io.write_string_to_file(object_graph_filename,
                               object_proto.SerializeToString())


@tf_export("saved_model.save", v1=["saved_model.experimental.save"])
def save(obj, export_dir, signatures=None):
  # pylint: disable=line-too-long
  """Exports the Checkpointable object `obj` to [SavedModel format](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md).

  Example usage:

  ```python
  class Adder(tf.train.Checkpoint):

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def add(self, x):
      return x + x + 1.

  to_export = Adder()
  tf.saved_model.save(to_export, '/tmp/adder')
  ```

  The resulting SavedModel is then servable with an input named "x", its value
  having any shape and dtype float32.

  The optional `signatures` argument controls which methods in `obj` will be
  available to programs which consume `SavedModel`s, for example serving
  APIs. Python functions may be decorated with
  `@tf.function(input_signature=...)` and passed as signatures directly, or
  lazily with a call to `get_concrete_function` on the method decorated with
  `@tf.function`.

  If the `signatures` argument is omitted, `obj` will be searched for
  `@tf.function`-decorated methods. If exactly one `@tf.function` is found, that
  method will be used as the default signature for the SavedModel. This behavior
  is expected to change in the future, when a corresponding
  `tf.saved_model.load` symbol is added. At that point signatures will be
  completely optional, and any `@tf.function` attached to `obj` or its
  dependencies will be exported for use with `load`.

  When invoking a signature in an exported SavedModel, `Tensor` arguments are
  identified by name. These names will come from the Python function's argument
  names by default. They may be overridden by specifying a `name=...` argument
  in the corresponding `tf.TensorSpec` object. Explicit naming is required if
  multiple `Tensor`s are passed through a single argument to the Python
  function.

  The outputs of functions used as `signatures` must either be flat lists, in
  which case outputs will be numbered, or a dictionary mapping string keys to
  `Tensor`, in which case the keys will be used to name outputs.

  Since `tf.keras.Model` objects are also Checkpointable, this function can be
  used to export Keras models. For example, exporting with a signature
  specified:

  ```python
  class Model(tf.keras.Model):

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def serve(self, serialized):
      ...

  m = Model()
  tf.saved_model.save(m, '/tmp/saved_model/')
  ```

  Exporting from a function without a fixed signature:

  ```python
  class Model(tf.keras.Model):

    @tf.function
    def call(self, x):
      ...

  m = Model()
  tf.saved_model.save(
      m, '/tmp/saved_model/',
      signatures=m.call.get_concrete_function(
          tf.TensorSpec(shape=[None, 3], dtype=tf.float32, name="inp")))
  ```

  `tf.keras.Model` instances constructed from inputs and outputs already have a
  signature and so do not require a `@tf.function` decorator or a `signatures`
  argument. If neither are specified, the model's forward pass is exported.

  ```python
  x = input_layer.Input((4,), name="x")
  y = core.Dense(5, name="out")(x)
  model = training.Model(x, y)
  tf.saved_model.save(model, '/tmp/saved_model/')
  # The exported SavedModel takes "x" with shape [None, 4] and returns "out"
  # with shape [None, 5]
  ```

  Variables must be tracked by assigning them to an attribute of a tracked
  object or to an attribute of `obj` directly. TensorFlow objects (e.g. layers
  from `tf.keras.layers`, optimizers from `tf.train`) track their variables
  automatically. This is the same tracking scheme that `tf.train.Checkpoint`
  uses, and an exported `Checkpoint` object may be restored as a training
  checkpoint by pointing `tf.train.Checkpoint.restore` to the SavedModel's
  "variables/" subdirectory. Currently variables are the only stateful objects
  supported by `tf.saved_model.save`, but others (e.g. tables) will be supported
  in the future.

  `tf.function` does not hard-code device annotations from outside the function
  body, instead using the calling context's device. This means for example that
  exporting a model which runs on a GPU and serving it on a CPU will generally
  work, with some exceptions. `tf.device` annotations inside the body of the
  function will be hard-coded in the exported model; this type of annotation is
  discouraged. Device-specific operations, e.g. with "cuDNN" in the name or with
  device-specific layouts, may cause issues. Currently a `DistributionStrategy`
  is another exception: active distribution strategies will cause device
  placements to be hard-coded in a function. Exporting a single-device
  computation and importing under a `DistributionStrategy` is not currently
  supported, but may be in the future.

  SavedModels exported with `tf.saved_model.save` [strip default-valued
  attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes)
  automatically, which removes one source of incompatibilities when the consumer
  of a SavedModel is running an older TensorFlow version than the
  producer. There are however other sources of incompatibilities which are not
  handled automatically, such as when the exported model contains operations
  which the consumer does not have definitions for.

  The current implementation of `tf.saved_model.save` targets serving use-cases,
  but omits information which will be necessary for the planned future
  implementation of `tf.saved_model.load`. Exported models using the current
  `save` implementation, and other existing SavedModels, will not be compatible
  with `tf.saved_model.load` when it is implemented. Further, `save` will in the
  future attempt to export `@tf.function`-decorated methods which it does not
  currently inspect, so some objects which are exportable today will raise
  exceptions on export in the future (e.g. due to complex/non-serializable
  default arguments). Such backwards-incompatible API changes are expected only
  prior to the TensorFlow 2.0 release.

  Args:
    obj: A checkpointable object to export.
    export_dir: A directory in which to write the SavedModel.
    signatures: Optional, either a `tf.function` with an input signature
      specified or the result of `f.get_concrete_function` on a
      `@tf.function`-decorated function `f`, in which case `f` will be used to
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
  if not isinstance(obj, base.CheckpointableBase):
    raise ValueError(
        "Expected a Checkpointable object for export, got {}.".format(obj))
  if signatures is None:
    # Note that we run this before saving the checkpoint, since looping over
    # attributes may have the side effect of creating variables in some cases.
    signatures = _find_function_to_export(obj)
  object_saver = util.CheckpointableSaver(obj)
  utils_impl.get_or_create_variables_dir(export_dir)
  object_saver.save(utils_impl.get_variables_path(export_dir))

  signatures = _canonicalize_signatures(signatures)
  graph_def, signatures, saver_def = _make_graph_def(
      obj, signatures, object_saver)
  saved_model = saved_model_pb2.SavedModel()
  saved_model.saved_model_schema_version = (
      constants.SAVED_MODEL_SCHEMA_VERSION)
  meta_graph_def = saved_model.meta_graphs.add()
  meta_graph_def.meta_info_def.tags.append(tag_constants.SERVING)
  meta_graph_def.saver_def.CopyFrom(saver_def)
  # TODO(allenl): Factor out some subset of SavedModelBuilder which is 2.x
  # compatible (no sessions) and share it with this export API rather than
  # making a SavedModel proto and writing it directly.
  meta_graph_def.graph_def.MergeFrom(graph_def)
  for signature_key, signature in signatures.items():
    meta_graph_def.signature_def[signature_key].MergeFrom(signature)
  meta_graph.strip_graph_default_valued_attrs(meta_graph_def)
  path = os.path.join(
      compat.as_bytes(export_dir),
      compat.as_bytes(constants.SAVED_MODEL_FILENAME_PB))
  file_io.write_string_to_file(path, saved_model.SerializeToString())
  _write_object_graph(obj, export_dir)
