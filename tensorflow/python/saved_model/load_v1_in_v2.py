# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Import a TF v1-style SavedModel when executing eagerly."""

import functools

from tensorflow.python.eager import context
from tensorflow.python.eager import lift_to_graph
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import function_deserialization
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import signature_serialization
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import resource
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.util import nest

# API label for SavedModel metrics.
_LOAD_V1_V2_LABEL = "load_v1_in_v2"


class _Initializer(resource.CapturableResource):
  """Represents an initialization operation restored from a SavedModel.

  Without this object re-export of imported 1.x SavedModels would omit the
  original SavedModel's initialization procedure.

  Created when `tf.saved_model.load` loads a TF 1.x-style SavedModel with an
  initialization op. This object holds a function that runs the
  initialization. It does not require any manual user intervention;
  `tf.saved_model.save` will see this object and automatically add it to the
  exported SavedModel, and `tf.saved_model.load` runs the initialization
  function automatically.
  """

  def __init__(self, init_fn, asset_paths):
    super(_Initializer, self).__init__()
    self._asset_paths = asset_paths
    self._init_fn = init_fn

  def _create_resource(self):
    # Return a constant here so that when re-saved, the traced `create_resource`
    # has valid returns.
    return constant_op.constant(1.0)

  def _initialize(self):
    return self._init_fn(*[path.asset_path for path in self._asset_paths])


class _EagerSavedModelLoader(loader_impl.SavedModelLoader):
  """Loads a SavedModel without using Sessions."""

  def get_meta_graph_def_from_tags(self, tags):
    """Override to support implicit one-MetaGraph loading with tags=None."""
    if tags is None:
      if len(self._saved_model.meta_graphs) != 1:
        tag_sets = [
            mg.meta_info_def.tags for mg in self._saved_model.meta_graphs
        ]
        raise ValueError(
            "Importing a SavedModel with `tf.saved_model.load` requires a "
            "`tags=` argument if there is more than one MetaGraph. Got "
            f"`tags=None`, but there are {len(self._saved_model.meta_graphs)} "
            f"MetaGraphs in the SavedModel with tag sets: {tag_sets}. Pass a "
            "`tags=` argument to load this SavedModel."
        )
      return self._saved_model.meta_graphs[0]
    return super(_EagerSavedModelLoader, self).get_meta_graph_def_from_tags(
        tags
    )

  def load_graph(self, returns, meta_graph_def):
    """Called from wrap_function to import `meta_graph_def`."""
    # pylint: disable=protected-access
    saver, _ = tf_saver._import_meta_graph_with_return_elements(meta_graph_def)
    # pylint: enable=protected-access
    returns[0] = saver

  def _extract_saver_restore(self, wrapped, saver):
    if saver is None:
      return None
    saver_def = saver.saver_def
    filename_tensor = wrapped.graph.as_graph_element(
        saver_def.filename_tensor_name
    )
    # We both feed and fetch filename_tensor so we have an operation to use to
    # feed into variable initializers (only relevant for v1 graph building).
    return wrapped.prune(
        feeds=[filename_tensor],
        fetches=[
            filename_tensor,
            wrapped.graph.as_graph_element(saver_def.restore_op_name),
        ],
    )

  def restore_variables(self, wrapped, restore_from_saver):
    """Restores variables from the checkpoint."""
    if restore_from_saver is not None:
      initializer, _ = restore_from_saver(
          constant_op.constant(self._variables_path)
      )
      if not ops.executing_eagerly_outside_functions():
        # Add the initialization operation to the "saved_model_initializers"
        # collection in case we don't have any lifted variables to attach it to.
        ops.add_to_collection("saved_model_initializers", initializer)
        one_unlifted = False

        for variable in wrapped.graph.get_collection_ref(
            ops.GraphKeys.GLOBAL_VARIABLES
        ):
          if variable.graph is wrapped.graph:
            one_unlifted = True
          # pylint: disable=protected-access
          variable._initializer_op = initializer
          # pylint: enable=protected-access
        if one_unlifted:
          logging.warning(
              "Some variables could not be lifted out of a loaded function. "
              "Please run "
              '`sess.run(tf.get_collection("saved_model_initializers"))`to '
              "restore these variables."
          )

  def _extract_signatures(self, wrapped, meta_graph_def):
    """Creates ConcreteFunctions for signatures in `meta_graph_def`."""
    signature_functions = {}
    for signature_key, signature_def in meta_graph_def.signature_def.items():
      if signature_def.inputs:
        input_items = sorted(
            signature_def.inputs.items(), key=lambda item: item[0]
        )
        original_input_names, input_specs = zip(*input_items)
      else:
        original_input_names = []
        input_specs = []
      # TODO(b/205015292): Support optional arguments
      feeds = [
          wrap_function._get_element_from_tensor_info(input_spec, wrapped.graph)  # pylint: disable=protected-access
          for input_spec in input_specs
      ]
      input_names = []
      input_tensors = []
      for original_input_name, feed in zip(original_input_names, feeds):
        if isinstance(feed, sparse_tensor.SparseTensor):
          # We have to give explicit name for SparseTensor arguments, because
          # these are not present in the TensorInfo.
          indices_name = "%s_indices" % original_input_name
          values_name = "%s_values" % original_input_name
          dense_shape_name = "%s_dense_shape" % original_input_name
          input_names.extend([indices_name, values_name, dense_shape_name])
          input_tensors.extend([feed.indices, feed.values, feed.dense_shape])
        elif isinstance(feed, composite_tensor.CompositeTensor):
          component_tensors = nest.flatten(feed, expand_composites=True)
          input_names.extend(
              "%s_component_%d" % (original_input_name, n)
              for n in range(len(component_tensors))
          )
          input_tensors.extend(component_tensors)
        else:
          input_names.append(original_input_name)
          input_tensors.append(feed)
      fetches = {name: out for name, out in signature_def.outputs.items()}
      try:
        signature_fn = wrapped.prune(feeds=feeds, fetches=fetches)
      except lift_to_graph.UnliftableError as ex:
        # Mutate the exception to add a bit more detail.
        args = ex.args
        if not args:
          message = ""
        else:
          message = args[0]
        message = (
            "A SavedModel signature needs an input for each placeholder the "
            "signature's outputs use. An output for signature '{}' depends on "
            "a placeholder which is not an input (i.e. the placeholder is not "
            "fed a value).\n\n"
        ).format(signature_key) + message
        ex.args = (message,) + args[1:]
        raise
      # pylint: disable=protected-access
      signature_fn._arg_keywords = input_names
      signature_fn._func_graph.structured_input_signature = (
          (),
          func_graph.convert_structure_to_signature(
              dict(zip(input_names, input_tensors))
          ),
      )

      if len(input_names) == 1:
        # Allowing positional arguments does not create any ambiguity if there's
        # only one.
        signature_fn._num_positional_args = 1
      else:
        signature_fn._num_positional_args = 0
      # pylint: enable=protected-access
      signature_functions[signature_key] = signature_fn
    return signature_functions

  def load(self, tags):
    """Creates an object from the MetaGraph identified by `tags`."""
    meta_graph_def = self.get_meta_graph_def_from_tags(tags)
    load_shared_name_suffix = "_load_{}".format(ops.uid())
    functions = function_deserialization.load_function_def_library(
        meta_graph_def.graph_def.library,
        load_shared_name_suffix=load_shared_name_suffix,
    )
    # Replace existing functions in the MetaGraphDef with renamed functions so
    # we don't have duplicates or name collisions.
    meta_graph_def.graph_def.library.Clear()
    for function in functions.values():
      meta_graph_def.graph_def.library.function.add().CopyFrom(
          function.function_def
      )
    # We've renamed functions and shared names. We need the same operation on
    # the GraphDef itself for consistency.
    for node_def in meta_graph_def.graph_def.node:
      function_deserialization.fix_node_def(
          node_def, functions, load_shared_name_suffix
      )

    load_graph_returns = [None]
    wrapped = wrap_function.wrap_function(
        functools.partial(self.load_graph, load_graph_returns, meta_graph_def),
        signature=[],
    )
    (saver,) = load_graph_returns
    restore_from_saver = self._extract_saver_restore(wrapped, saver)
    self.restore_variables(wrapped, restore_from_saver)
    with wrapped.graph.as_default():
      init_op = (
          loader_impl.get_init_op(meta_graph_def)
          or monitored_session.Scaffold.default_local_init_op()
      )
      # Add a dummy Tensor we know we can fetch to add control dependencies to.
      init_anchor = constant_op.constant(0.0, name="dummy_fetch")

    root = autotrackable.AutoTrackable()
    if restore_from_saver is not None:
      root.restore = lambda path: restore_from_saver(constant_op.constant(path))
    asset_feed_tensors = []
    asset_paths = []
    for tensor_name, value in loader_impl.get_asset_tensors(
        self._export_dir, meta_graph_def
    ).items():
      asset_feed_tensors.append(wrapped.graph.as_graph_element(tensor_name))
      asset_paths.append(asset.Asset(value))
    init_fn = wrapped.prune(
        feeds=asset_feed_tensors,
        fetches=[init_anchor, wrapped.graph.as_graph_element(init_op)],
    )
    initializer = _Initializer(init_fn, asset_paths)
    # pylint: disable=protected-access
    local_init_op, _ = initializer._initialize()
    # pylint: enable=protected-access
    with ops.init_scope():
      if not context.executing_eagerly():
        ops.add_to_collection(ops.GraphKeys.TABLE_INITIALIZERS, local_init_op)
        for variable in wrapped.graph.get_collection_ref(
            ops.GraphKeys.LOCAL_VARIABLES
        ):
          # pylint: disable=protected-access
          variable._initializer_op = local_init_op
          # pylint: enable=protected-access
    root.initializer = initializer
    root.asset_paths = asset_paths
    signature_functions = self._extract_signatures(wrapped, meta_graph_def)

    root.signatures = signature_serialization.create_signature_map(
        signature_functions
    )
    root.variables = list(wrapped.graph.variables)
    root.tensorflow_version = meta_graph_def.meta_info_def.tensorflow_version
    root.tensorflow_git_version = (
        meta_graph_def.meta_info_def.tensorflow_git_version
    )
    root.graph = wrapped.graph
    root.prune = wrapped.prune
    return root


def load(export_dir, tags):
  """Load a v1-style SavedModel as an object."""
  metrics.IncrementReadApi(_LOAD_V1_V2_LABEL)
  loader = _EagerSavedModelLoader(export_dir)
  result = loader.load(tags=tags)
  metrics.IncrementRead(write_version="1")
  return result
