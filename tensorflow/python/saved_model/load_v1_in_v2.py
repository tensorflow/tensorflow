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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import signature_serialization
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training.tracking import tracking


class _Initializer(tracking.TrackableResource):
  """Represents an initialization operation restored from a SavedModel.

  Without this object re-export of imported 1.x SavedModels would omit the
  original SavedModel's initialization procedure.

  Created when `tf.saved_model.load` loads a TF 1.x-style SavedModel with an
  initialization op. This object holds a function which runs the
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
    return array_ops.placeholder(
        dtype=dtypes.resource, shape=[], name="unused_resource")

  def _initialize(self):
    self._init_fn(*[path.asset_path for path in self._asset_paths])


class _EagerSavedModelLoader(loader_impl.SavedModelLoader):
  """Loads a SavedModel without using Sessions."""

  def get_meta_graph_def_from_tags(self, tags):
    """Override to support implicit one-MetaGraph loading with tags=None."""
    if tags is None:
      if len(self._saved_model.meta_graphs) != 1:
        tag_sets = [mg.meta_info_def.tags
                    for mg in self._saved_model.meta_graphs]
        raise ValueError(
            ("Importing a SavedModel with tf.saved_model.load requires a "
             "'tags=' argument if there is more than one MetaGraph. Got "
             "'tags=None', but there are {} MetaGraphs in the SavedModel with "
             "tag sets {}. Pass a 'tags=' argument to load this SavedModel.")
            .format(len(self._saved_model.meta_graphs), tag_sets))
      return self._saved_model.meta_graphs[0]
    return super(_EagerSavedModelLoader, self).get_meta_graph_def_from_tags(
        tags)

  def load_graph(self, returns, meta_graph_def):
    """Called from wrap_function to import `meta_graph_def`."""
    # pylint: disable=protected-access
    saver, _ = tf_saver._import_meta_graph_with_return_elements(
        meta_graph_def)
    # pylint: enable=protected-access
    returns[0] = saver

  def restore_variables(self, wrapped, saver):
    """Restores variables from the checkpoint."""
    if saver is not None:
      saver_def = saver.saver_def
      restore_fn = wrapped.prune(
          feeds=[wrapped.graph.as_graph_element(
              saver_def.filename_tensor_name)],
          fetches=[wrapped.graph.as_graph_element(saver_def.restore_op_name)])
      restore_fn(constant_op.constant(self._variables_path))

  def _extract_signatures(self, wrapped, meta_graph_def):
    """Creates ConcreteFunctions for signatures in `meta_graph_def`."""
    signature_functions = {}
    for signature_key, signature_def in meta_graph_def.signature_def.items():
      if signature_def.inputs:
        input_names, input_specs = zip(*signature_def.inputs.items())
      else:
        input_names = []
        input_specs = []
      # TODO(allenl): Support optional arguments
      signature_fn = wrapped.prune(
          feeds=[wrapped.graph.as_graph_element(inp.name)
                 for inp in input_specs],
          fetches={name: wrapped.graph.as_graph_element(out.name)
                   for name, out in signature_def.outputs.items()})
      # pylint: disable=protected-access
      signature_fn._arg_keywords = input_names
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
    load_graph_returns = [None]
    wrapped = wrap_function.wrap_function(
        functools.partial(self.load_graph, load_graph_returns, meta_graph_def),
        signature=[])
    saver, = load_graph_returns
    self.restore_variables(wrapped, saver)
    with wrapped.graph.as_default():
      init_op = loader_impl.get_init_op(meta_graph_def)
    root = tracking.AutoTrackable()
    if init_op is not None:
      asset_feed_tensors = []
      asset_paths = []
      for tensor_name, value in loader_impl.get_asset_tensors(
          self._export_dir, meta_graph_def).items():
        asset_feed_tensors.append(wrapped.graph.as_graph_element(tensor_name))
        asset_paths.append(tracking.TrackableAsset(value))
      init_fn = wrapped.prune(
          feeds=asset_feed_tensors,
          fetches=[wrapped.graph.as_graph_element(init_op)])
      initializer = _Initializer(init_fn, asset_paths)
      initializer._initialize()  # pylint: disable=protected-access
      root.initializer = initializer
      root.asset_paths = asset_paths
    else:
      root.asset_paths = []
    signature_functions = self._extract_signatures(wrapped, meta_graph_def)

    root.signatures = signature_serialization.create_signature_map(
        signature_functions)
    root.variables = list(wrapped.graph.variables)
    return root


def load(export_dir, tags):
  """Load a v1-style SavedModel as an object."""
  loader = _EagerSavedModelLoader(export_dir)
  return loader.load(tags=tags)
