# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Export utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.session_bundle import exporter
from tensorflow.contrib.session_bundle import gc
from tensorflow.python.client import session as tf_session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import saver as tf_saver


def _get_first_op_from_collection(collection_name):
  """Get first element from the collection."""
  elements = ops.get_collection(collection_name)
  if elements is not None:
    if elements:
      return elements[0]
  return None


def _get_saver():
  """Lazy init and return saver."""
  saver = _get_first_op_from_collection(ops.GraphKeys.SAVERS)
  if saver is not None:
    if saver:
      saver = saver[0]
    else:
      saver = None
  if saver is None and variables.all_variables():
    saver = tf_saver.Saver()
    ops.add_to_collection(ops.GraphKeys.SAVERS, saver)
  return saver


def _export_graph(graph, saver, checkpoint_path, export_dir,
                  default_graph_signature, named_graph_signatures,
                  exports_to_keep):
  """Exports graph via session_bundle, by creating a Session."""
  with graph.as_default():
    with tf_session.Session('') as session:
      variables.initialize_local_variables()
      data_flow_ops.initialize_all_tables()
      saver.restore(session, checkpoint_path)

      export = exporter.Exporter(saver)
      export.init(init_op=control_flow_ops.group(
          variables.initialize_local_variables(),
          data_flow_ops.initialize_all_tables()),
                  default_graph_signature=default_graph_signature,
                  named_graph_signatures=named_graph_signatures)
      export.export(export_dir, contrib_variables.get_global_step(), session,
                    exports_to_keep=exports_to_keep)


def _generic_signature_fn(examples, unused_features, predictions):
  """Creates generic signature from given examples and predictions.

  This is neeed for backward compatibility with default behaviour of
  export_estimator.

  Args:
    examples: `Tensor`.
    unused_features: `dict` of `Tensor`s.
    predictions: `dict` of `Tensor`s.

  Returns:
    Tuple of default signature and named signature.
  """
  tensors = {'inputs': examples}
  if not isinstance(predictions, dict):
    predictions = {'outputs': predictions}
  tensors.update(predictions)
  default_signature = exporter.generic_signature(tensors)
  return default_signature, {}


# pylint: disable=protected-access
def _default_input_fn(estimator, examples):
  """Creates default input parsing using Estimator's feature signatures."""
  return estimator._get_feature_ops_from_example(examples)


def export_estimator(estimator,
                     export_dir,
                     input_fn=_default_input_fn,
                     signature_fn=None,
                     default_batch_size=1,
                     exports_to_keep=None):
  """Exports inference graph into given dir.

  Args:
    estimator: Estimator to export
    export_dir: A string containing a directory to write the exported graph
      and checkpoints.
    input_fn: Function that given `Tensor` of `Example` strings, parses it into
      features that are then passed to the model.
    signature_fn: Function that given `Tensor` of `Example` strings,
      `dict` of `Tensor`s for features and `dict` of `Tensor`s for predictions
      and returns default and named exporting signautres.
    default_batch_size: Default batch size of the `Example` placeholder.
    exports_to_keep: Number of exports to keep.
  """
  checkpoint_path = tf_saver.latest_checkpoint(estimator._model_dir)
  with ops.Graph().as_default() as g:
    contrib_variables.create_global_step(g)
    examples = array_ops.placeholder(dtype=dtypes.string,
                                     shape=[default_batch_size],
                                     name='input_example_tensor')
    features = input_fn(estimator, examples)
    predictions = estimator._get_predict_ops(features)
    if signature_fn:
      default_signature, named_graph_signatures = signature_fn(examples,
                                                               features,
                                                               predictions)
    else:
      default_signature, named_graph_signatures = _generic_signature_fn(
          examples, features, predictions)
    if exports_to_keep is not None:
      exports_to_keep = gc.largest_export_versions(exports_to_keep)
    _export_graph(g, _get_saver(), checkpoint_path, export_dir,
                  default_graph_signature=default_signature,
                  named_graph_signatures=named_graph_signatures,
                  exports_to_keep=exports_to_keep)
# pylint: enable=protected-access
