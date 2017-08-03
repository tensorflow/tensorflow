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

from tensorflow.contrib.framework import deprecated
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.session_bundle import exporter
from tensorflow.contrib.session_bundle import gc
from tensorflow.python.client import session as tf_session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as tf_saver


@deprecated('2017-03-25', 'Please use Estimator.export_savedmodel() instead.')
def _get_first_op_from_collection(collection_name):
  """Get first element from the collection."""
  elements = ops.get_collection(collection_name)
  if elements is not None:
    if elements:
      return elements[0]
  return None


@deprecated('2017-03-25', 'Please use Estimator.export_savedmodel() instead.')
def _get_saver():
  """Lazy init and return saver."""
  saver = _get_first_op_from_collection(ops.GraphKeys.SAVERS)
  if saver is not None:
    if saver:
      saver = saver[0]
    else:
      saver = None
  if saver is None and variables.global_variables():
    saver = tf_saver.Saver()
    ops.add_to_collection(ops.GraphKeys.SAVERS, saver)
  return saver


@deprecated('2017-03-25', 'Please use Estimator.export_savedmodel() instead.')
def _export_graph(graph, saver, checkpoint_path, export_dir,
                  default_graph_signature, named_graph_signatures,
                  exports_to_keep):
  """Exports graph via session_bundle, by creating a Session."""
  with graph.as_default():
    with tf_session.Session('') as session:
      variables.local_variables_initializer()
      lookup_ops.tables_initializer()
      saver.restore(session, checkpoint_path)

      export = exporter.Exporter(saver)
      export.init(
          init_op=control_flow_ops.group(
              variables.local_variables_initializer(),
              lookup_ops.tables_initializer()),
          default_graph_signature=default_graph_signature,
          named_graph_signatures=named_graph_signatures,
          assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS))
      return export.export(export_dir, contrib_variables.get_global_step(),
                           session, exports_to_keep=exports_to_keep)


@deprecated('2017-03-25',
            'signature_fns are deprecated. For canned Estimators they are no '
            'longer needed. For custom Estimators, please return '
            'output_alternatives from your model_fn via ModelFnOps.')
def generic_signature_fn(examples, unused_features, predictions):
  """Creates generic signature from given examples and predictions.

  This is needed for backward compatibility with default behavior of
  export_estimator.

  Args:
    examples: `Tensor`.
    unused_features: `dict` of `Tensor`s.
    predictions: `Tensor` or `dict` of `Tensor`s.

  Returns:
    Tuple of default signature and empty named signatures.

  Raises:
    ValueError: If examples is `None`.
  """
  if examples is None:
    raise ValueError('examples cannot be None when using this signature fn.')

  tensors = {'inputs': examples}
  if not isinstance(predictions, dict):
    predictions = {'outputs': predictions}
  tensors.update(predictions)
  default_signature = exporter.generic_signature(tensors)
  return default_signature, {}


@deprecated('2017-03-25',
            'signature_fns are deprecated. For canned Estimators they are no '
            'longer needed. For custom Estimators, please return '
            'output_alternatives from your model_fn via ModelFnOps.')
def classification_signature_fn(examples, unused_features, predictions):
  """Creates classification signature from given examples and predictions.

  Args:
    examples: `Tensor`.
    unused_features: `dict` of `Tensor`s.
    predictions: `Tensor` or dict of tensors that contains the classes tensor
      as in {'classes': `Tensor`}.

  Returns:
    Tuple of default classification signature and empty named signatures.

  Raises:
    ValueError: If examples is `None`.
  """
  if examples is None:
    raise ValueError('examples cannot be None when using this signature fn.')

  if isinstance(predictions, dict):
    default_signature = exporter.classification_signature(
        examples, classes_tensor=predictions['classes'])
  else:
    default_signature = exporter.classification_signature(
        examples, classes_tensor=predictions)
  return default_signature, {}


@deprecated('2017-03-25',
            'signature_fns are deprecated. For canned Estimators they are no '
            'longer needed. For custom Estimators, please return '
            'output_alternatives from your model_fn via ModelFnOps.')
def classification_signature_fn_with_prob(
    examples, unused_features, predictions):
  """Classification signature from given examples and predicted probabilities.

  Args:
    examples: `Tensor`.
    unused_features: `dict` of `Tensor`s.
    predictions: `Tensor` of predicted probabilities or dict that contains the
      probabilities tensor as in {'probabilities', `Tensor`}.

  Returns:
    Tuple of default classification signature and empty named signatures.

  Raises:
    ValueError: If examples is `None`.
  """
  if examples is None:
    raise ValueError('examples cannot be None when using this signature fn.')

  if isinstance(predictions, dict):
    default_signature = exporter.classification_signature(
        examples, scores_tensor=predictions['probabilities'])
  else:
    default_signature = exporter.classification_signature(
        examples, scores_tensor=predictions)
  return default_signature, {}


@deprecated('2017-03-25',
            'signature_fns are deprecated. For canned Estimators they are no '
            'longer needed. For custom Estimators, please return '
            'output_alternatives from your model_fn via ModelFnOps.')
def regression_signature_fn(examples, unused_features, predictions):
  """Creates regression signature from given examples and predictions.

  Args:
    examples: `Tensor`.
    unused_features: `dict` of `Tensor`s.
    predictions: `Tensor`.

  Returns:
    Tuple of default regression signature and empty named signatures.

  Raises:
    ValueError: If examples is `None`.
  """
  if examples is None:
    raise ValueError('examples cannot be None when using this signature fn.')

  default_signature = exporter.regression_signature(
      input_tensor=examples, output_tensor=predictions)
  return default_signature, {}


@deprecated('2017-03-25',
            'signature_fns are deprecated. For canned Estimators they are no '
            'longer needed. For custom Estimators, please return '
            'output_alternatives from your model_fn via ModelFnOps.')
def logistic_regression_signature_fn(examples, unused_features, predictions):
  """Creates logistic regression signature from given examples and predictions.

  Args:
    examples: `Tensor`.
    unused_features: `dict` of `Tensor`s.
    predictions: `Tensor` of shape [batch_size, 2] of predicted probabilities or
      dict that contains the probabilities tensor as in
      {'probabilities', `Tensor`}.

  Returns:
    Tuple of default regression signature and named signature.

  Raises:
    ValueError: If examples is `None`.
  """
  if examples is None:
    raise ValueError('examples cannot be None when using this signature fn.')

  if isinstance(predictions, dict):
    predictions_tensor = predictions['probabilities']
  else:
    predictions_tensor = predictions
  # predictions should have shape [batch_size, 2] where first column is P(Y=0|x)
  # while second column is P(Y=1|x). We are only interested in the second
  # column for inference.
  predictions_shape = predictions_tensor.get_shape()
  predictions_rank = len(predictions_shape)
  if predictions_rank != 2:
    logging.fatal(
        'Expected predictions to have rank 2, but received predictions with '
        'rank: {} and shape: {}'.format(predictions_rank, predictions_shape))
  if predictions_shape[1] != 2:
    logging.fatal(
        'Expected predictions to have 2nd dimension: 2, but received '
        'predictions with 2nd dimension: {} and shape: {}. Did you mean to use '
        'regression_signature_fn or classification_signature_fn_with_prob '
        'instead?'.format(predictions_shape[1], predictions_shape))

  positive_predictions = predictions_tensor[:, 1]
  default_signature = exporter.regression_signature(
      input_tensor=examples, output_tensor=positive_predictions)
  return default_signature, {}


# pylint: disable=protected-access
@deprecated('2017-03-25', 'Please use Estimator.export_savedmodel() instead.')
def _default_input_fn(estimator, examples):
  """Creates default input parsing using Estimator's feature signatures."""
  return estimator._get_feature_ops_from_example(examples)


@deprecated('2016-09-23', 'Please use Estimator.export_savedmodel() instead.')
def export_estimator(estimator,
                     export_dir,
                     signature_fn=None,
                     input_fn=_default_input_fn,
                     default_batch_size=1,
                     exports_to_keep=None):
  """Deprecated, please use Estimator.export_savedmodel()."""
  _export_estimator(estimator=estimator,
                    export_dir=export_dir,
                    signature_fn=signature_fn,
                    input_fn=input_fn,
                    default_batch_size=default_batch_size,
                    exports_to_keep=exports_to_keep)


@deprecated('2017-03-25', 'Please use Estimator.export_savedmodel() instead.')
def _export_estimator(estimator,
                      export_dir,
                      signature_fn,
                      input_fn,
                      default_batch_size,
                      exports_to_keep,
                      input_feature_key=None,
                      use_deprecated_input_fn=True,
                      prediction_key=None,
                      checkpoint_path=None):
  if use_deprecated_input_fn:
    input_fn = input_fn or _default_input_fn
  elif input_fn is None:
    raise ValueError('input_fn must be defined.')

  # If checkpoint_path is specified, use the specified checkpoint path.
  checkpoint_path = (checkpoint_path or
                     tf_saver.latest_checkpoint(estimator._model_dir))
  with ops.Graph().as_default() as g:
    contrib_variables.create_global_step(g)

    if use_deprecated_input_fn:
      examples = array_ops.placeholder(dtype=dtypes.string,
                                       shape=[default_batch_size],
                                       name='input_example_tensor')
      features = input_fn(estimator, examples)
    else:
      features, _ = input_fn()
      examples = None
      if input_feature_key is not None:
        examples = features.pop(input_feature_key)

    if (not features) and (examples is None):
      raise ValueError('Either features or examples must be defined.')

    predictions = estimator._get_predict_ops(features).predictions

    if prediction_key is not None:
      predictions = predictions[prediction_key]

    # Explicit signature_fn takes priority
    if signature_fn:
      default_signature, named_graph_signatures = signature_fn(examples,
                                                               features,
                                                               predictions)
    else:
      try:
        # Some estimators provide a signature function.
        # TODO(zakaria): check if the estimator has this function,
        #   raise helpful error if not
        signature_fn = estimator._create_signature_fn()

        default_signature, named_graph_signatures = (
            signature_fn(examples, features, predictions))
      except AttributeError:
        logging.warn(
            'Change warning: `signature_fn` will be required after'
            '2016-08-01.\n'
            'Using generic signatures for now.  To maintain this behavior, '
            'pass:\n'
            '  signature_fn=export.generic_signature_fn\n'
            'Also consider passing a regression or classification signature; '
            'see cl/126430915 for an example.')
        default_signature, named_graph_signatures = generic_signature_fn(
            examples, features, predictions)
    if exports_to_keep is not None:
      exports_to_keep = gc.largest_export_versions(exports_to_keep)
    return _export_graph(
        g,
        _get_saver(),
        checkpoint_path,
        export_dir,
        default_graph_signature=default_signature,
        named_graph_signatures=named_graph_signatures,
        exports_to_keep=exports_to_keep)
# pylint: enable=protected-access
