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

from tensorflow.contrib import layers
from tensorflow.contrib.framework import deprecated
from tensorflow.contrib.framework import deprecated_arg_values
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
from tensorflow.python.platform import tf_logging as logging
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
                  named_graph_signatures=named_graph_signatures,
                  assets_collection=ops.get_collection(
                      ops.GraphKeys.ASSET_FILEPATHS))
      return export.export(export_dir, contrib_variables.get_global_step(),
                           session, exports_to_keep=exports_to_keep)


def generic_signature_fn(examples, unused_features, predictions):
  """Creates generic signature from given examples and predictions.

  This is needed for backward compatibility with default behaviour of
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
@deprecated(
    '2016-09-23',
    'The signature of the input_fn accepted by export is changing to be '
    'consistent with what\'s used by tf.Learn Estimator\'s train/evaluate, '
    'which makes this function useless. This will be removed after the '
    'deprecation date.')
def _default_input_fn(estimator, examples):
  """Creates default input parsing using Estimator's feature signatures."""
  return estimator._get_feature_ops_from_example(examples)


@deprecated('2016-09-23', 'Please use BaseEstimator.export')
def export_estimator(estimator,
                     export_dir,
                     signature_fn=None,
                     input_fn=_default_input_fn,
                     default_batch_size=1,
                     exports_to_keep=None):
  """Deprecated, please use BaseEstimator.export."""
  _export_estimator(estimator=estimator,
                    export_dir=export_dir,
                    signature_fn=signature_fn,
                    input_fn=input_fn,
                    default_batch_size=default_batch_size,
                    exports_to_keep=exports_to_keep)


@deprecated_arg_values(
    '2016-09-23',
    'The signature of the input_fn accepted by export is changing to be '
    'consistent with what\'s used by tf.Learn Estimator\'s train/evaluate. '
    'input_fn and (and in most cases, input_feature_key) will become required '
    'args. use_deprecated_input_fn will default to False and be removed. '
    'default_batch_size will also be removed since it will now be a part of '
    'the input_fn.',
    use_deprecated_input_fn=True,
    default_batch_size=1)
def _export_estimator(estimator,
                      export_dir,
                      signature_fn,
                      input_fn,
                      default_batch_size,
                      exports_to_keep,
                      input_feature_key=None,
                      use_deprecated_input_fn=True,
                      prediction_key=None):
  if use_deprecated_input_fn:
    input_fn = input_fn or _default_input_fn
  elif input_fn is None:
    raise ValueError('input_fn must be defined.')

  checkpoint_path = tf_saver.latest_checkpoint(estimator._model_dir)
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
        examples = features[input_feature_key]

    if not features and not examples:
      raise ValueError('Either features or examples must be defined.')

    predictions = estimator._get_predict_ops(features)
    if prediction_key is not None:
      predictions = predictions[prediction_key]

    # Explicit signature_fn takes priority
    if signature_fn:
      default_signature, named_graph_signatures = signature_fn(examples,
                                                               features,
                                                               predictions)
    else:
      try:
        # Some estimators provide a target_column of known type
        target_column = estimator._get_target_column()
        problem_type = target_column.problem_type

        if problem_type == layers.ProblemType.CLASSIFICATION:
          signature_fn = classification_signature_fn
        elif problem_type == layers.ProblemType.LINEAR_REGRESSION:
          signature_fn = regression_signature_fn
        elif problem_type == layers.ProblemType.LOGISTIC_REGRESSION:
          signature_fn = logistic_regression_signature_fn
        else:
          raise ValueError(
              'signature_fn must be provided because the TargetColumn is a %s, '
              'which does not have a standard problem type and so cannot use a '
              'standard export signature.' % type(target_column).__name__)

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
