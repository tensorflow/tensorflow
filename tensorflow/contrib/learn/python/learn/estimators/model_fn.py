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

"""Classes and methods related to model_fn (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import six

from tensorflow.contrib.framework import get_graph_from_inputs
from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators import metric_key
from tensorflow.contrib.learn.python.learn.estimators import prediction_key
from tensorflow.python.estimator import model_fn as core_model_fn_lib
from tensorflow.python.estimator.export import export_output as core_export_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import session_run_hook
from tensorflow.python.util.deprecation import deprecated


class ModeKeys(object):
  """Standard names for model modes (deprecated).

  THIS CLASS IS DEPRECATED.

  The following standard keys are defined:

  * `TRAIN`: training mode.
  * `EVAL`: evaluation mode.
  * `INFER`: inference mode.
  """

  TRAIN = 'train'
  EVAL = 'eval'
  INFER = 'infer'

  @classmethod
  def validate(cls, key):
    if key not in (cls.TRAIN, cls.EVAL, cls.INFER):
      raise ValueError('Invalid mode %s.' % key)


class ModelFnOps(
    collections.namedtuple('ModelFnOps', [
        'predictions', 'loss', 'train_op', 'eval_metric_ops',
        'output_alternatives', 'training_chief_hooks', 'training_hooks',
        'scaffold', 'mode'
    ])):
  """Ops returned from a model_fn.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.
  """

  @deprecated(None, 'When switching to tf.estimator.Estimator, use '
              'tf.estimator.EstimatorSpec. You can use the `estimator_spec`'
              ' method to create an equivalent one.')
  def __new__(cls,
              mode,
              predictions=None,
              loss=None,
              train_op=None,
              eval_metric_ops=None,
              output_alternatives=None,
              training_chief_hooks=None,
              training_hooks=None,
              scaffold=None):
    """Creates a validated `ModelFnOps` instance.

    For a multi-headed model, the predictions dict here will contain the outputs
    of all of the heads.  However: at serving time, requests will be made
    specifically for one or more heads, and the RPCs used for these requests may
    differ by problem type (i.e., regression, classification, other).  The
    purpose of the output_alternatives dict is to aid in exporting a SavedModel
    from which such head-specific queries can be served.  These
    output_alternatives will be combined with input_alternatives (see
    `saved_model_export_utils`) to produce a set of `SignatureDef`s specifying
    the valid requests that can be served from this model.

    For a single-headed model, it is still adviseable to provide
    output_alternatives with a single entry, because this is how the problem
    type is communicated for export and serving.  If output_alternatives is not
    given, the resulting SavedModel will support only one head of unspecified
    type.

    Args:
      mode: One of `ModeKeys`. Specifies if this training, evaluation or
        prediction.
      predictions: Predictions `Tensor` or dict of `Tensor`.
      loss: Training loss `Tensor`.
      train_op: Op for the training step.
      eval_metric_ops: Dict of metric results keyed by name. The values of the
        dict are the results of calling a metric function, such as `Tensor`.
      output_alternatives: a dict of
        `{submodel_name: (problem_type, {tensor_name: Tensor})}`, where
        `submodel_name` is a submodel identifier that should be consistent
        across the pipeline (here likely taken from the name of each `Head`,
        for models that use them), `problem_type` is a `ProblemType`,
        `tensor_name` is a symbolic name for an output Tensor possibly but not
        necessarily taken from `PredictionKey`, and `Tensor` is the
        corresponding output Tensor itself.
      training_chief_hooks: A list of `SessionRunHook` objects that will be
        run on the chief worker during training.
      training_hooks: A list of `SessionRunHook` objects that will be run on
        all workers during training.
      scaffold: A `tf.compat.v1.train.Scaffold` object that can be used to set
        initialization, saver, and more to be used in training.

    Returns:
      A validated `ModelFnOps` object.

    Raises:
      ValueError: If validation fails.
    """
    ModeKeys.validate(mode)

    # Assert all ops are from the same graph.
    get_graph_from_inputs((predictions, loss, train_op))

    # Validate train_op.
    if train_op is None:
      if mode == ModeKeys.TRAIN:
        raise ValueError('Missing train_op.')
    elif not isinstance(train_op, ops.Operation):
      # TODO(ptucker): Should this be allowed? Consider raising error.
      train_op = ops.convert_to_tensor(train_op).op

    # Validate loss.
    if loss is None:
      if mode in (ModeKeys.TRAIN, ModeKeys.EVAL):
        raise ValueError('Missing loss.')
    else:
      loss = ops.convert_to_tensor(loss)
      loss_shape = loss.get_shape()
      if loss_shape.num_elements() not in (None, 1):
        raise ValueError('Loss must be scalar: %s.' % loss)
      if not loss_shape.is_compatible_with(tensor_shape.scalar()):
        loss = array_ops.reshape(loss, [])

    # Validate predictions.
    if predictions is None:
      if mode == ModeKeys.INFER or mode == ModeKeys.EVAL:
        raise ValueError('Missing predictions.')
    else:
      if isinstance(predictions, dict):
        predictions = {
            k: sparse_tensor.convert_to_tensor_or_sparse_tensor(v)
            for k, v in six.iteritems(predictions)
        }
      else:
        predictions = sparse_tensor.convert_to_tensor_or_sparse_tensor(
            predictions)

    # Validate eval_metric_ops
    if eval_metric_ops is None:
      eval_metric_ops = {}
    else:
      if not isinstance(eval_metric_ops, dict):
        raise ValueError('eval_metric_ops must be a dict.')

    # Validate hooks
    if training_chief_hooks is None:
      training_chief_hooks = []
    if training_hooks is None:
      training_hooks = []
    for hook in training_hooks + training_chief_hooks:
      if not isinstance(hook, session_run_hook.SessionRunHook):
        raise TypeError('All hooks returned from model_fn must be '
                        'SessionRunHook instances, got instance of %s: %s' %
                        (type(hook), hook))

    return super(ModelFnOps, cls).__new__(
        cls,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        output_alternatives=output_alternatives,
        training_chief_hooks=training_chief_hooks,
        training_hooks=training_hooks,
        scaffold=scaffold,
        mode=mode)

  def estimator_spec(self, default_serving_output_alternative_key=None):
    """Creates an equivalent `EstimatorSpec`.

    Args:
      default_serving_output_alternative_key: Required for multiple heads. If
        you have multiple entries in `output_alternatives` dict (comparable to
        multiple heads), `EstimatorSpec` requires a default head that will be
        used if a Servo request does not explicitly mention which head to infer
        on. Pass the key of the output alternative here that you want to
        designate as default. A separate ExportOutpout for this default head
        will be added to the export_outputs dict with the special key
        saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY, unless there is
        already an enry in output_alternatives with this special key.

    Returns:
      Instance of `EstimatorSpec` that is equivalent to this `ModelFnOps`

    Raises:
      ValueError: If problem type is unknown.
    """
    def _scores(output_tensors):
      scores = output_tensors.get(prediction_key.PredictionKey.SCORES)
      if scores is None:
        scores = output_tensors.get(prediction_key.PredictionKey.PROBABILITIES)
      return scores

    def _classes(output_tensors):  # pylint: disable=missing-docstring
      classes = output_tensors.get(prediction_key.PredictionKey.CLASSES)
      if classes is None:
        logging.warning(
            'classes is None, Servo inference will not have class ids.')
        return None
      elif classes.dtype != dtypes.string:
        # Servo classification can only serve string classes
        logging.warning(
            'classes is not string, Servo inference will not have class ids.')
        return None

      return classes

    def _export_output(problem_type, predictions):  # pylint: disable=missing-docstring
      if problem_type == constants.ProblemType.LINEAR_REGRESSION:
        return core_export_lib.RegressionOutput(_scores(predictions))

      if (problem_type == constants.ProblemType.CLASSIFICATION or
          problem_type == constants.ProblemType.LOGISTIC_REGRESSION):
        return core_export_lib.ClassificationOutput(
            scores=_scores(predictions), classes=_classes(predictions))

      if problem_type == constants.ProblemType.UNSPECIFIED:
        return core_export_lib.PredictOutput(predictions)

      raise ValueError('Unknown problem_type=%s' % problem_type)

    # Converts output_alternatives
    export_outputs_dict = None
    if self.output_alternatives:
      output_alternatives = self.output_alternatives
      # Adds default output_alternative if needed.
      if (len(output_alternatives) > 1 and
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY not in
          output_alternatives):
        output_alternatives = output_alternatives.copy()
        output_alternatives[
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = (
                output_alternatives[default_serving_output_alternative_key])
      export_outputs_dict = {key: _export_output(*val) for key, val in
                             output_alternatives.items()}

    def _get_eval_metric_ops():
      """Returns self.eval_metric_ops without loss metric."""
      result = {}
      for key, value in six.iteritems(self.eval_metric_ops):
        if key != metric_key.MetricKey.LOSS:
          result[key] = value
      return result

    # Convert the contrib mode enum to the core mode enum.
    # Note: mode already validated in __new__().
    if self.mode == ModeKeys.TRAIN:
      core_mode = core_model_fn_lib.ModeKeys.TRAIN
    elif self.mode == ModeKeys.EVAL:
      core_mode = core_model_fn_lib.ModeKeys.EVAL
    elif self.mode == ModeKeys.INFER:
      core_mode = core_model_fn_lib.ModeKeys.PREDICT

    return core_model_fn_lib.EstimatorSpec(
        mode=core_mode,
        predictions=self.predictions,
        loss=self.loss,
        train_op=self.train_op,
        eval_metric_ops=_get_eval_metric_ops(),
        export_outputs=export_outputs_dict,
        training_chief_hooks=self.training_chief_hooks,
        training_hooks=self.training_hooks,
        scaffold=self.scaffold)
