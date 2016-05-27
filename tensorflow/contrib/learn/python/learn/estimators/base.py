# pylint: disable=g-bad-file-header
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

"""Base estimator class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from six import string_types

from tensorflow.contrib.learn.python.learn.estimators import _sklearn
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators._sklearn import NotFittedError
from tensorflow.contrib.learn.python.learn.io.data_feeder import setup_train_data_feeder
from tensorflow.contrib.learn.python.learn.utils import checkpoints

from tensorflow.python.framework import ops
from tensorflow.python.ops import constant_op
from tensorflow.python.platform import gfile


def _write_with_backup(filename, content):
  if gfile.Exists(filename):
    gfile.Rename(filename, filename + '.old', overwrite=True)
  with gfile.Open(filename, 'w') as f:
    f.write(content)


def _copy_dir(dir_in, dir_out):
  gfile.MakeDirs(dir_out)
  for name in gfile.ListDirectory(dir_in):
    name_in = os.path.join(dir_in, name)
    name_out = os.path.join(dir_out, name)
    if gfile.IsDirectory(name_in):
      gfile.MakeDirs(name_out)
      _copy_dir(name_in, name_out)
    else:
      gfile.Copy(name_in, name_out, overwrite=True)


def _new_tf_model_fn(model_fn, class_weight):
  """Backward compatibility way of adding class weight and IS_TRAINING.

  TODO(ipolosukhin): Remove this function after new layers are available.
  Specifically:
   * dropout and batch norm should work via update ops.
   * class weights should be retrieved from weights column or hparams.

  Args:
    model_fn: Core model function.
    class_weight: Class weight.
  Returns:
    Model function.
  """
  def _model_fn(features, targets, mode):
    ops.get_default_graph().add_to_collection('IS_TRAINING', mode == 'train')
    if class_weight is not None:
      constant_op.constant(class_weight, name='class_weight')
    return model_fn(features, targets)
  return _model_fn


class TensorFlowEstimator(estimator.Estimator):
  """Base class for all TensorFlow estimators.

  Parameters:
    model_fn: Model function, that takes input X, y tensors and outputs
      prediction and loss tensors.
    n_classes: Number of classes in the target.
    batch_size: Mini batch size.
    steps: Number of steps to run over data.
    optimizer: Optimizer name (or class), for example "SGD", "Adam",
      "Adagrad".
    learning_rate: If this is constant float value, no decay function is used.
      Instead, a customized decay function can be passed that accepts
      global_step as parameter and returns a Tensor.
      e.g. exponential decay function:
      def exp_decay(global_step):
          return tf.train.exponential_decay(
              learning_rate=0.1, global_step,
              decay_steps=2, decay_rate=0.001)
    clip_gradients: Clip norm of the gradients to this value to stop
      gradient explosion.
    class_weight: None or list of n_classes floats. Weight associated with
      classes for loss computation. If not given, all classes are supposed to
      have weight one.
    continue_training: when continue_training is True, once initialized
      model will be continuely trained on every call of fit.
    config: RunConfig object that controls the configurations of the
      session, e.g. num_cores, gpu_memory_fraction, etc.
    verbose: Controls the verbosity, possible values:
      0: the algorithm and debug information is muted.
      1: trainer prints the progress.
      2: log device placement is printed.
  """

  def __init__(self,
               model_fn,
               n_classes,
               batch_size=32,
               steps=200,
               optimizer='Adagrad',
               learning_rate=0.1,
               clip_gradients=5.0,
               class_weight=None,
               continue_training=False,
               config=None,
               verbose=1):
    super(TensorFlowEstimator, self).__init__(
        model_fn=_new_tf_model_fn(model_fn, class_weight),
        classification=n_classes > 1,
        learning_rate=learning_rate,
        optimizer=optimizer,
        clip_gradients=clip_gradients,
        config=config)
    self.n_classes = n_classes
    self.batch_size = batch_size
    self.steps = steps
    self.verbose = verbose
    self.continue_training = continue_training
    self._data_feeder = None

  def fit(self, x, y, steps=None, monitors=None, logdir=None):
    """Neural network model from provided `model_fn` and training data.

    Note: called first time constructs the graph and initializers
    variables. Consecutives times it will continue training the same model.
    This logic follows partial_fit() interface in scikit-learn.

    To restart learning, create new estimator.

    Args:
      x: matrix or tensor of shape [n_samples, n_features...]. Can be
      iterator that returns arrays of features. The training input
      samples for fitting the model.
      y: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
      iterator that returns array of targets. The training target values
      (class labels in classification, real numbers in regression).
      steps: int, number of steps to train.
             If None or 0, train for `self.steps`.
      monitors: List of `BaseMonitor` objects to print training progress and
        invoke early stopping.
      logdir: the directory to save the log file that can be used for
      optional visualization.

    Returns:
      Returns self.
    """
    if logdir is not None:
      self._model_dir = logdir
    self._data_feeder = setup_train_data_feeder(
        x, y, n_classes=self.n_classes, batch_size=self.batch_size)
    self._train_model(input_fn=self._data_feeder.input_builder,
                      feed_fn=self._data_feeder.get_feed_dict_fn(),
                      steps=steps or self.steps,
                      monitors=monitors)
    return self

  def evaluate(self, x=None, y=None, input_fn=None, steps=None):
    """See base class."""
    feed_fn = None
    if x is not None:
      eval_data_feeder = setup_train_data_feeder(
          x, y, n_classes=self.n_classes, batch_size=self.batch_size, epochs=1)
      input_fn, feed_fn = (eval_data_feeder.input_builder,
                           eval_data_feeder.get_feed_dict_fn())
    return self._evaluate_model(
        input_fn=input_fn, feed_fn=feed_fn, steps=steps or self.steps)

  def partial_fit(self, x, y):
    """Incremental fit on a batch of samples.

    This method is expected to be called several times consecutively
    on different or the same chunks of the dataset. This either can
    implement iterative training or out-of-core/online training.

    This is especially useful when the whole dataset is too big to
    fit in memory at the same time. Or when model is taking long time
    to converge, and you want to split up training into subparts.

    Args:
      x: matrix or tensor of shape [n_samples, n_features...]. Can be
      iterator that returns arrays of features. The training input
      samples for fitting the model.
      y: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
      iterator that returns array of targets. The training target values
      (class label in classification, real numbers in regression).

    Returns:
      Returns self.
    """
    return self.fit(x, y)

  def _predict(self, x, axis=-1, batch_size=None):
    if self._graph is None:
      raise NotFittedError()
    # Use the batch size for fitting if the user did not specify one.
    if batch_size is None:
      batch_size = self.batch_size

    predict_data_feeder = setup_train_data_feeder(
        x, None, n_classes=None,
        batch_size=batch_size,
        shuffle=False, epochs=1)

    preds = self._infer_model(
        input_fn=predict_data_feeder.input_builder,
        feed_fn=predict_data_feeder.get_feed_dict_fn())
    if self.n_classes > 1 and axis != -1:
      preds = preds.argmax(axis=axis)
    else:
      preds = preds
    return preds

  def predict(self, x, axis=1, batch_size=None):
    """Predict class or regression for X.

    For a classification model, the predicted class for each sample in X is
    returned. For a regression model, the predicted value based on X is
    returned.

    Args:
      x: array-like matrix, [n_samples, n_features...] or iterator.
      axis: Which axis to argmax for classification.
        By default axis 1 (next after batch) is used.
        Use 2 for sequence predictions.
      batch_size: If test set is too big, use batch size to split
        it into mini batches. By default the batch_size member
        variable is used.

    Returns:
      y: array of shape [n_samples]. The predicted classes or predicted
      value.
    """
    return self._predict(x, axis=axis, batch_size=batch_size)

  def predict_proba(self, x, batch_size=None):
    """Predict class probability of the input samples X.

    Args:
      x: array-like matrix, [n_samples, n_features...] or iterator.
      batch_size: If test set is too big, use batch size to split
        it into mini batches. By default the batch_size member variable is used.

    Returns:
      y: array of shape [n_samples, n_classes]. The predicted
      probabilities for each class.
    """
    return self._predict(x, batch_size=batch_size)

  def get_tensor(self, name):
    """Returns tensor by name.

    Args:
      name: string, name of the tensor.

    Returns:
      Tensor.
    """
    return self._graph.get_tensor_by_name(name)

  def get_tensor_value(self, name):
    """Returns value of the tensor give by name.

    Args:
      name: string, name of the tensor.

    Returns:
      Numpy array - value of the tensor.
    """
    if name.endswith(':0'):
      name = name[:-2]
    return checkpoints.load_variable(self.model_dir, name)

  def get_variable_names(self):
    """Returns list of all variable names in this model.

    Returns:
      List of names.
    """
    return [name for name, _ in checkpoints.list_variables(self.model_dir)]

  def save(self, path):
    """Saves checkpoints and graph to given path.

    Args:
      path: Folder to save model to.
    """
    if self._graph is None:
      raise NotFittedError

    # Copy model dir into new path.
    _copy_dir(self.model_dir, path)

    # Save model definition.
    all_params = self.get_params()
    params = {}
    for key, value in all_params.items():
      if not callable(value) and value is not None:
        params[key] = value
    params['class_name'] = type(self).__name__
    model_def = json.dumps(
        params,
        default=lambda o: o.__dict__ if hasattr(o, '__dict__') else None)
    _write_with_backup(os.path.join(path, 'model.def'), model_def)

  def _restore(self, path):
    """Restores this estimator from given path.

    Note: will rebuild the graph and initialize all parameters,
    and will ignore provided model.

    Args:
      path: Path to checkpoints and other information.
    """
    raise NotImplementedError

  @classmethod
  def restore(cls, path, config=None):
    # pylint: disable=unused-argument
    """Restores model from give path.

    Args:
      path: Path to the checkpoints and other model information.
      config: RunConfig object that controls the configurations of the session,
        e.g. num_cores, gpu_memory_fraction, etc. This is allowed to be
          reconfigured.

    Returns:
      Estimator, object of the subclass of TensorFlowEstimator.

    Raises:
      ValueError: if `path` does not contain a model definition.
    """
    model_def_filename = os.path.join(path, 'model.def')
    if not os.path.exists(model_def_filename):
      raise ValueError("Restore folder doesn't contain model definition.")
    # list of parameters that are allowed to be reconfigured
    reconfigurable_params = ['_config']
    _config = config
    with gfile.Open(model_def_filename) as fmodel:
      model_def = json.loads(fmodel.read())
      # TensorFlow binding requires parameters to be strings not unicode.
      # Only issue in Python2.
      for key, value in model_def.items():
        if isinstance(value, string_types) and not isinstance(value, str):
          model_def[key] = str(value)
        if key in reconfigurable_params:
          new_value = locals()[key]
          if new_value is not None:
            model_def[key] = new_value

    class_name = model_def.pop('class_name')
    if class_name == 'TensorFlowEstimator':
      custom_estimator = TensorFlowEstimator(model_fn=None, **model_def)
      # pylint: disable=protected-access
      custom_estimator._restore(path)
      return custom_estimator

    # To avoid cyclical dependencies, import inside the function instead of
    # the beginning of the file.
    # pylint: disable=g-import-not-at-top
    from tensorflow.contrib.learn.python.learn import estimators
    # Estimator must be one of the defined estimators in the __init__ file.
    result = getattr(estimators, class_name)(**model_def)
    # pylint: disable=protected-access
    result._restore(path)
    return result


class TensorFlowBaseTransformer(TensorFlowEstimator, _sklearn.TransformerMixin):
  """TensorFlow Base Transformer class."""

  def transform(self, X):
    """Transform X using trained transformer."""
    return(super(TensorFlowBaseTransformer, self).predict(
        X, axis=1, batch_size=None))

  def fit(self, X, y=None, monitor=None, logdir=None):
    """Fit a transformer."""
    return(super(TensorFlowBaseTransformer, self).fit(
        X, y, monitors=None, logdir=None))

  def fit_transform(self, X, y=None, monitor=None, logdir=None):
    """Fit transformer and transform X using trained transformer."""
    return self.fit(X, y, monitor=None, logdir=None).transform(X)
