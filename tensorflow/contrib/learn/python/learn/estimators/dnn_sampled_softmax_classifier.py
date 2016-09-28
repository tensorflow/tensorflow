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
#
# ==============================================================================

"""Deep Neural Network estimator for large multi-class multi-label problems.

The Training is sped up using Candidate Sampling. Evaluation and Inference
uses full softmax.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tempfile

from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import optimizers
from tensorflow.contrib.learn.python.learn import evaluable
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.learn.python.learn import trainable
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.metrics.python.ops import metric_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import training as train


_CLASSES = "classes"
_TOP_K = "top_k"
_PROBABILITIES = "probabilities"
_DEFAULT_LEARNING_RATE = 0.01


def _as_iterable(preds, output):
  for pred in preds:
    yield pred[output]


def _get_optimizer(optimizer):
  if callable(optimizer):
    return optimizer()
  else:
    return optimizer


def _get_default_optimizer():
  """Default optimizer for DNN models."""
  return train.AdagradOptimizer(_DEFAULT_LEARNING_RATE)


def dnn_sampled_softmax_classifier_model_fn(features, target_indices,
                                            mode, params):
  """model_fn that uses candidate sampling.

  Args:
    features: Single Tensor or dict of Tensor (depends on data passed to `fit`)
    target_indices: A single Tensor of shape [batch_size, n_labels] containing
      the target indices.
    mode: Represents if this training, evaluation or prediction. See `ModeKeys`.
    params: A dict of hyperparameters that are listed below.
      hidden_units- List of hidden units per layer. All layers are fully
        connected. Ex. `[64, 32]` means first layer has 64 nodes and second one
        has 32.
      feature_columns- An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      n_classes- number of target classes. It must be greater than 2.
      n_samples- number of sample target classes. Needs to be tuned - A good
        starting point could be 2% of n_classes.
      n_labels- number of labels in each example.
      top_k- The number of classes to predict.
      optimizer- An instance of `tf.Optimizer` used to train the model. If
        `None`, will use an Adagrad optimizer.
      dropout- When not `None`, the probability we will drop out a given
        coordinate.
      gradient_clip_norm- A float > 0. If provided, gradients are
        clipped to their global norm with this clipping ratio. See
        tf.clip_by_global_norm for more details.
      num_ps_replicas- The number of parameter server replicas.

  Returns:
    predictions: A single Tensor or a dict of Tensors.
    loss: A scalar containing the loss of the step.
    train_op: The op for training.
  """

  hidden_units = params["hidden_units"]
  feature_columns = params["feature_columns"]
  n_classes = params["n_classes"]
  n_samples = params["n_samples"]
  n_labels = params["n_labels"]
  top_k = params["top_k"]
  optimizer = params["optimizer"]
  dropout = params["dropout"]
  gradient_clip_norm = params["gradient_clip_norm"]
  num_ps_replicas = params["num_ps_replicas"]

  parent_scope = "dnn_ss"

  # Setup the input layer partitioner.
  input_layer_partitioner = (
      partitioned_variables.min_max_variable_partitioner(
          max_partitions=num_ps_replicas,
          min_slice_size=64 << 20))

  # Create the input layer.
  with variable_scope.variable_scope(
      parent_scope + "/input_from_feature_columns",
      features.values(),
      partitioner=input_layer_partitioner) as scope:
    net = layers.input_from_feature_columns(
        features,
        feature_columns,
        weight_collections=[parent_scope],
        scope=scope)

  # Setup the hidden layer partitioner.
  hidden_layer_partitioner = (
      partitioned_variables.min_max_variable_partitioner(
          max_partitions=num_ps_replicas))

  final_hidden_layer_dim = None
  # Create hidden layers using fully_connected.
  for layer_id, num_hidden_units in enumerate(hidden_units):
    with variable_scope.variable_scope(
        parent_scope + "/hiddenlayer_%d" % layer_id, [net],
        partitioner=hidden_layer_partitioner) as scope:
      net = layers.fully_connected(net,
                                   num_hidden_units,
                                   variables_collections=[parent_scope],
                                   scope=scope)
      final_hidden_layer_dim = num_hidden_units
      # Add dropout if it is enabled.
      if dropout is not None and mode == estimator.ModeKeys.TRAIN:
        net = layers.dropout(net, keep_prob=(1.0 - dropout))

  # Create the weights and biases for the logit layer.
  with variable_scope.variable_scope(
      parent_scope + "/logits", [net],
      partitioner=hidden_layer_partitioner) as scope:
    dtype = net.dtype.base_dtype
    weights_shape = [n_classes, final_hidden_layer_dim]
    weights = variables.model_variable(
        "weights",
        shape=weights_shape,
        dtype=dtype,
        initializer=initializers.xavier_initializer(),
        trainable=True,
        collections=[parent_scope])
    biases = variables.model_variable(
        "biases",
        shape=[n_classes,],
        dtype=dtype,
        initializer=init_ops.zeros_initializer,
        trainable=True,
        collections=[parent_scope])

  if mode == estimator.ModeKeys.TRAIN:
    # Call the candidate sampling APIs and calculate the loss.
    sampled_values = nn.learned_unigram_candidate_sampler(
        true_classes=math_ops.to_int64(target_indices),
        num_true=n_labels,
        num_sampled=n_samples,
        unique=True,
        range_max=n_classes)

    sampled_softmax_loss = nn.sampled_softmax_loss(
        weights=weights,
        biases=biases,
        inputs=net,
        labels=math_ops.to_int64(target_indices),
        num_sampled=n_samples,
        num_classes=n_classes,
        num_true=n_labels,
        sampled_values=sampled_values)

    loss = math_ops.reduce_mean(sampled_softmax_loss, name="loss")

    train_op = optimizers.optimize_loss(
        loss=loss, global_step=contrib_framework.get_global_step(),
        learning_rate=_DEFAULT_LEARNING_RATE,
        optimizer=_get_optimizer(optimizer), clip_gradients=gradient_clip_norm,
        name=parent_scope)
    return None, loss, train_op

  elif mode == estimator.ModeKeys.EVAL:
    logits = nn.bias_add(standard_ops.matmul(net, array_ops.transpose(weights)),
                         biases)
    predictions = {}
    predictions[_PROBABILITIES] = nn.softmax(logits)
    predictions[_CLASSES] = math_ops.argmax(logits, 1)
    _, predictions[_TOP_K] = nn.top_k(logits, top_k)

    # Since the targets have multiple labels, setup the target probabilities
    # as 1.0/n_labels for each of the labels.
    target_one_hot = array_ops.one_hot(
        indices=target_indices,
        depth=n_classes,
        on_value=1.0 / n_labels)
    target_one_hot = math_ops.reduce_sum(
        input_tensor=target_one_hot,
        reduction_indices=[1])

    loss = math_ops.reduce_mean(
        nn.softmax_cross_entropy_with_logits(logits, target_one_hot))

    return predictions, loss, None

  elif mode == estimator.ModeKeys.INFER:
    logits = nn.bias_add(standard_ops.matmul(net, array_ops.transpose(weights)),
                         biases)
    predictions = {}
    predictions[_PROBABILITIES] = nn.softmax(logits)
    predictions[_CLASSES] = math_ops.argmax(logits, 1)
    _, predictions[_TOP_K] = nn.top_k(logits, top_k)

    return predictions, None, None


class _DNNSampledSoftmaxClassifier(trainable.Trainable, evaluable.Evaluable):
  """A classifier for TensorFlow DNN models.

  Example:

  ```python
  legos = sparse_column_with_hash_bucket(column_name="legos",
                                         hash_bucket_size=1000)
  watched_videos = sparse_column_with_hash_bucket(
                     column_name="watched_videos",
                     hash_bucket_size=20000)

  legos_emb = embedding_column(sparse_id_column=legos, dimension=16,
                               combiner="sum")
  watched_videos_emb = embedding_column(sparse_id_column=watched_videos,
                                        dimension=256,
                                        combiner="sum")

  estimator = DNNSampledSoftmaxClassifier(
      n_classes=500000, n_samples=10000, n_labels=5,
      feature_columns=[legos_emb, watched_videos_emb],
      hidden_units=[1024, 512, 256])

  # Or estimator using the Adam optimizer with dropout.
  estimator = DNNSampledSoftmaxClassifier(
      feature_columns=[education_emb, occupation_emb],
      hidden_units=[1024, 512, 256],
      optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1),
      dropout=0.1)

  # Input builders
  def input_fn_train: # returns x, Y
    pass
  estimator.fit(input_fn=input_fn_train)

  def input_fn_eval: # returns x, Y
    pass
  estimator.evaluate(input_fn=input_fn_eval)
  estimator.predict(x=x)
  ```

  Input of `fit` and `evaluate` should have following features,
    otherwise there will be a `KeyError`:

  * for each `column` in `feature_columns`:
    - if `column` is a `SparseColumn`, a feature with `key=column.name`
      whose `value` is a `SparseTensor`.
    - if `column` is a `EmbeddingColumn`, a feature with `key=column.name`
      whose `value` is a `SparseTensor`.
    - if `column` is a `WeightedSparseColumn`, two features: the first with
      `key` the id column name, the second with `key` the weight column name.
      Both features' `value` must be a `SparseTensor`.
    - if `column` is a `RealValuedColumn`, a feature with `key=column.name`
      whose `value` is a `Tensor`.
  """

  def __init__(self,
               hidden_units,
               feature_columns,
               n_classes,
               n_samples,
               n_labels=1,
               top_k=1,
               model_dir=None,
               optimizer=None,
               dropout=None,
               gradient_clip_norm=None,
               config=None):
    """Initializes a DNNSampledSoftmaxClassifier instance.

    Args:
      hidden_units: List of hidden units per layer. All layers are fully
        connected. Ex. `[64, 32]` means first layer has 64 nodes and second one
        has 32.
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      n_classes: number of target classes. It must be greater than 2.
      n_samples: number of sample target classes. Needs to be tuned - A good
        starting point could be 2% of n_classes.
      n_labels: number of labels in each example.
      top_k: The number of classes to predict.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      optimizer: An instance of `tf.Optimizer` used to train the model. If
        `None`, will use an Adagrad optimizer.
      dropout: When not `None`, the probability we will drop out a given
        coordinate.
      gradient_clip_norm: A float > 0. If provided, gradients are
        clipped to their global norm with this clipping ratio. See
        tf.clip_by_global_norm for more details.
      config: `RunConfig` object to configure the runtime settings.

    Returns:
      A `DNNSampledSoftmaxClassifier` estimator.

    Raises:
      ValueError: If n_classes <= 2.
      ValueError: If n_classes < n_samples.
      ValueError: If n_classes < n_labels.
    """
    # Validate all the inputs.
    if n_classes <= 2:
      raise ValueError("n_classes should be greater than 2. For n_classes <= 2,"
                       " use DNNClassifier.")
    if n_classes < n_samples:
      raise ValueError("n_classes (%d) should be greater than n_samples (%d)." %
                       (n_classes, n_samples))
    if n_classes < n_labels:
      raise ValueError("n_classes (%d) should be greater than n_labels"
                       " (%d)." % (n_classes, n_labels))

    self._top_k = top_k
    self._feature_columns = feature_columns
    assert self._feature_columns
    self._model_dir = model_dir or tempfile.mkdtemp()

    # Build the estimator with dnn_sampled_softmax_classifier_model_fn.
    self._estimator = estimator.Estimator(
        model_fn=dnn_sampled_softmax_classifier_model_fn,
        model_dir=self._model_dir,
        config=config,
        params={
            "hidden_units": hidden_units,
            "feature_columns": feature_columns,
            "n_classes": n_classes,
            "n_samples": n_samples,
            "n_labels": n_labels,
            "top_k": top_k,
            "optimizer": optimizer or _get_default_optimizer(),
            "dropout": dropout,
            "gradient_clip_norm": gradient_clip_norm,
            "num_ps_replicas": config.num_ps_replicas if config else 0})

  def get_estimator(self):
    return self._estimator

  def fit(self, x=None, y=None, input_fn=None, steps=None, batch_size=None,
          monitors=None, max_steps=None):
    """See trainable.Trainable."""
    return self._estimator.fit(x=x, y=y, input_fn=input_fn, steps=steps,
                               batch_size=batch_size, monitors=monitors,
                               max_steps=max_steps)

  def evaluate(self, x=None, y=None, input_fn=None, feed_fn=None,
               batch_size=None, steps=None, metrics=None, name=None,
               range_k=None):
    # pylint: disable=g-doc-args,g-doc-return-or-yield
    """See evaluable.Evaluable for a description of the Args.

    range_k: A list of numbers where precision and recall have to be obtained.
      For eg. range_k=[1,5] will calculate precision@1, precision@5,
      recall@1 and recall@5.
    """
    # Setup the default metrics if metrics are not specified - precision@1,
    # recall@1 and precision@top_k and recall@top_k if top_k is
    # greater than 1.
    if not metrics:
      metrics = {}
      if range_k is None:
        if self._top_k > 1:
          range_k = [1, self._top_k]
        else:
          range_k = [1]
      for k in range_k:
        metrics.update({
            "precision_at_%d" % k: metric_spec.MetricSpec(
                metric_fn=functools.partial(
                    metric_ops.streaming_sparse_precision_at_k, k=k),
                prediction_key=_PROBABILITIES,)})
        metrics.update({
            "recall_at_%d" % k: metric_spec.MetricSpec(
                metric_fn=functools.partial(
                    metric_ops.streaming_sparse_recall_at_k, k=k),
                prediction_key=_PROBABILITIES,)})

    return self._estimator.evaluate(x=x, y=y, input_fn=input_fn,
                                    feed_fn=feed_fn, batch_size=batch_size,
                                    steps=steps, metrics=metrics, name=name)

  def predict(self, x=None, input_fn=None, batch_size=None, as_iterable=False,
              get_top_k=False):
    """Returns predicted classes for given features.

    Args:
      x: features.
      input_fn: Input function. If set, x must be None.
      batch_size: Override default batch size.
      as_iterable: If True, return an iterable which keeps yielding predictions
        for each example until inputs are exhausted. Note: The inputs must
        terminate if you want the iterable to terminate (e.g. be sure to pass
        num_epochs=1 if you are using something like read_batch_features).
      get_top_k : if set to true returns the top k classes otherwise returns
        the top class.

    Returns:
      Numpy array of predicted classes (or an iterable of predicted classes if
      as_iterable is True).
    """
    if get_top_k:
      key = _TOP_K
    else:
      key = _CLASSES
    preds = self._estimator.predict(x=x, input_fn=input_fn,
                                    batch_size=batch_size, outputs=[key],
                                    as_iterable=as_iterable)
    if as_iterable:
      return _as_iterable(preds, output=key)
    return preds[key]

  def predict_proba(self, x=None, input_fn=None, batch_size=None,
                    as_iterable=False):
    """Returns prediction probabilities for given features.

    Args:
      x: features.
      input_fn: Input function. If set, x and y must be None.
      batch_size: Override default batch size.
      as_iterable: If True, return an iterable which keeps yielding predictions
        for each example until inputs are exhausted. Note: The inputs must
        terminate if you want the iterable to terminate (e.g. be sure to pass
        num_epochs=1 if you are using something like read_batch_features).

    Returns:
      Numpy array of predicted probabilities (or an iterable of predicted
      probabilities if as_iterable is True).
    """
    preds = self._estimator.predict(x=x, input_fn=input_fn,
                                    batch_size=batch_size,
                                    outputs=[_PROBABILITIES],
                                    as_iterable=as_iterable)
    if as_iterable:
      return _as_iterable(preds, output=_PROBABILITIES)
    return preds[_PROBABILITIES]

  def export(self, export_dir, signature_fn=None,
             input_fn=None, default_batch_size=1,
             exports_to_keep=None):
    """Exports inference graph into given dir.

    Args:
      export_dir: A string containing a directory to write the exported graph
        and checkpoints.
      signature_fn: Function that returns a default signature and a named
        signature map, given `Tensor` of `Example` strings, `dict` of `Tensor`s
        for features and `Tensor` or `dict` of `Tensor`s for predictions.
      input_fn: If `use_deprecated_input_fn` is true, then a function that given
        `Tensor` of `Example` strings, parses it into features that are then
        passed to the model. Otherwise, a function that takes no argument and
        returns a tuple of (features, targets), where features is a dict of
        string key to `Tensor` and targets is a `Tensor` that's currently not
        used (and so can be `None`).
      default_batch_size: Default batch size of the `Example` placeholder.
      exports_to_keep: Number of exports to keep.

    Returns:
      The string path to the exported directory. NB: this functionality was
      added ca. 2016/09/25; clients that depend on the return value may need
      to handle the case where this function returns None because subclasses
      are not returning a value.
    """
    def default_input_fn(unused_estimator, examples):
      return layers.parse_feature_columns_from_examples(
          examples, self._feature_columns)
    return self._estimator.export(export_dir=export_dir,
                                  signature_fn=signature_fn,
                                  input_fn=input_fn or default_input_fn,
                                  default_batch_size=default_batch_size,
                                  exports_to_keep=exports_to_keep)

  def get_variable_names(self):
    return self._estimator.get_variable_names()

  @property
  def model_dir(self):
    return self._model_dir
