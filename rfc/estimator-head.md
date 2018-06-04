# Estimator Head

| Status        | Proposed                                          |
:-------------- | :-------------------------------------------------|
| **Author(s)** | George Roumpos (Google), TensorFlow team          |
| **Sponsor**   | Mustafa Ispir (Google)                            |
| **Updated**   | 2018-05-01                                        |

## Objective

In this doc we discuss the Head API, which helps users define customizable
models that follow the `tf.estimator` API. The API
includes:

*   A `Head` interface.
*   Factory methods to create common heads, such as regression head.
*   A `multi_head` method that combines more than one heads for multi-objective
    learning.
*   Canned estimators that can use those heads.

The API is already exposed under `tf.contrib.estimator`. The code is in
`tensorflow/contrib/estimator/python/estimator/head.py`,
`tensorflow/contrib/estimator/python/estimator/multi_head.py` and
`tensorflow/python/estimator/canned/head.py`, and an earlier (deprecated)
version is exposed under `tf.contrib.learn`. The goal of this design doc is to
graduate the API from contrib to core.

## Motivation

[Canned Estimators](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/estimator/canned),
such as DNNClassifier, are part of the core Tensorflow API. They allow users to
quickly create an Estimator instance to use in their model. Users need to
specify some hyperparameters when calling the canned estimator constructor, and
the canned estimator takes care of creating the Tensorflow graph for training,
eval etc. For users who build custom models, Tensorflow also offers the
[Estimator](https://www.tensorflow.org/programmers_guide/estimators) API. Users
need to specify a `model_fn`, which contains Tensorflow ops to define the graph.

Head offers an intermediate level of abstraction: the `model_fn` is split into a
`logit_fn` that contains the ops to create the logits Tensor, and a `Head` that
produces ops for train, eval and infer given those logits. Internally, canned
estimators use this scheme. E.g. DNNClassifier and DNNRegressor use the same
`logit_fn` but the former uses a classification head, whereas the latter uses a
regression head. This reduces code duplication, improves testing, and makes it
easier for users to adapt models for their own use.

Heads are responsible for creating an `EstimatorSpec` object, so it has control
over all properties of `EstimatorSpec`. As a summary, head is responsible for:

*   training loss: specify the type of loss function and calculating it.
*   evaluation metrics: such as precision/recall.
*   predictions: such as converting logits to probabilities.
*   export signatures: by specifying the `export_outputs` property. Some heads
    export multiple signatures and set one of them as default.

### Use Cases

The Head API targets users who wish to use a common network, for example the
canned DNN, Linear or DNNLinearCombined, but their use case is not fully covered
by canned estimators. The API gives users the option to:

*   Use one of the heads that the API provides, such as `multi_label_head`.
*   Write their own head.
*   Combine multiple heads using `multi_head` for multi-objective learning.

The Head API also targets users who use a `model_fn` to define their own
network, such as an RNN, but want to use a common head, such as
`classification_head`.

## Design Proposal

### Examples

Here we give example code that constructs an estimator using a pre-defined head.

Use a pre-defined head with a canned estimator:


```python
# Define feature columns
column_a = tf.feature_column.sparse_column_with_hash_bucket(
    'column_a', hash_bucket_size=100)
column_b = tf.feature_column.sparse_column_with_keys(
    'column_b', keys=['a', 'b', 'c'])
embedding_a = tf.feature_column.embedding_column(
    column_a, dimension=10)
embedding_b = tf.feature_column.embedding_column(
    column_b, dimension=20)

# Define head.
my_head = tf.estimator.multi_label_head(n_classes=3)

# Define estimator.
my_estimator = tf.estimator.DNNEstimator(
    head=my_head,
    feature_columns=[embedding_a, embedding_b],
    hidden_units=...)
```


Use multi-head with a canned estimator:


```python
# Define head.
head_a = tf.estimator.multi_label_head(n_classes=3)
head_b = tf.estimator.binary_classification_head()
my_head = tf.estimator.multi_head([head_a, head_b])

# Define estimator.
my_estimator = tf.estimator.DNNEstimator(
    head=my_head,
    feature_columns=[embedding_a, embedding_b],
    hidden_units=...)
```


Use a pre-defined head with a custom estimator:


```python
# Define model_fn.
def _my_model_fn(features, labels, mode):
  n_classes = 3
  # Calculate logits.
  model = tf.keras.Model(...)
  logits = model(features)

  # Define head.
  my_head = tf.estimator.multi_label_head(n_classes=n_classes)

  # Return EstimatorSpec.
  return my_head.create_estimator_spec(
      features=features,
      mode=mode,
      labels=labels,
      optimizer=tf.AdagradOptimizer(learning_rate=0.1),
      logits=logits)

# Define estimator.
my_estimator = tf.estimator.Estimator(model_fn=_my_model_fn)
```

### API

#### Head Interface

The code for the Head interface as of 2018-05-01 can be found [here](https://github.com/tensorflow/tensorflow/blob/8bed1ea47d96c53db7d8b68b811b1487635d4106/tensorflow/python/estimator/canned/head.py).
We plan to expose it as **<code>tf.estimator.Head</code>**.
Here is the interface (with short pydoc):

```python
class Head(object):
  """Interface for the head/top of a model."""
  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def name(self):
    """The name of this head."""

  @abc.abstractproperty
  def logits_dimension(self):
    """Size of the last dimension of the logits `Tensor`."""

  @abc.abstractmethod
  def create_loss(self, features, mode, logits, labels):
    """Calculates loss and returns `LossSpec`."""

  @abc.abstractmethod
  def create_estimator_spec(
      self, features, mode, logits, labels=None, optimizer=None,
      train_op_fn=None, regularization_losses=None):
    """Returns `EstimatorSpec` that a model_fn can return."""


# A tuple with loss-related tensors.
#
# Contains
# * a scalar `Tensor` representing reduced weighted training loss
# * a `Tensor` representing the unreduced unweighted loss
# * a `Tensor` representing the example weights
# * possibly processed labels (e.g. vocabulary lookup, shape manipulation, etc)
LossSpec = collections.namedtuple(
    'LossSpec', ['training_loss', 'unreduced_loss', 'weights',
                 'processed_labels'])
```


Here is some more explanation and motivation for each method:

*   **<code>name(self)</code>:** The name of the head. This is typically given
    by the user at construction time, and can be <code>None</code> in the case
    of single head. The motivation for this method is to create the graph under
    <code>tf.name_scope(head.name)</code> and to name the metrics as
    <code>head.name/metric_name</code>. This is useful in the multi-head case.
    Multi-head also uses <code>head.name</code> to export serving signatures for all heads
    using distinct signature names.
*   <strong>logits_dimension(self):</strong> Last dimension of the logits
    Tensor. For <code>multi_class_head</code>, this equals the number of
    classes. Typically, the logits Tensor has shape [batch_size,
    logits_dimension]. This method is used by canned estimators to create logits
    of the appropriate shape. Head implementations also use this to validate the
    shape of input Tensors, such as logits, to catch common errors.
*   <strong>create_loss(self, features, mode, logits, labels)</strong>:
    Implementations do not have to implement this method, but we found that it
    helps streamline the code. Additionally, it can be used by framework
    developers in cases that a full <code>EstimatorSpec</code> is not needed.
    
    Returns a namedtuple that contains
    *   reduced weighted training loss: A scalar Tensor.
    *   unreduced unweighted loss: Typically of shape [batch_size, 1].
    *   weights: Typically, can be 1.0, or Tensor of shape [batch_size, 1].
    *   processed labels: E.g. if the head uses label vocabulary, these are the
    labels after the vocabulary lookup.
    
*   <strong>create_estimator_spec(self, features, mode, logits, labels=None,
    optimizer=None, train_op_fn=None, regularization_losses=None):
    </strong>Returns <code>EstimatorSpec</code>. This is the main method that
    needs to be implemented for head to work with the tf.estimator API.

When writing a custom head, developers are encouraged to implement the Head
interface. Once the interface is implemented, the head will automatically:

*   work with multi-head, canned estimators, or other libraries that may be
    developed in the future.
*   be easier for users to use the head and understand what it does, since it
    follows a familiar API.

#### Factory Methods

The following methods that create common heads are exposed under
`tf.contrib.estimator` in
`tensorflow/contrib/estimator/python/estimator/head.py`. We will expose them
under `tf.estimator`.


```python
def multi_class_head(
    n_classes,
    weight_column=None,
    label_vocabulary=None,
    loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE,
    loss_fn=None,
    name=None):
  """Creates a `Head` for multi class classification.

  Uses `sparse_softmax_cross_entropy` loss.
  """


def binary_classification_head(
    weight_column=None,
    thresholds=None,
    label_vocabulary=None,
    loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE,
    loss_fn=None,
    name=None):
  """Creates a `Head` for single label binary classification.

  Uses `sigmoid_cross_entropy_with_logits` loss.
  """


def regression_head(
    weight_column=None,
    label_dimension=1,
    loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE,
    loss_fn=None,
    inverse_link_fn=None,
    name=None):
  """Creates a `Head` for regression using the `mean_squared_error` loss."""


def poisson_regression_head(
    weight_column=None,
    label_dimension=1,
    loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE,
    compute_full_loss=True,
    name=None):
  """Creates a `Head` for poisson regression using `tf.nn.log_poisson_loss`."""


def logistic_regression_head(
    weight_column=None,
    loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE,
    name=None):
  """Creates a `_Head` for logistic regression.

  Uses `sigmoid_cross_entropy_with_logits` loss, which is the same as
  `binary_classification_head`. The differences compared to
  `binary_classification_head` are:

  * Does not support `label_vocabulary`. Instead, labels must be float in the
    range [0, 1].
  * Does not calculate some metrics that do not make sense, such as AUC.
  * In `PREDICT` mode, only returns logits and predictions
    (`=tf.sigmoid(logits)`), whereas `binary_classification_head` also returns
    probabilities, classes, and class_ids.
  * Export output defaults to `RegressionOutput`, whereas
    `binary_classification_head` defaults to `PredictOutput`.
  """


def multi_label_head(
    n_classes,
    weight_column=None,
    thresholds=None,
    label_vocabulary=None,
    loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE,
    loss_fn=None,
    name=None):
  """Creates a `Head` for multi-label classification.

  Uses `sigmoid_cross_entropy` loss averaged over classes.

  Multi-label classification handles the case where output classes are
  idependent. Each example may have zero or more associated labels from a
  discrete set. This is distinct from `multi_class_head` which has exactly one
  label per example.
  """
```


All those heads allow for some customization, especially through the `loss_fn`
argument. `regression_head` also offers an `inverse_link_fn` argument to
implement generalized regression, see
[here](https://en.wikipedia.org/wiki/Generalized_linear_model#Link_function).

#### Multi-Head

This is a Head implementation that combines multiple single heads, and is
appropriate for multi-objective learning. Contrib code is in
`tensorflow/contrib/estimator/python/estimator/multi_head.py`
and will be exposed in `tf.estimator` as a factory method. Each individual head
is associated with a scalar weight specified with `head_weights`.

```python
def multi_head(heads, head_weights=None):
  """Creates a `Head` for multi-objective learning."""
```

When calling `create_estimator_spec` on multi-head, the multi-head calls
`create_estimator_spec` on each individual head, merges the results and returns
one `EstimatorSpec` instance. The procedure for merging the results is:

*   For **TRAIN**:
    *   Sets the loss equal to the weighted sum of the individual head losses,
        using the optional `head_weights`. If `head_weights` was not set, all
        losses are weighted equally.
    *   Merges `training_hooks` and `training_chief_hooks` from all heads
*   For **EVAL**:
    *   Merges the `eval_metric_ops` dicts from all heads. Each head must
        produce metrics that contain the head name to avoid naming collisions.
    *   Adds metrics **loss/head_name** for each head.
    *   Merges losses in the same way as in TRAIN model.
    *   Merges `evaluation_hooks` from all heads.
*   For **PREDICT**:
    *   Merges the `export_outputs` dicts from all heads. The new dict contains
        string keys of the form **head_name/original_key** to prevent naming
        conflicts.
    *   Sets `export_outputs['serving_default']` to the default export output of
        the first head.
    *   Merges the `predictions` dicts from all heads. The new dict contains
        tuple keys of the form `(head_name, original_key)` to prevent naming
        conflicts.


#### Canned Estimators

These cover the case that the user wants to use a common network, such as DNN or
DNNLinearCombined with a head that is not supported by the existing canned
estimators, such as a multi-head. We will expose canned estimators similar to
existing ones, but where the head is a construction argument. Contrib code is in
`tensorflow/contrib/estimator/python/estimator/` and will be
moved to `tensorflow/python/estimator/canned/`.


```python
class DNNEstimator(estimator.Estimator):
  """`Estimator` that uses a DNN network and a custom `Head`."""

  def __init__(
    self,
    head,
    hidden_units,
    feature_columns,
    model_dir=None,
    optimizer='Adagrad',
    activation_fn=nn.relu,
    dropout=None,
    input_layer_partitioner=None,
    warm_start_from=None,
    config=None):


class LinearEstimator(estimator.Estimator):
  """`Estimator` that uses a Linear network and a custom `Head`."""

  def __init__(
    self,
    head,
    feature_columns,
    model_dir=None,
    optimizer='Ftrl',
    config=None,
    partitioner=None,
    warm_start_from=None):


class DNNLinearCombinedEstimator(estimator.Estimator):
  """`Estimator` that uses a DNN-Linear joined network and a custom `Head`."""

  def __init__(
    self,
    head,
    model_dir=None,
    linear_feature_columns=None,
    linear_optimizer='Ftrl',
    dnn_feature_columns=None,
    dnn_optimizer='Adagrad',
    dnn_hidden_units=None,
    dnn_activation_fn=nn.relu,
    dnn_dropout=None,
    input_layer_partitioner=None,
    warm_start_from=None,
    config=None):
```

### Implementation Plan

The code is in `tensorflow/contrib/estimator/python/estimator/`. We will move it
under `tensorflow/python/estimator/canned/` and expose the API under
`tf.estimator`. We will keep the existing API exposed under
`tf.contrib.estimator`.

### User Migration

The API under `tf.contrib.learn` is deprecated. We will provide instructions for
migrating from `tf.contrib.learn` to `tf.estimator`.

New users will be encouraged to start from the core API under `tf.estimator`.

### Relation to Keras

Keras users can use head in the following way:

*   Use `tf.keras.layers` to construct the logits Tensor.
*   Pass the logits Tensor to head.
*   Wrap this inside a `model_fn`.
*   Use this `model_fn` to construct a `tf.estimator.Estimator`.

`keras.Model` offers a multi-output option similar to multi-head, see
[here](https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models).
However, the Head API is more powerful. In addition to metrics and multiple
outputs with different dimensions and losses offered by `keras.Model`, the Head
API handles

*   prediction transformations, such as transforming logits to probabilities.
*   export signatures by specifying the `export_outputs` property. Those exports
    can be served in Servo. Some heads export multiple signatures and set one of
    them as default.


### Alternatives Considered

*   Do nothing: this leads to duplicated and poorly tested code and maintenance
    issues
*   Have independent functions: train, evaluate, predict:
    *   This leads duplicated code in model_fns.
*   The `create_estimator_spec` method could use the final hidden layer as input
    instead of logits. But this would not work for linear and
    dnn-linear-combined models. There is code in `tf.contrib.learn` that accepts
    either `logits` or `logits_input` (final hidden layer). But that choice
    complicates the code significantly.
