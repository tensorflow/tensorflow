A classifier for TensorFlow DNN models.

Example:

```python
education = sparse_column_with_hash_bucket(column_name="education",
                                           hash_bucket_size=1000)
occupation = sparse_column_with_hash_bucket(column_name="occupation",
                                            hash_bucket_size=1000)

education_emb = embedding_column(sparse_id_column=education, dimension=16,
                                 combiner="sum")
occupation_emb = embedding_column(sparse_id_column=occupation, dimension=16,
                                 combiner="sum")

estimator = DNNClassifier(
    feature_columns=[education_emb, occupation_emb],
    hidden_units=[1024, 512, 256])

# Or estimator using the ProximalAdagradOptimizer optimizer with
# regularization.
estimator = DNNClassifier(
    feature_columns=[education_emb, occupation_emb],
    hidden_units=[1024, 512, 256],
    optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001
    ))

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
    if `weight_column_name` is not `None`, a feature with
      `key=weight_column_name` whose value is a `Tensor`.
    for each `column` in `feature_columns`:
    - if `column` is a `SparseColumn`, a feature with `key=column.name`
      whose `value` is a `SparseTensor`.
    - if `column` is a `RealValuedColumn, a feature with `key=column.name`
      whose `value` is a `Tensor`.
    - if `feauture_columns` is None, then `input` must contains only real
      valued `Tensor`.
- - -

#### `tf.contrib.learn.DNNClassifier.__init__(hidden_units, feature_columns=None, model_dir=None, n_classes=2, weight_column_name=None, optimizer=None, activation_fn=relu, dropout=None, config=None)` {#DNNClassifier.__init__}




- - -

#### `tf.contrib.learn.DNNClassifier.bias_` {#DNNClassifier.bias_}




- - -

#### `tf.contrib.learn.DNNClassifier.dnn_bias_` {#DNNClassifier.dnn_bias_}

Returns bias of deep neural network part.


- - -

#### `tf.contrib.learn.DNNClassifier.dnn_weights_` {#DNNClassifier.dnn_weights_}

Returns weights of deep neural network part.


- - -

#### `tf.contrib.learn.DNNClassifier.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=None, steps=None, metrics=None, name=None)` {#DNNClassifier.evaluate}

Evaluates given model with provided evaluation data.

##### Args:


*  <b>`x`</b>: features.
*  <b>`y`</b>: targets.
*  <b>`input_fn`</b>: Input function. If set, `x`, `y`, and `batch_size` must be
    `None`.
*  <b>`feed_fn`</b>: Function creating a feed dict every time it is called. Called
    once per iteration.
*  <b>`batch_size`</b>: minibatch size to use on the input, defaults to first
    dimension of `x`. Must be `None` if `input_fn` is provided.
*  <b>`steps`</b>: Number of steps for which to evaluate model. If `None`, evaluate
    forever.
*  <b>`metrics`</b>: Dict of metric ops to run. If None, the default metric functions
    are used; if {}, no metrics are used.
*  <b>`name`</b>: Name of the evaluation if user needs to run multiple evaluation on
    different data sets, such as evaluate on training data vs test data.

##### Returns:

  Returns `dict` with evaluation results.

##### Raises:


*  <b>`ValueError`</b>: If at least one of `x` or `y` is provided, and at least one of
      `input_fn` or `feed_fn` is provided.


- - -

#### `tf.contrib.learn.DNNClassifier.fit(x=None, y=None, input_fn=None, steps=None, batch_size=None, monitors=None)` {#DNNClassifier.fit}

Trains a model given training data `x` predictions and `y` targets.

##### Args:


*  <b>`x`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
     iterator that returns arrays of features. The training input
     samples for fitting the model. If set, `input_fn` must be `None`.
*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
     iterator that returns array of targets. The training target values
     (class labels in classification, real numbers in regression). If set,
     `input_fn` must be `None`.
*  <b>`input_fn`</b>: Input function. If set, `x`, `y`, and `batch_size` must be
    `None`.
*  <b>`steps`</b>: Number of steps for which to train model. If `None`, train forever.
*  <b>`batch_size`</b>: minibatch size to use on the input, defaults to first
    dimension of `x`. Must be `None` if `input_fn` is provided.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
    inside the training loop.

##### Returns:

  `self`, for chaining.

##### Raises:


*  <b>`ValueError`</b>: If `x` or `y` are not `None` while `input_fn` is not `None`.

##### Raises:


*  <b>`ValueError`</b>: If at least one of `x` and `y` is provided, and `input_fn` is
      provided.


- - -

#### `tf.contrib.learn.DNNClassifier.get_params(deep=True)` {#DNNClassifier.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.DNNClassifier.get_variable_names()` {#DNNClassifier.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.DNNClassifier.get_variable_value(name)` {#DNNClassifier.get_variable_value}

Returns value of the variable given by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.DNNClassifier.linear_bias_` {#DNNClassifier.linear_bias_}

Returns bias of the linear part.


- - -

#### `tf.contrib.learn.DNNClassifier.linear_weights_` {#DNNClassifier.linear_weights_}

Returns weights per feature of the linear part.


- - -

#### `tf.contrib.learn.DNNClassifier.model_dir` {#DNNClassifier.model_dir}




- - -

#### `tf.contrib.learn.DNNClassifier.partial_fit(x=None, y=None, input_fn=None, steps=1, batch_size=None, monitors=None)` {#DNNClassifier.partial_fit}

Incremental fit on a batch of samples.

This method is expected to be called several times consecutively
on different or the same chunks of the dataset. This either can
implement iterative training or out-of-core/online training.

This is especially useful when the whole dataset is too big to
fit in memory at the same time. Or when model is taking long time
to converge, and you want to split up training into subparts.

##### Args:


*  <b>`x`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
    iterator that returns arrays of features. The training input
    samples for fitting the model. If set, `input_fn` must be `None`.
*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
    iterator that returns array of targets. The training target values
    (class label in classification, real numbers in regression). If set,
     `input_fn` must be `None`.
*  <b>`input_fn`</b>: Input function. If set, `x`, `y`, and `batch_size` must be
    `None`.
*  <b>`steps`</b>: Number of steps for which to train model. If `None`, train forever.
*  <b>`batch_size`</b>: minibatch size to use on the input, defaults to first
    dimension of `x`. Must be `None` if `input_fn` is provided.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
    inside the training loop.

##### Returns:

  `self`, for chaining.

##### Raises:


*  <b>`ValueError`</b>: If at least one of `x` and `y` is provided, and `input_fn` is
      provided.


- - -

#### `tf.contrib.learn.DNNClassifier.predict(x=None, input_fn=None, batch_size=None)` {#DNNClassifier.predict}

Returns predictions for given features.

##### Args:


*  <b>`x`</b>: features.
*  <b>`input_fn`</b>: Input function. If set, x must be None.
*  <b>`batch_size`</b>: Override default batch size.

##### Returns:

  Numpy array of predicted classes or regression values.


- - -

#### `tf.contrib.learn.DNNClassifier.predict_proba(x=None, input_fn=None, batch_size=None)` {#DNNClassifier.predict_proba}

Returns prediction probabilities for given features.

##### Args:


*  <b>`x`</b>: features.
*  <b>`input_fn`</b>: Input function. If set, x and y must be None.
*  <b>`batch_size`</b>: Override default batch size.

##### Returns:

  Numpy array of predicted probabilities.


- - -

#### `tf.contrib.learn.DNNClassifier.set_params(**params)` {#DNNClassifier.set_params}

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The former have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

##### Args:


*  <b>`**params`</b>: Parameters.

##### Returns:

  self

##### Raises:


*  <b>`ValueError`</b>: If params contain invalid names.


- - -

#### `tf.contrib.learn.DNNClassifier.weights_` {#DNNClassifier.weights_}




