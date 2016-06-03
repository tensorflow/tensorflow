
- - -

#### `tf.contrib.learn.TensorFlowRegressor.__init__(*args, **kwargs)` {#TensorFlowRegressor.__init__}




- - -

#### `tf.contrib.learn.TensorFlowRegressor.bias_` {#TensorFlowRegressor.bias_}




- - -

#### `tf.contrib.learn.TensorFlowRegressor.dnn_bias_` {#TensorFlowRegressor.dnn_bias_}

Returns bias of deep neural network part.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.dnn_weights_` {#TensorFlowRegressor.dnn_weights_}

Returns weights of deep neural network part.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=None, steps=None, metrics=None, name=None)` {#TensorFlowRegressor.evaluate}

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

#### `tf.contrib.learn.TensorFlowRegressor.fit(x, y, steps=None, batch_size=None, monitors=None, logdir=None)` {#TensorFlowRegressor.fit}




- - -

#### `tf.contrib.learn.TensorFlowRegressor.get_params(deep=True)` {#TensorFlowRegressor.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.get_variable_names()` {#TensorFlowRegressor.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.get_variable_value(name)` {#TensorFlowRegressor.get_variable_value}

Returns value of the variable given by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.linear_bias_` {#TensorFlowRegressor.linear_bias_}

Returns bias of the linear part.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.linear_weights_` {#TensorFlowRegressor.linear_weights_}

Returns weights per feature of the linear part.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.model_dir` {#TensorFlowRegressor.model_dir}




- - -

#### `tf.contrib.learn.TensorFlowRegressor.partial_fit(x=None, y=None, input_fn=None, steps=1, batch_size=None, monitors=None)` {#TensorFlowRegressor.partial_fit}

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

#### `tf.contrib.learn.TensorFlowRegressor.predict(x=None, input_fn=None, batch_size=None, outputs=None, axis=1)` {#TensorFlowRegressor.predict}




- - -

#### `tf.contrib.learn.TensorFlowRegressor.predict_proba(x=None, input_fn=None, batch_size=None, outputs=None)` {#TensorFlowRegressor.predict_proba}




- - -

#### `tf.contrib.learn.TensorFlowRegressor.save(path)` {#TensorFlowRegressor.save}

Saves checkpoints and graph to given path.

##### Args:


*  <b>`path`</b>: Folder to save model to.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.set_params(**params)` {#TensorFlowRegressor.set_params}

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

#### `tf.contrib.learn.TensorFlowRegressor.weights_` {#TensorFlowRegressor.weights_}




