
- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.__init__(*args, **kwargs)` {#TensorFlowLinearRegressor.__init__}




- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.bias_` {#TensorFlowLinearRegressor.bias_}




- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.dnn_bias_` {#TensorFlowLinearRegressor.dnn_bias_}

Returns bias of deep neural network part.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.dnn_weights_` {#TensorFlowLinearRegressor.dnn_weights_}

Returns weights of deep neural network part.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=None, steps=None, metrics=None, name=None)` {#TensorFlowLinearRegressor.evaluate}

See `Evaluable`.

##### Raises:


*  <b>`ValueError`</b>: If at least one of `x` or `y` is provided, and at least one of
      `input_fn` or `feed_fn` is provided.
      Or if `metrics` is not `None` or `dict`.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.fit(x, y, steps=None, batch_size=None, monitors=None, logdir=None)` {#TensorFlowLinearRegressor.fit}




- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.get_params(deep=True)` {#TensorFlowLinearRegressor.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional

    If `True`, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.get_variable_names()` {#TensorFlowLinearRegressor.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.get_variable_value(name)` {#TensorFlowLinearRegressor.get_variable_value}

Returns value of the variable given by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.linear_bias_` {#TensorFlowLinearRegressor.linear_bias_}

Returns bias of the linear part.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.linear_weights_` {#TensorFlowLinearRegressor.linear_weights_}

Returns weights per feature of the linear part.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.model_dir` {#TensorFlowLinearRegressor.model_dir}




- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.partial_fit(x=None, y=None, input_fn=None, steps=1, batch_size=None, monitors=None)` {#TensorFlowLinearRegressor.partial_fit}

Incremental fit on a batch of samples.

This method is expected to be called several times consecutively
on different or the same chunks of the dataset. This either can
implement iterative training or out-of-core/online training.

This is especially useful when the whole dataset is too big to
fit in memory at the same time. Or when model is taking long time
to converge, and you want to split up training into subparts.

##### Args:


*  <b>`x`</b>: Matrix of shape [n_samples, n_features...]. Can be iterator that
     returns arrays of features. The training input samples for fitting the
     model. If set, `input_fn` must be `None`.
*  <b>`y`</b>: Vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
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


*  <b>`ValueError`</b>: If at least one of `x` and `y` is provided, and `input_fn` is
      provided.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.predict(x=None, input_fn=None, batch_size=None, outputs=None, axis=1)` {#TensorFlowLinearRegressor.predict}

Predict class or regression for `x`.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.predict_proba(x=None, input_fn=None, batch_size=None, outputs=None)` {#TensorFlowLinearRegressor.predict_proba}




- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.save(path)` {#TensorFlowLinearRegressor.save}

Saves checkpoints and graph to given path.

##### Args:


*  <b>`path`</b>: Folder to save model to.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.set_params(**params)` {#TensorFlowLinearRegressor.set_params}

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

#### `tf.contrib.learn.TensorFlowLinearRegressor.weights_` {#TensorFlowLinearRegressor.weights_}




