
- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.__init__(*args, **kwargs)` {#TensorFlowDNNClassifier.__init__}




- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.bias_` {#TensorFlowDNNClassifier.bias_}




- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.dnn_bias_` {#TensorFlowDNNClassifier.dnn_bias_}

Returns bias of deep neural network part.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.dnn_weights_` {#TensorFlowDNNClassifier.dnn_weights_}

Returns weights of deep neural network part.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=None, steps=None, metrics=None, name=None)` {#TensorFlowDNNClassifier.evaluate}

See `Evaluable`.

##### Raises:


*  <b>`ValueError`</b>: If at least one of `x` or `y` is provided, and at least one of
      `input_fn` or `feed_fn` is provided.
      Or if `metrics` is not `None` or `dict`.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.fit(x, y, steps=None, batch_size=None, monitors=None, logdir=None)` {#TensorFlowDNNClassifier.fit}




- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.get_params(deep=True)` {#TensorFlowDNNClassifier.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional

    If `True`, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.get_variable_names()` {#TensorFlowDNNClassifier.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.get_variable_value(name)` {#TensorFlowDNNClassifier.get_variable_value}

Returns value of the variable given by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.linear_bias_` {#TensorFlowDNNClassifier.linear_bias_}

Returns bias of the linear part.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.linear_weights_` {#TensorFlowDNNClassifier.linear_weights_}

Returns weights per feature of the linear part.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.model_dir` {#TensorFlowDNNClassifier.model_dir}




- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.partial_fit(x=None, y=None, input_fn=None, steps=1, batch_size=None, monitors=None)` {#TensorFlowDNNClassifier.partial_fit}

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

#### `tf.contrib.learn.TensorFlowDNNClassifier.predict(x=None, input_fn=None, batch_size=None, outputs=None, axis=1)` {#TensorFlowDNNClassifier.predict}

Predict class or regression for `x`.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.predict_proba(x=None, input_fn=None, batch_size=None, outputs=None)` {#TensorFlowDNNClassifier.predict_proba}




- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.save(path)` {#TensorFlowDNNClassifier.save}

Saves checkpoints and graph to given path.

##### Args:


*  <b>`path`</b>: Folder to save model to.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.set_params(**params)` {#TensorFlowDNNClassifier.set_params}

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

#### `tf.contrib.learn.TensorFlowDNNClassifier.weights_` {#TensorFlowDNNClassifier.weights_}




