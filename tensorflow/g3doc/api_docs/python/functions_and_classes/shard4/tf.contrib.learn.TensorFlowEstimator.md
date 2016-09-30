Base class for all TensorFlow estimators.
- - -

#### `tf.contrib.learn.TensorFlowEstimator.__init__(model_fn, n_classes, batch_size=32, steps=200, optimizer='Adagrad', learning_rate=0.1, clip_gradients=5.0, class_weight=None, continue_training=False, config=None, verbose=1)` {#TensorFlowEstimator.__init__}

Initializes a TensorFlowEstimator instance.

##### Args:


*  <b>`model_fn`</b>: Model function, that takes input `x`, `y` tensors and outputs
    prediction and loss tensors.
*  <b>`n_classes`</b>: Number of classes in the target.
*  <b>`batch_size`</b>: Mini batch size.
*  <b>`steps`</b>: Number of steps to run over data.
*  <b>`optimizer`</b>: Optimizer name (or class), for example "SGD", "Adam",
    "Adagrad".
*  <b>`learning_rate`</b>: If this is constant float value, no decay function is used.
    Instead, a customized decay function can be passed that accepts
    global_step as parameter and returns a Tensor.
    e.g. exponential decay function:

    ````python
    def exp_decay(global_step):
        return tf.train.exponential_decay(
            learning_rate=0.1, global_step,
            decay_steps=2, decay_rate=0.001)
    ````


*  <b>`clip_gradients`</b>: Clip norm of the gradients to this value to stop
    gradient explosion.
*  <b>`class_weight`</b>: None or list of n_classes floats. Weight associated with
    classes for loss computation. If not given, all classes are supposed to
    have weight one.
*  <b>`continue_training`</b>: when continue_training is True, once initialized
    model will be continually trained on every call of fit.
*  <b>`config`</b>: RunConfig object that controls the configurations of the
    session, e.g. num_cores, gpu_memory_fraction, etc.
*  <b>`verbose`</b>: Controls the verbosity, possible values:

    * 0: the algorithm and debug information is muted.
    * 1: trainer prints the progress.
    * 2: log device placement is printed.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.__repr__()` {#TensorFlowEstimator.__repr__}




- - -

#### `tf.contrib.learn.TensorFlowEstimator.config` {#TensorFlowEstimator.config}




- - -

#### `tf.contrib.learn.TensorFlowEstimator.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=None, steps=None, metrics=None, name=None)` {#TensorFlowEstimator.evaluate}

Evaluates given model with provided evaluation data.

See superclass Estimator for more details.

##### Args:


*  <b>`x`</b>: features.
*  <b>`y`</b>: targets.
*  <b>`input_fn`</b>: Input function.
*  <b>`feed_fn`</b>: Function creating a feed dict every time it is called.
*  <b>`batch_size`</b>: minibatch size to use on the input.
*  <b>`steps`</b>: Number of steps for which to evaluate model.
*  <b>`metrics`</b>: Dict of metric ops to run. If None, the default metrics are used.
*  <b>`name`</b>: Name of the evaluation.

##### Returns:

  Returns `dict` with evaluation results.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.export(*args, **kwargs)` {#TensorFlowEstimator.export}

Exports inference graph into given dir. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-23.
Instructions for updating:
The signature of the input_fn accepted by export is changing to be consistent with what's used by tf.Learn Estimator's train/evaluate. input_fn (and in most cases, input_feature_key) will become required args, and use_deprecated_input_fn will default to False and be removed altogether.

    Args:
      export_dir: A string containing a directory to write the exported graph
        and checkpoints.
      input_fn: If `use_deprecated_input_fn` is true, then a function that given
        `Tensor` of `Example` strings, parses it into features that are then
        passed to the model. Otherwise, a function that takes no argument and
        returns a tuple of (features, targets), where features is a dict of
        string key to `Tensor` and targets is a `Tensor` that's currently not
        used (and so can be `None`).
      input_feature_key: Only used if `use_deprecated_input_fn` is false. String
        key into the features dict returned by `input_fn` that corresponds to
        the raw `Example` strings `Tensor` that the exported model will take as
        input. Can only be `None` if you're using a custom `signature_fn` that
        does not use the first arg (examples).
      use_deprecated_input_fn: Determines the signature format of `input_fn`.
      signature_fn: Function that returns a default signature and a named
        signature map, given `Tensor` of `Example` strings, `dict` of `Tensor`s
        for features and `Tensor` or `dict` of `Tensor`s for predictions.
      prediction_key: The key for a tensor in the `predictions` dict (output
        from the `model_fn`) to use as the `predictions` input to the
        `signature_fn`. Optional. If `None`, predictions will pass to
        `signature_fn` without filtering.
      default_batch_size: Default batch size of the `Example` placeholder.
      exports_to_keep: Number of exports to keep.

    Returns:
      The string path to the exported directory. NB: this functionality was
      added ca. 2016/09/25; clients that depend on the return value may need
      to handle the case where this function returns None because subclasses
      are not returning a value.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.fit(x, y, steps=None, monitors=None, logdir=None)` {#TensorFlowEstimator.fit}

Neural network model from provided `model_fn` and training data.

Note: called first time constructs the graph and initializers
variables. Subsequently, it will continue training the same model.
This logic follows partial_fit() interface in scikit-learn.
To restart learning, create new estimator.

##### Args:


*  <b>`x`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
  iterator that returns arrays of features. The training input
  samples for fitting the model.

*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
  iterator that returns array of targets. The training target values
  (class labels in classification, real numbers in regression).

*  <b>`steps`</b>: int, number of steps to train.
         If None or 0, train for `self.steps`.
*  <b>`monitors`</b>: List of `BaseMonitor` objects to print training progress and
    invoke early stopping.
*  <b>`logdir`</b>: the directory to save the log file that can be used for
  optional visualization.

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.get_params(deep=True)` {#TensorFlowEstimator.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional

    If `True`, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.get_tensor(name)` {#TensorFlowEstimator.get_tensor}

Returns tensor by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Tensor.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.get_variable_names()` {#TensorFlowEstimator.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.get_variable_value(name)` {#TensorFlowEstimator.get_variable_value}

Returns value of the variable given by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.model_dir` {#TensorFlowEstimator.model_dir}




- - -

#### `tf.contrib.learn.TensorFlowEstimator.partial_fit(x, y)` {#TensorFlowEstimator.partial_fit}

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
  samples for fitting the model.

*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
  iterator that returns array of targets. The training target values
  (class label in classification, real numbers in regression).

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.predict(x, axis=1, batch_size=None)` {#TensorFlowEstimator.predict}

Predict class or regression for `x`.

For a classification model, the predicted class for each sample in `x` is
returned. For a regression model, the predicted value based on `x` is
returned.

##### Args:


*  <b>`x`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`axis`</b>: Which axis to argmax for classification.
    By default axis 1 (next after batch) is used.
    Use 2 for sequence predictions.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
    it into mini batches. By default the batch_size member
    variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples]. The predicted classes or predicted
  value.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.predict_proba(x, batch_size=None)` {#TensorFlowEstimator.predict_proba}

Predict class probability of the input samples `x`.

##### Args:


*  <b>`x`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
    it into mini batches. By default the batch_size member variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples, n_classes]. The predicted
  probabilities for each class.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.restore(cls, path, config=None)` {#TensorFlowEstimator.restore}

Restores model from give path.

##### Args:


*  <b>`path`</b>: Path to the checkpoints and other model information.
*  <b>`config`</b>: RunConfig object that controls the configurations of the session,
    e.g. num_cores, gpu_memory_fraction, etc. This is allowed to be
      reconfigured.

##### Returns:

  Estimator, object of the subclass of TensorFlowEstimator.

##### Raises:


*  <b>`ValueError`</b>: if `path` does not contain a model definition.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.save(path)` {#TensorFlowEstimator.save}

Saves checkpoints and graph to given path.

##### Args:


*  <b>`path`</b>: Folder to save model to.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.set_params(**params)` {#TensorFlowEstimator.set_params}

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


