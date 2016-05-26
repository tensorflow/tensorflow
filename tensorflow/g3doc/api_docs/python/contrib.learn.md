<!-- This file is machine generated: DO NOT EDIT! -->

# Learn (contrib)
[TOC]

High level API for learning with TensorFlow.

## Estimators

Train and evaluate TensorFlow models.

- - -

### `class tf.contrib.learn.BaseEstimator` {#BaseEstimator}

Abstract BaseEstimator class to train and evaluate TensorFlow models.

Concrete implementation of this class should provide following functions:
  * _get_train_ops
  * _get_eval_ops
  * _get_predict_ops
It may override _get_default_metric_functions.

`Estimator` implemented below is a good example of how to use this class.

Parameters:
  model_dir: Directory to save model parameters, graph and etc.
- - -

#### `tf.contrib.learn.BaseEstimator.__init__(model_dir=None, config=None)` {#BaseEstimator.__init__}




- - -

#### `tf.contrib.learn.BaseEstimator.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=32, steps=None, metrics=None, name=None)` {#BaseEstimator.evaluate}

Evaluates given model with provided evaluation data.

##### Args:


*  <b>`x`</b>: features.
*  <b>`y`</b>: targets.
*  <b>`input_fn`</b>: Input function. If set, x and y must be None.
*  <b>`feed_fn`</b>: Function creating a feed dict every time it is called. Called
    once per iteration.
*  <b>`batch_size`</b>: minibatch size to use on the input, defaults to 32. Ignored
    if input_fn is set.
*  <b>`steps`</b>: Number of steps to evalute for.
*  <b>`metrics`</b>: Dict of metric ops to run. If None, the default metric functions
    are used; if {}, no metrics are used.
*  <b>`name`</b>: Name of the evaluation if user needs to run multiple evaluation on
    different data sets, such as evaluate on training data vs test data.

##### Returns:

  Returns self.

##### Raises:


*  <b>`ValueError`</b>: If x or y are not None while input_fn or feed_fn is not None.


- - -

#### `tf.contrib.learn.BaseEstimator.fit(x, y, steps, batch_size=32, monitors=None)` {#BaseEstimator.fit}

Trains a model given training data X and y.

##### Args:


*  <b>`x`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
     iterator that returns arrays of features. The training input
     samples for fitting the model.
*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
     iterator that returns array of targets. The training target values
     (class labels in classification, real numbers in regression).
*  <b>`steps`</b>: number of steps to train model for.
*  <b>`batch_size`</b>: minibatch size to use on the input, defaults to 32.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
            inside the training loop.

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.BaseEstimator.get_params(deep=True)` {#BaseEstimator.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.BaseEstimator.model_dir` {#BaseEstimator.model_dir}




- - -

#### `tf.contrib.learn.BaseEstimator.partial_fit(x, y, steps=1, batch_size=32, monitors=None)` {#BaseEstimator.partial_fit}

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
*  <b>`steps`</b>: number of steps to train model for.
*  <b>`batch_size`</b>: minibatch size to use on the input, defaults to 32.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
            inside the training loop.

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.BaseEstimator.predict(x=None, input_fn=None, batch_size=None)` {#BaseEstimator.predict}

Returns predictions for given features.

##### Args:


*  <b>`x`</b>: features.
*  <b>`input_fn`</b>: Input function. If set, x must be None.
*  <b>`batch_size`</b>: Override default batch size.

##### Returns:

  Numpy array of predicted classes or regression values.


- - -

#### `tf.contrib.learn.BaseEstimator.set_params(**params)` {#BaseEstimator.set_params}

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The former have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

##### Returns:

  self


- - -

#### `tf.contrib.learn.BaseEstimator.train(input_fn, steps, monitors=None)` {#BaseEstimator.train}

Trains a model given input builder function.

##### Args:


*  <b>`input_fn`</b>: Input builder function, returns tuple of dicts or
            dict and Tensor.
*  <b>`steps`</b>: number of steps to train model for.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
            inside the training loop.

##### Returns:

  Returns self.



- - -

### `class tf.contrib.learn.Estimator` {#Estimator}

Estimator class is the basic TensorFlow model trainer/evaluator.

Parameters:
  model_fn: Model function, takes features and targets tensors or dicts of
            tensors and returns predictions and loss tensors.
            E.g. `(features, targets) -> (predictions, loss)`.
  model_dir: Directory to save model parameters, graph and etc.
  classification: boolean, true if classification problem.
  learning_rate: learning rate for the model.
  optimizer: optimizer for the model, can be:
             string: name of optimizer, like 'SGD', 'Adam', 'Adagrad', 'Ftl',
               'Momentum', 'RMSProp', 'Momentum').
               Full list in contrib/layers/optimizers.py
             class: sub-class of Optimizer
               (like tf.train.GradientDescentOptimizer).
  clip_gradients: clip_norm value for call to `clip_by_global_norm`. None
                  denotes no gradient clipping.
  config: Configuration object.
- - -

#### `tf.contrib.learn.Estimator.__init__(model_fn=None, model_dir=None, classification=True, learning_rate=0.1, optimizer='Adagrad', clip_gradients=None, config=None)` {#Estimator.__init__}




- - -

#### `tf.contrib.learn.Estimator.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=32, steps=None, metrics=None, name=None)` {#Estimator.evaluate}

Evaluates given model with provided evaluation data.

##### Args:


*  <b>`x`</b>: features.
*  <b>`y`</b>: targets.
*  <b>`input_fn`</b>: Input function. If set, x and y must be None.
*  <b>`feed_fn`</b>: Function creating a feed dict every time it is called. Called
    once per iteration.
*  <b>`batch_size`</b>: minibatch size to use on the input, defaults to 32. Ignored
    if input_fn is set.
*  <b>`steps`</b>: Number of steps to evalute for.
*  <b>`metrics`</b>: Dict of metric ops to run. If None, the default metric functions
    are used; if {}, no metrics are used.
*  <b>`name`</b>: Name of the evaluation if user needs to run multiple evaluation on
    different data sets, such as evaluate on training data vs test data.

##### Returns:

  Returns self.

##### Raises:


*  <b>`ValueError`</b>: If x or y are not None while input_fn or feed_fn is not None.


- - -

#### `tf.contrib.learn.Estimator.fit(x, y, steps, batch_size=32, monitors=None)` {#Estimator.fit}

Trains a model given training data X and y.

##### Args:


*  <b>`x`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
     iterator that returns arrays of features. The training input
     samples for fitting the model.
*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
     iterator that returns array of targets. The training target values
     (class labels in classification, real numbers in regression).
*  <b>`steps`</b>: number of steps to train model for.
*  <b>`batch_size`</b>: minibatch size to use on the input, defaults to 32.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
            inside the training loop.

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.Estimator.get_params(deep=True)` {#Estimator.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.Estimator.model_dir` {#Estimator.model_dir}




- - -

#### `tf.contrib.learn.Estimator.partial_fit(x, y, steps=1, batch_size=32, monitors=None)` {#Estimator.partial_fit}

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
*  <b>`steps`</b>: number of steps to train model for.
*  <b>`batch_size`</b>: minibatch size to use on the input, defaults to 32.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
            inside the training loop.

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.Estimator.predict(x=None, input_fn=None, axis=None, batch_size=None)` {#Estimator.predict}

Returns predictions for given features.

##### Args:


*  <b>`x`</b>: features.
*  <b>`input_fn`</b>: Input function. If set, x must be None.
*  <b>`axis`</b>: Axis on which to argmax (for classification).
        Last axis is used by default.
*  <b>`batch_size`</b>: Override default batch size.

##### Returns:

  Numpy array of predicted classes or regression values.


- - -

#### `tf.contrib.learn.Estimator.predict_proba(x=None, input_fn=None, batch_size=None)` {#Estimator.predict_proba}

Returns prediction probabilities for given features (classification).

##### Args:


*  <b>`x`</b>: features.
*  <b>`input_fn`</b>: Input function. If set, x and y must be None.
*  <b>`batch_size`</b>: Override default batch size.

##### Returns:

  Numpy array of predicted probabilities.


- - -

#### `tf.contrib.learn.Estimator.set_params(**params)` {#Estimator.set_params}

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The former have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

##### Returns:

  self


- - -

#### `tf.contrib.learn.Estimator.train(input_fn, steps, monitors=None)` {#Estimator.train}

Trains a model given input builder function.

##### Args:


*  <b>`input_fn`</b>: Input builder function, returns tuple of dicts or
            dict and Tensor.
*  <b>`steps`</b>: number of steps to train model for.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
            inside the training loop.

##### Returns:

  Returns self.



- - -

### `class tf.contrib.learn.ModeKeys` {#ModeKeys}

Standard names for model modes.

The following standard keys are defined:

* `TRAIN`: training mode.
* `EVAL`: evaluation mode.
* `INFER`: inference mode.

- - -

### `class tf.contrib.learn.TensorFlowClassifier` {#TensorFlowClassifier}

TensorFlow Linear Classifier model.
- - -

#### `tf.contrib.learn.TensorFlowClassifier.__init__(n_classes, batch_size=32, steps=200, optimizer='Adagrad', learning_rate=0.1, class_weight=None, clip_gradients=5.0, continue_training=False, config=None, verbose=1)` {#TensorFlowClassifier.__init__}




- - -

#### `tf.contrib.learn.TensorFlowClassifier.bias_` {#TensorFlowClassifier.bias_}

Returns weights of the linear classifier.


- - -

#### `tf.contrib.learn.TensorFlowClassifier.evaluate(x=None, y=None, input_fn=None, steps=None)` {#TensorFlowClassifier.evaluate}

See base class.


- - -

#### `tf.contrib.learn.TensorFlowClassifier.fit(x, y, steps=None, monitors=None, logdir=None)` {#TensorFlowClassifier.fit}

Builds a neural network model given provided `model_fn` and training
data X and y.

Note: called first time constructs the graph and initializers
variables. Consecutives times it will continue training the same model.
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

#### `tf.contrib.learn.TensorFlowClassifier.get_params(deep=True)` {#TensorFlowClassifier.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.TensorFlowClassifier.get_tensor(name)` {#TensorFlowClassifier.get_tensor}

Returns tensor by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Tensor.


- - -

#### `tf.contrib.learn.TensorFlowClassifier.get_tensor_value(name)` {#TensorFlowClassifier.get_tensor_value}

Returns value of the tensor give by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowClassifier.get_variable_names()` {#TensorFlowClassifier.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.TensorFlowClassifier.model_dir` {#TensorFlowClassifier.model_dir}




- - -

#### `tf.contrib.learn.TensorFlowClassifier.partial_fit(x, y)` {#TensorFlowClassifier.partial_fit}

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

#### `tf.contrib.learn.TensorFlowClassifier.predict(x, axis=1, batch_size=None)` {#TensorFlowClassifier.predict}

Predict class or regression for X.

For a classification model, the predicted class for each sample in X is
returned. For a regression model, the predicted value based on X is
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

#### `tf.contrib.learn.TensorFlowClassifier.predict_proba(x, batch_size=None)` {#TensorFlowClassifier.predict_proba}

Predict class probability of the input samples X.

##### Args:


*  <b>`x`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
    it into mini batches. By default the batch_size member variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples, n_classes]. The predicted
  probabilities for each class.


- - -

#### `tf.contrib.learn.TensorFlowClassifier.restore(cls, path, config=None)` {#TensorFlowClassifier.restore}

Restores model from give path.

##### Args:


*  <b>`path`</b>: Path to the checkpoints and other model information.
*  <b>`config`</b>: RunConfig object that controls the configurations of the session,
    e.g. num_cores, gpu_memory_fraction, etc. This is allowed to be
      reconfigured.

##### Returns:

  Estimator, object of the subclass of TensorFlowEstimator.


- - -

#### `tf.contrib.learn.TensorFlowClassifier.save(path)` {#TensorFlowClassifier.save}

Saves checkpoints and graph to given path.

##### Args:


*  <b>`path`</b>: Folder to save model to.


- - -

#### `tf.contrib.learn.TensorFlowClassifier.set_params(**params)` {#TensorFlowClassifier.set_params}

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The former have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

##### Returns:

  self


- - -

#### `tf.contrib.learn.TensorFlowClassifier.train(input_fn, steps, monitors=None)` {#TensorFlowClassifier.train}

Trains a model given input builder function.

##### Args:


*  <b>`input_fn`</b>: Input builder function, returns tuple of dicts or
            dict and Tensor.
*  <b>`steps`</b>: number of steps to train model for.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
            inside the training loop.

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.TensorFlowClassifier.weights_` {#TensorFlowClassifier.weights_}

Returns weights of the linear classifier.



- - -

### `class tf.contrib.learn.TensorFlowDNNClassifier` {#TensorFlowDNNClassifier}

TensorFlow DNN Classifier model.

Parameters:
  hidden_units: List of hidden units per layer.
  n_classes: Number of classes in the target.
  batch_size: Mini batch size.
  steps: Number of steps to run over data.
  optimizer: Optimizer name (or class), for example "SGD", "Adam", "Adagrad".
  learning_rate: If this is constant float value, no decay function is used.
    Instead, a customized decay function can be passed that accepts
    global_step as parameter and returns a Tensor.
    e.g. exponential decay function:
    def exp_decay(global_step):
        return tf.train.exponential_decay(
            learning_rate=0.1, global_step,
            decay_steps=2, decay_rate=0.001)
  class_weight: None or list of n_classes floats. Weight associated with
    classes for loss computation. If not given, all classes are
    supposed to have weight one.
  continue_training: when continue_training is True, once initialized
    model will be continuely trained on every call of fit.
  config: RunConfig object that controls the configurations of the
    session, e.g. num_cores, gpu_memory_fraction, etc.
  dropout: When not None, the probability we will drop out a given coordinate.
- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.__init__(hidden_units, n_classes, batch_size=32, steps=200, optimizer='Adagrad', learning_rate=0.1, class_weight=None, clip_gradients=5.0, continue_training=False, config=None, verbose=1, dropout=None)` {#TensorFlowDNNClassifier.__init__}




- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.bias_` {#TensorFlowDNNClassifier.bias_}

Returns bias of the DNN's bias layers.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.evaluate(x=None, y=None, input_fn=None, steps=None)` {#TensorFlowDNNClassifier.evaluate}

See base class.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.fit(x, y, steps=None, monitors=None, logdir=None)` {#TensorFlowDNNClassifier.fit}

Builds a neural network model given provided `model_fn` and training
data X and y.

Note: called first time constructs the graph and initializers
variables. Consecutives times it will continue training the same model.
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

#### `tf.contrib.learn.TensorFlowDNNClassifier.get_params(deep=True)` {#TensorFlowDNNClassifier.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.get_tensor(name)` {#TensorFlowDNNClassifier.get_tensor}

Returns tensor by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Tensor.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.get_tensor_value(name)` {#TensorFlowDNNClassifier.get_tensor_value}

Returns value of the tensor give by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.get_variable_names()` {#TensorFlowDNNClassifier.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.model_dir` {#TensorFlowDNNClassifier.model_dir}




- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.partial_fit(x, y)` {#TensorFlowDNNClassifier.partial_fit}

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

#### `tf.contrib.learn.TensorFlowDNNClassifier.predict(x, axis=1, batch_size=None)` {#TensorFlowDNNClassifier.predict}

Predict class or regression for X.

For a classification model, the predicted class for each sample in X is
returned. For a regression model, the predicted value based on X is
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

#### `tf.contrib.learn.TensorFlowDNNClassifier.predict_proba(x, batch_size=None)` {#TensorFlowDNNClassifier.predict_proba}

Predict class probability of the input samples X.

##### Args:


*  <b>`x`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
    it into mini batches. By default the batch_size member variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples, n_classes]. The predicted
  probabilities for each class.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.restore(cls, path, config=None)` {#TensorFlowDNNClassifier.restore}

Restores model from give path.

##### Args:


*  <b>`path`</b>: Path to the checkpoints and other model information.
*  <b>`config`</b>: RunConfig object that controls the configurations of the session,
    e.g. num_cores, gpu_memory_fraction, etc. This is allowed to be
      reconfigured.

##### Returns:

  Estimator, object of the subclass of TensorFlowEstimator.


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

##### Returns:

  self


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.train(input_fn, steps, monitors=None)` {#TensorFlowDNNClassifier.train}

Trains a model given input builder function.

##### Args:


*  <b>`input_fn`</b>: Input builder function, returns tuple of dicts or
            dict and Tensor.
*  <b>`steps`</b>: number of steps to train model for.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
            inside the training loop.

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.weights_` {#TensorFlowDNNClassifier.weights_}

Returns weights of the DNN weight layers.



- - -

### `class tf.contrib.learn.TensorFlowDNNRegressor` {#TensorFlowDNNRegressor}

TensorFlow DNN Regressor model.

Parameters:
  hidden_units: List of hidden units per layer.
  batch_size: Mini batch size.
  steps: Number of steps to run over data.
  optimizer: Optimizer name (or class), for example "SGD", "Adam", "Adagrad".
  learning_rate: If this is constant float value, no decay function is
    used. Instead, a customized decay function can be passed that accepts
    global_step as parameter and returns a Tensor.
    e.g. exponential decay function:
    def exp_decay(global_step):
        return tf.train.exponential_decay(
            learning_rate=0.1, global_step,
            decay_steps=2, decay_rate=0.001)
  continue_training: when continue_training is True, once initialized
    model will be continuely trained on every call of fit.
  config: RunConfig object that controls the configurations of the session,
    e.g. num_cores, gpu_memory_fraction, etc.
  verbose: Controls the verbosity, possible values:
    0: the algorithm and debug information is muted.
    1: trainer prints the progress.
    2: log device placement is printed.
  dropout: When not None, the probability we will drop out a given coordinate.
- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.__init__(hidden_units, n_classes=0, batch_size=32, steps=200, optimizer='Adagrad', learning_rate=0.1, clip_gradients=5.0, continue_training=False, config=None, verbose=1, dropout=None)` {#TensorFlowDNNRegressor.__init__}




- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.bias_` {#TensorFlowDNNRegressor.bias_}

Returns bias of the DNN's bias layers.


- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.evaluate(x=None, y=None, input_fn=None, steps=None)` {#TensorFlowDNNRegressor.evaluate}

See base class.


- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.fit(x, y, steps=None, monitors=None, logdir=None)` {#TensorFlowDNNRegressor.fit}

Builds a neural network model given provided `model_fn` and training
data X and y.

Note: called first time constructs the graph and initializers
variables. Consecutives times it will continue training the same model.
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

#### `tf.contrib.learn.TensorFlowDNNRegressor.get_params(deep=True)` {#TensorFlowDNNRegressor.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.get_tensor(name)` {#TensorFlowDNNRegressor.get_tensor}

Returns tensor by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Tensor.


- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.get_tensor_value(name)` {#TensorFlowDNNRegressor.get_tensor_value}

Returns value of the tensor give by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.get_variable_names()` {#TensorFlowDNNRegressor.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.model_dir` {#TensorFlowDNNRegressor.model_dir}




- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.partial_fit(x, y)` {#TensorFlowDNNRegressor.partial_fit}

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

#### `tf.contrib.learn.TensorFlowDNNRegressor.predict(x, axis=1, batch_size=None)` {#TensorFlowDNNRegressor.predict}

Predict class or regression for X.

For a classification model, the predicted class for each sample in X is
returned. For a regression model, the predicted value based on X is
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

#### `tf.contrib.learn.TensorFlowDNNRegressor.predict_proba(x, batch_size=None)` {#TensorFlowDNNRegressor.predict_proba}

Predict class probability of the input samples X.

##### Args:


*  <b>`x`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
    it into mini batches. By default the batch_size member variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples, n_classes]. The predicted
  probabilities for each class.


- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.restore(cls, path, config=None)` {#TensorFlowDNNRegressor.restore}

Restores model from give path.

##### Args:


*  <b>`path`</b>: Path to the checkpoints and other model information.
*  <b>`config`</b>: RunConfig object that controls the configurations of the session,
    e.g. num_cores, gpu_memory_fraction, etc. This is allowed to be
      reconfigured.

##### Returns:

  Estimator, object of the subclass of TensorFlowEstimator.


- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.save(path)` {#TensorFlowDNNRegressor.save}

Saves checkpoints and graph to given path.

##### Args:


*  <b>`path`</b>: Folder to save model to.


- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.set_params(**params)` {#TensorFlowDNNRegressor.set_params}

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The former have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

##### Returns:

  self


- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.train(input_fn, steps, monitors=None)` {#TensorFlowDNNRegressor.train}

Trains a model given input builder function.

##### Args:


*  <b>`input_fn`</b>: Input builder function, returns tuple of dicts or
            dict and Tensor.
*  <b>`steps`</b>: number of steps to train model for.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
            inside the training loop.

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.weights_` {#TensorFlowDNNRegressor.weights_}

Returns weights of the DNN weight layers.



- - -

### `class tf.contrib.learn.TensorFlowEstimator` {#TensorFlowEstimator}

Base class for all TensorFlow estimators.

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
- - -

#### `tf.contrib.learn.TensorFlowEstimator.__init__(model_fn, n_classes, batch_size=32, steps=200, optimizer='Adagrad', learning_rate=0.1, clip_gradients=5.0, class_weight=None, continue_training=False, config=None, verbose=1)` {#TensorFlowEstimator.__init__}




- - -

#### `tf.contrib.learn.TensorFlowEstimator.evaluate(x=None, y=None, input_fn=None, steps=None)` {#TensorFlowEstimator.evaluate}

See base class.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.fit(x, y, steps=None, monitors=None, logdir=None)` {#TensorFlowEstimator.fit}

Builds a neural network model given provided `model_fn` and training
data X and y.

Note: called first time constructs the graph and initializers
variables. Consecutives times it will continue training the same model.
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
    If True, will return the parameters for this estimator and
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

#### `tf.contrib.learn.TensorFlowEstimator.get_tensor_value(name)` {#TensorFlowEstimator.get_tensor_value}

Returns value of the tensor give by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.get_variable_names()` {#TensorFlowEstimator.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


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

Predict class or regression for X.

For a classification model, the predicted class for each sample in X is
returned. For a regression model, the predicted value based on X is
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

Predict class probability of the input samples X.

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

##### Returns:

  self


- - -

#### `tf.contrib.learn.TensorFlowEstimator.train(input_fn, steps, monitors=None)` {#TensorFlowEstimator.train}

Trains a model given input builder function.

##### Args:


*  <b>`input_fn`</b>: Input builder function, returns tuple of dicts or
            dict and Tensor.
*  <b>`steps`</b>: number of steps to train model for.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
            inside the training loop.

##### Returns:

  Returns self.



- - -

### `class tf.contrib.learn.TensorFlowLinearClassifier` {#TensorFlowLinearClassifier}

TensorFlow Linear Classifier model.
- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.__init__(n_classes, batch_size=32, steps=200, optimizer='Adagrad', learning_rate=0.1, class_weight=None, clip_gradients=5.0, continue_training=False, config=None, verbose=1)` {#TensorFlowLinearClassifier.__init__}




- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.bias_` {#TensorFlowLinearClassifier.bias_}

Returns weights of the linear classifier.


- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.evaluate(x=None, y=None, input_fn=None, steps=None)` {#TensorFlowLinearClassifier.evaluate}

See base class.


- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.fit(x, y, steps=None, monitors=None, logdir=None)` {#TensorFlowLinearClassifier.fit}

Builds a neural network model given provided `model_fn` and training
data X and y.

Note: called first time constructs the graph and initializers
variables. Consecutives times it will continue training the same model.
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

#### `tf.contrib.learn.TensorFlowLinearClassifier.get_params(deep=True)` {#TensorFlowLinearClassifier.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.get_tensor(name)` {#TensorFlowLinearClassifier.get_tensor}

Returns tensor by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Tensor.


- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.get_tensor_value(name)` {#TensorFlowLinearClassifier.get_tensor_value}

Returns value of the tensor give by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.get_variable_names()` {#TensorFlowLinearClassifier.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.model_dir` {#TensorFlowLinearClassifier.model_dir}




- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.partial_fit(x, y)` {#TensorFlowLinearClassifier.partial_fit}

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

#### `tf.contrib.learn.TensorFlowLinearClassifier.predict(x, axis=1, batch_size=None)` {#TensorFlowLinearClassifier.predict}

Predict class or regression for X.

For a classification model, the predicted class for each sample in X is
returned. For a regression model, the predicted value based on X is
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

#### `tf.contrib.learn.TensorFlowLinearClassifier.predict_proba(x, batch_size=None)` {#TensorFlowLinearClassifier.predict_proba}

Predict class probability of the input samples X.

##### Args:


*  <b>`x`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
    it into mini batches. By default the batch_size member variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples, n_classes]. The predicted
  probabilities for each class.


- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.restore(cls, path, config=None)` {#TensorFlowLinearClassifier.restore}

Restores model from give path.

##### Args:


*  <b>`path`</b>: Path to the checkpoints and other model information.
*  <b>`config`</b>: RunConfig object that controls the configurations of the session,
    e.g. num_cores, gpu_memory_fraction, etc. This is allowed to be
      reconfigured.

##### Returns:

  Estimator, object of the subclass of TensorFlowEstimator.


- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.save(path)` {#TensorFlowLinearClassifier.save}

Saves checkpoints and graph to given path.

##### Args:


*  <b>`path`</b>: Folder to save model to.


- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.set_params(**params)` {#TensorFlowLinearClassifier.set_params}

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The former have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

##### Returns:

  self


- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.train(input_fn, steps, monitors=None)` {#TensorFlowLinearClassifier.train}

Trains a model given input builder function.

##### Args:


*  <b>`input_fn`</b>: Input builder function, returns tuple of dicts or
            dict and Tensor.
*  <b>`steps`</b>: number of steps to train model for.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
            inside the training loop.

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.weights_` {#TensorFlowLinearClassifier.weights_}

Returns weights of the linear classifier.



- - -

### `class tf.contrib.learn.TensorFlowLinearRegressor` {#TensorFlowLinearRegressor}

TensorFlow Linear Regression model.
- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.__init__(n_classes=0, batch_size=32, steps=200, optimizer='Adagrad', learning_rate=0.1, clip_gradients=5.0, continue_training=False, config=None, verbose=1)` {#TensorFlowLinearRegressor.__init__}




- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.bias_` {#TensorFlowLinearRegressor.bias_}

Returns bias of the linear regression.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.evaluate(x=None, y=None, input_fn=None, steps=None)` {#TensorFlowLinearRegressor.evaluate}

See base class.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.fit(x, y, steps=None, monitors=None, logdir=None)` {#TensorFlowLinearRegressor.fit}

Builds a neural network model given provided `model_fn` and training
data X and y.

Note: called first time constructs the graph and initializers
variables. Consecutives times it will continue training the same model.
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

#### `tf.contrib.learn.TensorFlowLinearRegressor.get_params(deep=True)` {#TensorFlowLinearRegressor.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.get_tensor(name)` {#TensorFlowLinearRegressor.get_tensor}

Returns tensor by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Tensor.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.get_tensor_value(name)` {#TensorFlowLinearRegressor.get_tensor_value}

Returns value of the tensor give by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.get_variable_names()` {#TensorFlowLinearRegressor.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.model_dir` {#TensorFlowLinearRegressor.model_dir}




- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.partial_fit(x, y)` {#TensorFlowLinearRegressor.partial_fit}

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

#### `tf.contrib.learn.TensorFlowLinearRegressor.predict(x, axis=1, batch_size=None)` {#TensorFlowLinearRegressor.predict}

Predict class or regression for X.

For a classification model, the predicted class for each sample in X is
returned. For a regression model, the predicted value based on X is
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

#### `tf.contrib.learn.TensorFlowLinearRegressor.predict_proba(x, batch_size=None)` {#TensorFlowLinearRegressor.predict_proba}

Predict class probability of the input samples X.

##### Args:


*  <b>`x`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
    it into mini batches. By default the batch_size member variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples, n_classes]. The predicted
  probabilities for each class.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.restore(cls, path, config=None)` {#TensorFlowLinearRegressor.restore}

Restores model from give path.

##### Args:


*  <b>`path`</b>: Path to the checkpoints and other model information.
*  <b>`config`</b>: RunConfig object that controls the configurations of the session,
    e.g. num_cores, gpu_memory_fraction, etc. This is allowed to be
      reconfigured.

##### Returns:

  Estimator, object of the subclass of TensorFlowEstimator.


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

##### Returns:

  self


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.train(input_fn, steps, monitors=None)` {#TensorFlowLinearRegressor.train}

Trains a model given input builder function.

##### Args:


*  <b>`input_fn`</b>: Input builder function, returns tuple of dicts or
            dict and Tensor.
*  <b>`steps`</b>: number of steps to train model for.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
            inside the training loop.

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.weights_` {#TensorFlowLinearRegressor.weights_}

Returns weights of the linear regression.



- - -

### `class tf.contrib.learn.TensorFlowRNNClassifier` {#TensorFlowRNNClassifier}

TensorFlow RNN Classifier model.

Parameters:
  rnn_size: The size for rnn cell, e.g. size of your word embeddings.
  cell_type: The type of rnn cell, including rnn, gru, and lstm.
  num_layers: The number of layers of the rnn model.
  input_op_fn: Function that will transform the input tensor, such as
    creating word embeddings, byte list, etc. This takes
    an argument X for input and returns transformed X.
  bidirectional: boolean, Whether this is a bidirectional rnn.
  sequence_length: If sequence_length is provided, dynamic calculation is
    performed. This saves computational time when unrolling past max sequence
    length.
  initial_state: An initial state for the RNN. This must be a tensor of
    appropriate type and shape [batch_size x cell.state_size].
  n_classes: Number of classes in the target.
  batch_size: Mini batch size.
  steps: Number of steps to run over data.
  optimizer: Optimizer name (or class), for example "SGD", "Adam", "Adagrad".
  learning_rate: If this is constant float value, no decay function is
    used. Instead, a customized decay function can be passed that accepts
    global_step as parameter and returns a Tensor.
    e.g. exponential decay function:
    def exp_decay(global_step):
        return tf.train.exponential_decay(
            learning_rate=0.1, global_step,
            decay_steps=2, decay_rate=0.001)
  class_weight: None or list of n_classes floats. Weight associated with
    classes for loss computation. If not given, all classes are
    supposed to have weight one.
  continue_training: when continue_training is True, once initialized
    model will be continuely trained on every call of fit.
  config: RunConfig object that controls the configurations of the session,
    e.g. num_cores, gpu_memory_fraction, etc.
- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.__init__(rnn_size, n_classes, cell_type='gru', num_layers=1, input_op_fn=null_input_op_fn, initial_state=None, bidirectional=False, sequence_length=None, batch_size=32, steps=50, optimizer='Adagrad', learning_rate=0.1, class_weight=None, clip_gradients=5.0, continue_training=False, config=None, verbose=1)` {#TensorFlowRNNClassifier.__init__}




- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.bias_` {#TensorFlowRNNClassifier.bias_}

Returns bias of the rnn layer.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.evaluate(x=None, y=None, input_fn=None, steps=None)` {#TensorFlowRNNClassifier.evaluate}

See base class.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.fit(x, y, steps=None, monitors=None, logdir=None)` {#TensorFlowRNNClassifier.fit}

Builds a neural network model given provided `model_fn` and training
data X and y.

Note: called first time constructs the graph and initializers
variables. Consecutives times it will continue training the same model.
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

#### `tf.contrib.learn.TensorFlowRNNClassifier.get_params(deep=True)` {#TensorFlowRNNClassifier.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.get_tensor(name)` {#TensorFlowRNNClassifier.get_tensor}

Returns tensor by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Tensor.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.get_tensor_value(name)` {#TensorFlowRNNClassifier.get_tensor_value}

Returns value of the tensor give by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.get_variable_names()` {#TensorFlowRNNClassifier.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.model_dir` {#TensorFlowRNNClassifier.model_dir}




- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.partial_fit(x, y)` {#TensorFlowRNNClassifier.partial_fit}

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

#### `tf.contrib.learn.TensorFlowRNNClassifier.predict(x, axis=1, batch_size=None)` {#TensorFlowRNNClassifier.predict}

Predict class or regression for X.

For a classification model, the predicted class for each sample in X is
returned. For a regression model, the predicted value based on X is
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

#### `tf.contrib.learn.TensorFlowRNNClassifier.predict_proba(x, batch_size=None)` {#TensorFlowRNNClassifier.predict_proba}

Predict class probability of the input samples X.

##### Args:


*  <b>`x`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
    it into mini batches. By default the batch_size member variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples, n_classes]. The predicted
  probabilities for each class.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.restore(cls, path, config=None)` {#TensorFlowRNNClassifier.restore}

Restores model from give path.

##### Args:


*  <b>`path`</b>: Path to the checkpoints and other model information.
*  <b>`config`</b>: RunConfig object that controls the configurations of the session,
    e.g. num_cores, gpu_memory_fraction, etc. This is allowed to be
      reconfigured.

##### Returns:

  Estimator, object of the subclass of TensorFlowEstimator.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.save(path)` {#TensorFlowRNNClassifier.save}

Saves checkpoints and graph to given path.

##### Args:


*  <b>`path`</b>: Folder to save model to.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.set_params(**params)` {#TensorFlowRNNClassifier.set_params}

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The former have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

##### Returns:

  self


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.train(input_fn, steps, monitors=None)` {#TensorFlowRNNClassifier.train}

Trains a model given input builder function.

##### Args:


*  <b>`input_fn`</b>: Input builder function, returns tuple of dicts or
            dict and Tensor.
*  <b>`steps`</b>: number of steps to train model for.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
            inside the training loop.

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.weights_` {#TensorFlowRNNClassifier.weights_}

Returns weights of the rnn layer.



- - -

### `class tf.contrib.learn.TensorFlowRNNRegressor` {#TensorFlowRNNRegressor}

TensorFlow RNN Regressor model.

Parameters:
  rnn_size: The size for rnn cell, e.g. size of your word embeddings.
  cell_type: The type of rnn cell, including rnn, gru, and lstm.
  num_layers: The number of layers of the rnn model.
  input_op_fn: Function that will transform the input tensor, such as
    creating word embeddings, byte list, etc. This takes
    an argument X for input and returns transformed X.
  bidirectional: boolean, Whether this is a bidirectional rnn.
  sequence_length: If sequence_length is provided, dynamic calculation is
    performed. This saves computational time when unrolling past max sequence
    length.
  initial_state: An initial state for the RNN. This must be a tensor of
    appropriate type and shape [batch_size x cell.state_size].
  batch_size: Mini batch size.
  steps: Number of steps to run over data.
  optimizer: Optimizer name (or class), for example "SGD", "Adam", "Adagrad".
  learning_rate: If this is constant float value, no decay function is
    used. Instead, a customized decay function can be passed that accepts
    global_step as parameter and returns a Tensor.
    e.g. exponential decay function:
    def exp_decay(global_step):
        return tf.train.exponential_decay(
            learning_rate=0.1, global_step,
            decay_steps=2, decay_rate=0.001)
  continue_training: when continue_training is True, once initialized
    model will be continuely trained on every call of fit.
  config: RunConfig object that controls the configurations of the
    session, e.g. num_cores, gpu_memory_fraction, etc.
  verbose: Controls the verbosity, possible values:
    0: the algorithm and debug information is muted.
    1: trainer prints the progress.
    2: log device placement is printed.
- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.__init__(rnn_size, cell_type='gru', num_layers=1, input_op_fn=null_input_op_fn, initial_state=None, bidirectional=False, sequence_length=None, n_classes=0, batch_size=32, steps=50, optimizer='Adagrad', learning_rate=0.1, clip_gradients=5.0, continue_training=False, config=None, verbose=1)` {#TensorFlowRNNRegressor.__init__}




- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.bias_` {#TensorFlowRNNRegressor.bias_}

Returns bias of the rnn layer.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.evaluate(x=None, y=None, input_fn=None, steps=None)` {#TensorFlowRNNRegressor.evaluate}

See base class.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.fit(x, y, steps=None, monitors=None, logdir=None)` {#TensorFlowRNNRegressor.fit}

Builds a neural network model given provided `model_fn` and training
data X and y.

Note: called first time constructs the graph and initializers
variables. Consecutives times it will continue training the same model.
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

#### `tf.contrib.learn.TensorFlowRNNRegressor.get_params(deep=True)` {#TensorFlowRNNRegressor.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.get_tensor(name)` {#TensorFlowRNNRegressor.get_tensor}

Returns tensor by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Tensor.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.get_tensor_value(name)` {#TensorFlowRNNRegressor.get_tensor_value}

Returns value of the tensor give by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.get_variable_names()` {#TensorFlowRNNRegressor.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.model_dir` {#TensorFlowRNNRegressor.model_dir}




- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.partial_fit(x, y)` {#TensorFlowRNNRegressor.partial_fit}

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

#### `tf.contrib.learn.TensorFlowRNNRegressor.predict(x, axis=1, batch_size=None)` {#TensorFlowRNNRegressor.predict}

Predict class or regression for X.

For a classification model, the predicted class for each sample in X is
returned. For a regression model, the predicted value based on X is
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

#### `tf.contrib.learn.TensorFlowRNNRegressor.predict_proba(x, batch_size=None)` {#TensorFlowRNNRegressor.predict_proba}

Predict class probability of the input samples X.

##### Args:


*  <b>`x`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
    it into mini batches. By default the batch_size member variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples, n_classes]. The predicted
  probabilities for each class.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.restore(cls, path, config=None)` {#TensorFlowRNNRegressor.restore}

Restores model from give path.

##### Args:


*  <b>`path`</b>: Path to the checkpoints and other model information.
*  <b>`config`</b>: RunConfig object that controls the configurations of the session,
    e.g. num_cores, gpu_memory_fraction, etc. This is allowed to be
      reconfigured.

##### Returns:

  Estimator, object of the subclass of TensorFlowEstimator.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.save(path)` {#TensorFlowRNNRegressor.save}

Saves checkpoints and graph to given path.

##### Args:


*  <b>`path`</b>: Folder to save model to.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.set_params(**params)` {#TensorFlowRNNRegressor.set_params}

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The former have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

##### Returns:

  self


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.train(input_fn, steps, monitors=None)` {#TensorFlowRNNRegressor.train}

Trains a model given input builder function.

##### Args:


*  <b>`input_fn`</b>: Input builder function, returns tuple of dicts or
            dict and Tensor.
*  <b>`steps`</b>: number of steps to train model for.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
            inside the training loop.

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.weights_` {#TensorFlowRNNRegressor.weights_}

Returns weights of the rnn layer.



- - -

### `class tf.contrib.learn.TensorFlowRegressor` {#TensorFlowRegressor}

TensorFlow Linear Regression model.
- - -

#### `tf.contrib.learn.TensorFlowRegressor.__init__(n_classes=0, batch_size=32, steps=200, optimizer='Adagrad', learning_rate=0.1, clip_gradients=5.0, continue_training=False, config=None, verbose=1)` {#TensorFlowRegressor.__init__}




- - -

#### `tf.contrib.learn.TensorFlowRegressor.bias_` {#TensorFlowRegressor.bias_}

Returns bias of the linear regression.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.evaluate(x=None, y=None, input_fn=None, steps=None)` {#TensorFlowRegressor.evaluate}

See base class.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.fit(x, y, steps=None, monitors=None, logdir=None)` {#TensorFlowRegressor.fit}

Builds a neural network model given provided `model_fn` and training
data X and y.

Note: called first time constructs the graph and initializers
variables. Consecutives times it will continue training the same model.
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

#### `tf.contrib.learn.TensorFlowRegressor.get_tensor(name)` {#TensorFlowRegressor.get_tensor}

Returns tensor by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Tensor.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.get_tensor_value(name)` {#TensorFlowRegressor.get_tensor_value}

Returns value of the tensor give by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.get_variable_names()` {#TensorFlowRegressor.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.model_dir` {#TensorFlowRegressor.model_dir}




- - -

#### `tf.contrib.learn.TensorFlowRegressor.partial_fit(x, y)` {#TensorFlowRegressor.partial_fit}

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

#### `tf.contrib.learn.TensorFlowRegressor.predict(x, axis=1, batch_size=None)` {#TensorFlowRegressor.predict}

Predict class or regression for X.

For a classification model, the predicted class for each sample in X is
returned. For a regression model, the predicted value based on X is
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

#### `tf.contrib.learn.TensorFlowRegressor.predict_proba(x, batch_size=None)` {#TensorFlowRegressor.predict_proba}

Predict class probability of the input samples X.

##### Args:


*  <b>`x`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
    it into mini batches. By default the batch_size member variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples, n_classes]. The predicted
  probabilities for each class.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.restore(cls, path, config=None)` {#TensorFlowRegressor.restore}

Restores model from give path.

##### Args:


*  <b>`path`</b>: Path to the checkpoints and other model information.
*  <b>`config`</b>: RunConfig object that controls the configurations of the session,
    e.g. num_cores, gpu_memory_fraction, etc. This is allowed to be
      reconfigured.

##### Returns:

  Estimator, object of the subclass of TensorFlowEstimator.


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

##### Returns:

  self


- - -

#### `tf.contrib.learn.TensorFlowRegressor.train(input_fn, steps, monitors=None)` {#TensorFlowRegressor.train}

Trains a model given input builder function.

##### Args:


*  <b>`input_fn`</b>: Input builder function, returns tuple of dicts or
            dict and Tensor.
*  <b>`steps`</b>: number of steps to train model for.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
            inside the training loop.

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.weights_` {#TensorFlowRegressor.weights_}

Returns weights of the linear regression.




## Graph actions

Perform various training, evaluation, and inference actions on a graph.

- - -

### `class tf.contrib.learn.NanLossDuringTrainingError` {#NanLossDuringTrainingError}



- - -

### `class tf.contrib.learn.RunConfig` {#RunConfig}

This class specifies the specific configurations for the run.

Parameters:
  execution_mode: Runners use this flag to execute different tasks, like
    training vs evaluation. 'all' (the default) executes both training and
    eval.
  master: TensorFlow master. Empty string (the default) for local.
  task: Task id of the replica running the training (default: 0).
  num_ps_replicas: Number of parameter server tasks to use (default: 0).
  training_worker_session_startup_stagger_secs: Seconds to sleep between the
    startup of each worker task session (default: 5).
  training_worker_max_startup_secs: Max seconds to wait before starting any
    worker (default: 60).
  eval_delay_secs: Number of seconds between the beginning of each eval run.
    If one run takes more than this amount of time, the next run will start
    immediately once that run completes (default 60).
  eval_steps: Number of steps to run in each eval (default: 100).
  num_cores: Number of cores to be used (default: 4).
  verbose: Controls the verbosity, possible values:
    0: the algorithm and debug information is muted.
    1: trainer prints the progress.
    2: log device placement is printed.
  gpu_memory_fraction: Fraction of GPU memory used by the process on
    each GPU uniformly on the same machine.
  tf_random_seed: Random seed for TensorFlow initializers.
    Setting this value allows consistency between reruns.
  keep_checkpoint_max: The maximum number of recent checkpoint files to keep.
    As new files are created, older files are deleted.
    If None or 0, all checkpoint files are kept.
    Defaults to 5 (that is, the 5 most recent checkpoint files are kept.)
  keep_checkpoint_every_n_hours: Number of hours between each checkpoint
    to be saved. The default value of 10,000 hours effectively disables
    the feature.

Attributes:
  tf_master: Tensorflow master.
  tf_config: Tensorflow Session Config proto.
  tf_random_seed: Tensorflow random seed.
  keep_checkpoint_max: Maximum number of checkpoints to keep.
  keep_checkpoint_every_n_hours: Number of hours between each checkpoint.
- - -

#### `tf.contrib.learn.RunConfig.__init__(execution_mode='all', master='', task=0, num_ps_replicas=0, training_worker_session_startup_stagger_secs=5, training_worker_max_startup_secs=60, eval_delay_secs=60, eval_steps=100, num_cores=4, verbose=1, gpu_memory_fraction=1, tf_random_seed=42, keep_checkpoint_max=5, keep_checkpoint_every_n_hours=10000)` {#RunConfig.__init__}





- - -

### `tf.contrib.learn.evaluate(graph, output_dir, checkpoint_path, eval_dict, update_op=None, global_step_tensor=None, supervisor_master='', log_every_steps=10, feed_fn=None, max_steps=None)` {#evaluate}

Evaluate a model loaded from a checkpoint.

Given `graph`, a directory to write summaries to (`output_dir`), a checkpoint
to restore variables from, and a `dict` of `Tensor`s to evaluate, run an eval
loop for `max_steps` steps.

In each step of evaluation, all tensors in the `eval_dict` are evaluated, and
every `log_every_steps` steps, they are logged. At the very end of evaluation,
a summary is evaluated (finding the summary ops using `Supervisor`'s logic)
and written to `output_dir`.

##### Args:


*  <b>`graph`</b>: A `Graph` to train. It is expected that this graph is not in use
    elsewhere.
*  <b>`output_dir`</b>: A string containing the directory to write a summary to.
*  <b>`checkpoint_path`</b>: A string containing the path to a checkpoint to restore.
    Can be `None` if the graph doesn't require loading any variables.
*  <b>`eval_dict`</b>: A `dict` mapping string names to tensors to evaluate. It is
    evaluated in every logging step. The result of the final evaluation is
    returned. If update_op is None, then it's evaluated in every step.
*  <b>`update_op`</b>: A `Tensor` which is run in every step.
*  <b>`global_step_tensor`</b>: A `Variable` containing the global step. If `None`,
    one is extracted from the graph using the same logic as in `Supervisor`.
    Used to place eval summaries on training curves.
*  <b>`supervisor_master`</b>: The master string to use when preparing the session.
*  <b>`log_every_steps`</b>: Integer. Output logs every `log_every_steps` evaluation
    steps. The logs contain the `eval_dict` and timing information.
*  <b>`feed_fn`</b>: A function that is called every iteration to produce a `feed_dict`
    passed to `session.run` calls. Optional.
*  <b>`max_steps`</b>: Integer. Evaluate `eval_dict` this many times.

##### Returns:

  A tuple `(eval_results, global_step)`:

*  <b>`eval_results`</b>: A `dict` mapping `string` to numeric values (`int`, `float`)
    that are the result of running eval_dict in the last step. `None` if no
    eval steps were run.
*  <b>`global_step`</b>: The global step this evaluation corresponds to.


- - -

### `tf.contrib.learn.infer(restore_checkpoint_path, output_dict, feed_dict=None)` {#infer}




- - -

### `tf.contrib.learn.run_feeds(output_dict, feed_dicts, restore_checkpoint_path=None)` {#run_feeds}

Run `output_dict` tensors with each input in `feed_dicts`.

If `checkpoint_path` is supplied, restore from checkpoint. Otherwise, init all
variables.

##### Args:


*  <b>`output_dict`</b>: A `dict` mapping string names to `Tensor` objects to run.
    Tensors must all be from the same graph.
*  <b>`feed_dicts`</b>: Iterable of `dict` objects of input values to feed.
*  <b>`restore_checkpoint_path`</b>: A string containing the path to a checkpoint to
    restore.

##### Returns:

  A list of dicts of values read from `output_dict` tensors, one item in the
  list for each item in `feed_dicts`. Keys are the same as `output_dict`,
  values are the results read from the corresponding `Tensor` in
  `output_dict`.

##### Raises:


*  <b>`ValueError`</b>: if `output_dict` or `feed_dicts` is None or empty.


- - -

### `tf.contrib.learn.run_n(output_dict, feed_dict=None, restore_checkpoint_path=None, n=1)` {#run_n}

Run `output_dict` tensors `n` times, with the same `feed_dict` each run.

##### Args:


*  <b>`output_dict`</b>: A `dict` mapping string names to tensors to run. Must all be
    from the same graph.
*  <b>`feed_dict`</b>: `dict` of input values to feed each run.
*  <b>`restore_checkpoint_path`</b>: A string containing the path to a checkpoint to
    restore.
*  <b>`n`</b>: Number of times to repeat.

##### Returns:

  A list of `n` `dict` objects, each containing values read from `output_dict`
  tensors.


- - -

### `tf.contrib.learn.train(graph, output_dir, train_op, loss_op, global_step_tensor=None, init_op=None, init_feed_dict=None, init_fn=None, log_every_steps=10, supervisor_is_chief=True, supervisor_master='', supervisor_save_model_secs=600, supervisor_save_summaries_steps=100, feed_fn=None, max_steps=None, fail_on_nan_loss=True, monitors=None)` {#train}

Train a model.

Given `graph`, a directory to write outputs to (`output_dir`), and some ops,
run a training loop. The given `train_op` performs one step of training on the
model. The `loss_op` represents the objective function of the training. It is
expected to increment the `global_step_tensor`, a scalar integer tensor
counting training steps. This function uses `Supervisor` to initialize the
graph (from a checkpoint if one is available in `output_dir`), write summaries
defined in the graph, and write regular checkpoints as defined by
`supervisor_save_model_secs`.

Training continues until `global_step_tensor` evaluates to `max_steps`, or, if
`fail_on_nan_loss`, until `loss_op` evaluates to `NaN`. In that case the
program is terminated with exit code 1.

##### Args:


*  <b>`graph`</b>: A graph to train. It is expected that this graph is not in use
    elsewhere.
*  <b>`output_dir`</b>: A directory to write outputs to.
*  <b>`train_op`</b>: An op that performs one training step when run.
*  <b>`loss_op`</b>: A scalar loss tensor.
*  <b>`global_step_tensor`</b>: A tensor representing the global step. If none is given,
    one is extracted from the graph using the same logic as in `Supervisor`.
*  <b>`init_op`</b>: An op that initializes the graph. If `None`, use `Supervisor`'s
    default.
*  <b>`init_feed_dict`</b>: A dictionary that maps `Tensor` objects to feed values.
    This feed dictionary will be used when `init_op` is evaluated.
*  <b>`init_fn`</b>: Optional callable passed to Supervisor to initialize the model.
*  <b>`log_every_steps`</b>: Output logs regularly. The logs contain timing data and the
    current loss.
*  <b>`supervisor_is_chief`</b>: Whether the current process is the chief supervisor in
    charge of restoring the model and running standard services.
*  <b>`supervisor_master`</b>: The master string to use when preparing the session.
*  <b>`supervisor_save_model_secs`</b>: Save a checkpoint every
    `supervisor_save_model_secs` seconds when training.
*  <b>`supervisor_save_summaries_steps`</b>: Save summaries every
    `supervisor_save_summaries_steps` seconds when training.
*  <b>`feed_fn`</b>: A function that is called every iteration to produce a `feed_dict`
    passed to `session.run` calls. Optional.
*  <b>`max_steps`</b>: Train until `global_step_tensor` evaluates to this value.
*  <b>`fail_on_nan_loss`</b>: If true, raise `NanLossDuringTrainingError` if `loss_op`
    evaluates to `NaN`. If false, continue training as if nothing happened.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
    inside the training loop.

##### Returns:

  The final loss value.

##### Raises:


*  <b>`ValueError`</b>: If `global_step_tensor` is not provided. See
      `tf.contrib.framework.get_global_step` for how we look it up if not
      provided explicitly.
*  <b>`NanLossDuringTrainingError`</b>: If `fail_on_nan_loss` is `True`, and loss ever
      evaluates to `NaN`.



## Input processing

Queue and read batched input data.

- - -

### `tf.contrib.learn.extract_dask_data(data)` {#extract_dask_data}

Extract data from dask.Series or dask.DataFrame for predictors


- - -

### `tf.contrib.learn.extract_dask_labels(labels)` {#extract_dask_labels}

Extract data from dask.Series for labels


- - -

### `tf.contrib.learn.extract_pandas_data(data)` {#extract_pandas_data}

Extract data from pandas.DataFrame for predictors


- - -

### `tf.contrib.learn.extract_pandas_labels(labels)` {#extract_pandas_labels}

Extract data from pandas.DataFrame for labels


- - -

### `tf.contrib.learn.extract_pandas_matrix(data)` {#extract_pandas_matrix}

Extracts numpy matrix from pandas DataFrame.


- - -

### `tf.contrib.learn.read_batch_examples(file_pattern, batch_size, reader, randomize_input=True, num_epochs=None, queue_capacity=10000, num_threads=1, name=None)` {#read_batch_examples}

Adds operations to read, queue, batch `Example` protos.

Given file pattern (or list of files), will setup a queue for file names,
read `Example` proto using provided `reader`, use batch queue to create
batches of examples of size `batch_size`.

All queue runners are added to the queue runners collection, and may be
started via `start_queue_runners`.

All ops are added to the default graph.

##### Args:


*  <b>`file_pattern`</b>: List of files or pattern of file paths containing
      `Example` records. See `tf.gfile.Glob` for pattern rules.
*  <b>`batch_size`</b>: An int or scalar `Tensor` specifying the batch size to use.
*  <b>`reader`</b>: A function or class that returns an object with
    `read` method, (filename tensor) -> (example tensor).
*  <b>`randomize_input`</b>: Whether the input should be randomized.
*  <b>`num_epochs`</b>: Integer specifying the number of times to read through the
    dataset. If `None`, cycles through the dataset forever.
    NOTE - If specified, creates a variable that must be initialized, so call
    `tf.initialize_all_variables()` as shown in the tests.
*  <b>`queue_capacity`</b>: Capacity for input queue.
*  <b>`num_threads`</b>: The number of threads enqueuing examples.
*  <b>`name`</b>: Name of resulting op.

##### Returns:

  String `Tensor` of batched `Example` proto.

##### Raises:


*  <b>`ValueError`</b>: for invalid inputs.


- - -

### `tf.contrib.learn.read_batch_features(file_pattern, batch_size, features, reader, randomize_input=True, num_epochs=None, queue_capacity=10000, reader_num_threads=1, parser_num_threads=1, name=None)` {#read_batch_features}

Adds operations to read, queue, batch and parse `Example` protos.

Given file pattern (or list of files), will setup a queue for file names,
read `Example` proto using provided `reader`, use batch queue to create
batches of examples of size `batch_size` and parse example given `features`
specification.

All queue runners are added to the queue runners collection, and may be
started via `start_queue_runners`.

All ops are added to the default graph.

##### Args:


*  <b>`file_pattern`</b>: List of files or pattern of file paths containing
      `Example` records. See `tf.gfile.Glob` for pattern rules.
*  <b>`batch_size`</b>: An int or scalar `Tensor` specifying the batch size to use.
*  <b>`features`</b>: A `dict` mapping feature keys to `FixedLenFeature` or
    `VarLenFeature` values.
*  <b>`reader`</b>: A function or class that returns an object with
    `read` method, (filename tensor) -> (example tensor).
*  <b>`randomize_input`</b>: Whether the input should be randomized.
*  <b>`num_epochs`</b>: Integer specifying the number of times to read through the
    dataset. If None, cycles through the dataset forever. NOTE - If specified,
    creates a variable that must be initialized, so call
    tf.initialize_all_variables() as shown in the tests.
*  <b>`queue_capacity`</b>: Capacity for input queue.
*  <b>`reader_num_threads`</b>: The number of threads to read examples.
*  <b>`parser_num_threads`</b>: The number of threads to parse examples.
*  <b>`name`</b>: Name of resulting op.

##### Returns:

  A dict of `Tensor` or `SparseTensor` objects for each in `features`.

##### Raises:


*  <b>`ValueError`</b>: for invalid inputs.


- - -

### `tf.contrib.learn.read_batch_record_features(file_pattern, batch_size, features, randomize_input=True, num_epochs=None, queue_capacity=10000, reader_num_threads=1, parser_num_threads=1, name='dequeue_record_examples')` {#read_batch_record_features}

Reads TFRecord, queues, batches and parses `Example` proto.

See more detailed description in `read_examples`.

##### Args:


*  <b>`file_pattern`</b>: List of files or pattern of file paths containing
      `Example` records. See `tf.gfile.Glob` for pattern rules.
*  <b>`batch_size`</b>: An int or scalar `Tensor` specifying the batch size to use.
*  <b>`features`</b>: A `dict` mapping feature keys to `FixedLenFeature` or
    `VarLenFeature` values.
*  <b>`randomize_input`</b>: Whether the input should be randomized.
*  <b>`num_epochs`</b>: Integer specifying the number of times to read through the
    dataset. If None, cycles through the dataset forever. NOTE - If specified,
    creates a variable that must be initialized, so call
    tf.initialize_all_variables() as shown in the tests.
*  <b>`queue_capacity`</b>: Capacity for input queue.
*  <b>`reader_num_threads`</b>: The number of threads to read examples.
*  <b>`parser_num_threads`</b>: The number of threads to parse examples.
*  <b>`name`</b>: Name of resulting op.

##### Returns:

  A dict of `Tensor` or `SparseTensor` objects for each in `features`.

##### Raises:


*  <b>`ValueError`</b>: for invalid inputs.


