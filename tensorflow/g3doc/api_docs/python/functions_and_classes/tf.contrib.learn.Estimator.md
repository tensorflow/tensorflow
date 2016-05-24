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


