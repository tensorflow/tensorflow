Interface for objects that are trainable by, e.g., `Experiment`.
- - -

#### `tf.contrib.learn.Trainable.fit(x=None, y=None, input_fn=None, steps=None, batch_size=None, monitors=None, max_steps=None)` {#Trainable.fit}

Trains a model given training data `x` predictions and `y` labels.

##### Args:


*  <b>`x`</b>: Matrix of shape [n_samples, n_features...] or the dictionary of Matrices.
     Can be iterator that returns arrays of features or dictionary of arrays of features.
     The training input samples for fitting the model. If set, `input_fn` must be `None`.
*  <b>`y`</b>: Vector or matrix [n_samples] or [n_samples, n_outputs] or the dictionary of same.
     Can be iterator that returns array of labels or dictionary of array of labels.
     The training label values (class labels in classification, real numbers in regression).
     If set, `input_fn` must be `None`. Note: For classification, label values must
     be integers representing the class index (i.e. values from 0 to
     n_classes-1).
*  <b>`input_fn`</b>: Input function returning a tuple of:
      features - `Tensor` or dictionary of string feature name to `Tensor`.
      labels - `Tensor` or dictionary of `Tensor` with labels.
    If input_fn is set, `x`, `y`, and `batch_size` must be `None`.
*  <b>`steps`</b>: Number of steps for which to train model. If `None`, train forever.
    'steps' works incrementally. If you call two times fit(steps=10) then
    training occurs in total 20 steps. If you don't want to have incremental
    behaviour please set `max_steps` instead. If set, `max_steps` must be
    `None`.
*  <b>`batch_size`</b>: minibatch size to use on the input, defaults to first
    dimension of `x`. Must be `None` if `input_fn` is provided.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
    inside the training loop.
*  <b>`max_steps`</b>: Number of total steps for which to train model. If `None`,
    train forever. If set, `steps` must be `None`.

    Two calls to `fit(steps=100)` means 200 training
    iterations. On the other hand, two calls to `fit(max_steps=100)` means
    that the second call will not do any iteration since first call did
    all 100 steps.

##### Returns:

  `self`, for chaining.


