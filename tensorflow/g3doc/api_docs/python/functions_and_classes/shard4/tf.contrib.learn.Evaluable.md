Interface for objects that are evaluatable by, e.g., `Experiment`.
- - -

#### `tf.contrib.learn.Evaluable.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=None, steps=None, metrics=None, name=None, checkpoint_path=None, hooks=None)` {#Evaluable.evaluate}

Evaluates given model with provided evaluation data.

Stop conditions - we evaluate on the given input data until one of the
following:
- If `steps` is provided, and `steps` batches of size `batch_size` are
processed.
- If `input_fn` is provided, and it raises an end-of-input
exception (`OutOfRangeError` or `StopIteration`).
- If `x` is provided, and all items in `x` have been processed.

The return value is a dict containing the metrics specified in `metrics`, as
well as an entry `global_step` which contains the value of the global step
for which this evaluation was performed.

##### Args:


*  <b>`x`</b>: Matrix of shape [n_samples, n_features...] or dictionary of many matrices
     containing the input samples for fitting the model. Can be iterator that returns
     arrays of features or dictionary of array of features. If set, `input_fn` must
     be `None`.
*  <b>`y`</b>: Vector or matrix [n_samples] or [n_samples, n_outputs] containing the
     label values (class labels in classification, real numbers in
     regression) or dictionary of multiple vectors/matrices. Can be iterator
     that returns array of targets or dictionary of array of targets. If set,
     `input_fn` must be `None`. Note: For classification, label values must
     be integers representing the class index (i.e. values from 0 to
     n_classes-1).
*  <b>`input_fn`</b>: Input function returning a tuple of:
      features - Dictionary of string feature name to `Tensor` or `Tensor`.
      labels - `Tensor` or dictionary of `Tensor` with labels.
    If input_fn is set, `x`, `y`, and `batch_size` must be `None`. If
    `steps` is not provided, this should raise `OutOfRangeError` or
    `StopIteration` after the desired amount of data (e.g., one epoch) has
    been provided. See "Stop conditions" above for specifics.
*  <b>`feed_fn`</b>: Function creating a feed dict every time it is called. Called
    once per iteration. Must be `None` if `input_fn` is provided.
*  <b>`batch_size`</b>: minibatch size to use on the input, defaults to first
    dimension of `x`, if specified. Must be `None` if `input_fn` is
    provided.
*  <b>`steps`</b>: Number of steps for which to evaluate model. If `None`, evaluate
    until `x` is consumed or `input_fn` raises an end-of-input exception.
    See "Stop conditions" above for specifics.
*  <b>`metrics`</b>: Dict of metrics to run. If None, the default metric functions
    are used; if {}, no metrics are used. Otherwise, `metrics` should map
    friendly names for the metric to a `MetricSpec` object defining which
    model outputs to evaluate against which labels with which metric
    function.

    Metric ops should support streaming, e.g., returning `update_op` and
    `value` tensors. For example, see the options defined in
    `../../../metrics/python/ops/metrics_ops.py`.

*  <b>`name`</b>: Name of the evaluation if user needs to run multiple evaluations on
    different data sets, such as on training data vs test data.
*  <b>`checkpoint_path`</b>: Path of a specific checkpoint to evaluate. If `None`, the
    latest checkpoint in `model_dir` is used.
*  <b>`hooks`</b>: List of `SessionRunHook` subclass instances. Used for callbacks
    inside the evaluation call.

##### Returns:

  Returns `dict` with evaluation results.


- - -

#### `tf.contrib.learn.Evaluable.model_dir` {#Evaluable.model_dir}

Returns a path in which the eval process will look for checkpoints.


