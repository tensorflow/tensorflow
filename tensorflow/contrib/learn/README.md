EVERYTHING IN THIS DIRECTORY IS DEPRECATED.

Using functions or classes will result in warnings.

Instructions for converting to current alternatives are included in the
warnings. A high-level overview is below.

## Canned Estimators

Many canned estimators (subclasses of `Estimator`) have equivalents in core
exposed under `tf.estimator`:
`DNNClassifier`, `DNNRegressor`, `DNNEstimator`, `LinearClassifier`,
`LinearRegressor`, `LinearEstimator`, `DNNLinearCombinedClassifier`,
`DNNLinearCombinedRegressor` and `DNNLinearCombinedEstimator`.

To migrate to the new api, users need to take the following steps:

* Replace `tf.contrib.learn` with `tf.estimator`.
* If you subclass any of the estimators, stop doing that. You should be able to
  write a factory method that returns a canned estimator instead. If this is not
  possible (if you override methods from the canned estimator), consider writing
  a custom estimator instead. See `tf.estimator.Estimator`.
* Set `loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE` to preserve loss
  reduction as the average over batch.
* Some optimizer-related arguments are no longer passed in the estimator
  constructor. Instead, we provide methods that perform the same job by wrapping
  an optimizer. Specifically:
  *  `gradient_clip_norm`: Use `tf.contrib.estimator.clip_gradients_by_norm`
  *  `embedding_lr_multipliers`: Not supported.
  Other arguments:
  * `input_layer_min_slice_size`: Replaced by `input_layer_partitioner`
  * `enable_centered_bias`: Not supported. Dropping this argument is unlikely to
    harm your model.
  * `feature_engineering_fn`: Not supported. You can call your
    `feature_engineering_fn` inside your input_fn:
    ```python
    def new_input_fn():
      features, labels = old_input_fn()
      return feature_engineering_fn(features, labels)
    ```
* Use `tf.reshape` to reshape labels in your `input_fn`. `tf.estimator`
  classifiers and regressors expect labels as a 2D Tensor of shape
  `[batch_size, 1]`, or `[batch_size, n_labels]`. In contrast,
  `tf.contrib.learn` classifiers and regressors supported labels with shape
  `[batch_size]`.
* If you pass custom metrics from the `evaluate()` method call, use
  `tf.estimator.add_metrics`.
* Replace your `serving_input_fn` with a `serving_input_receiver_fn`.
  Note this should be entirely distinct from your training `input_fn`, so if you
  previously had one `input_fn` with different "modes", you should now factor
  that apart.  Where the former returned either a simple `(features, labels)`
  tuple or `InputFnOps`, you should now return a `ServingInputReceiver`.
  If you were generating your `serving_input_fn` using the
  `build_parsing_serving_input_fn` helper, you can simply drop in the
  replacement `build_parsing_serving_input_receiver_fn`.

Some remaining estimators/classes:

* `DynamicRnnEstimator`:  Consider a custom `model_fn`.
* `KMeansClustering`: Use `tf.contrib.factorization.KMeansClustering`.
* `LogisticRegressor`: Not supported. Instead, use `binary_classification_head`
  with a custom `model_fn`, or with `DNNEstimator`.
* `StateSavingRnnEstimator`: Consider a custom `model_fn`.
* SVM: Consider a custom `model_fn`.
* `LinearComposableModel` and `DNNComposableModel`: Not supported.
  Consider `tf.contrib.estimator.DNNEstimator`, or write a custom model_fn.
* `MetricSpec`: Deprecated. For adding custom metrics to canned Estimators, use
  `tf.estimator.add_metrics`.

## Estimator
`tf.contrib.learn.Estimator` is migrated to `tf.estimator.Estimator`.

To migrate, users need to take the following steps:

* Replace `tf.contrib.learn.Estimator` with `tf.estimator.Estimator`.
* If you pass a `config` argument to `Estimator`, this must be
  `tf.estimator.RunConfig`. You may need to edit your code accordingly.
* Edit your `model_fn` to return `tf.estimator.EstimatorSpec`. Refer to
  `EstimatorSpec` for documentation of specific fields.
* If your `model_fn` uses the `mode` argument, use `tf.estimator.ModeKeys`.

Some related classes:
* `Evaluable`, `Trainable`: Not supported, merged into `tf.estimator.Estimator`.
* ExportStrategy: Replaced by `tf.estimator.Exporter`.

## Head/MultiHead
These classes are now supported under `tf.contrib.estimator`, e.g.
`tf.contrib.estimator.multi_class_head` and `tf.contrib.estimator.multi_head`.

Some differences:

* `multi_class_head`: If you use `tf.contrib.learn.multi_class_head` with
  `n_classes=2`, switch to `tf.contrib.estimator.binary_classification_head`.
* `loss_only_head`: Not supported.
* `poisson_regression_head`: Not supported (yet).
* `binary_svm_head`: Not supported (yet).
* `no_op_train_fn`: Replace it with `tf.no_op`.

Some arguments are renamed, please refer to documentation. In addition:

* `loss_fn`: Supported for `multi_label_head`. If you need it for other heads,
  please open an issue.
* `metric_class_ids`: Not supported (yet).
* `enable_centered_bias`: Not supported. Dropping this argument is unlikely to
  harm your model.
* `label_name`: Not needed in `tf.estimator`. If you don’t use `multi_head`,
  drop this argument. If you use `multi_head`, refer to
  `tf.contrib.estimator.multi_head` documentation.

## Experiment Class - Distributed Training Tooling

Switch to `tf.estimator.train_and_evaluate`. Some differences:

* Most of the constructor arguments, like `train_input_fn`, `eval_input_fn`,
  should be wrapped into `tf.estimator.TrainSpec` and `tf.estimator.EvalSpec`.
* Remove the `experiment_fn`. Instead, create the `Estimator`,
  `train_spec` and `eval_spec`, then call `tf.estimator.train_and_evaluate`
  directly.
* Inside `tf.estimator.EvalSpec`, the `exporter` field is the replacement
  for `export_strategy`. To be precise, `tf.estimator.LatestExporter` is the
  replacement for `tf.contrib.learn.make_export_strategy`. If you want to export
  only at the end of training  use `tf.estimator.FinalExporter`.
* If the `TF_CONFIG` environment variable is constructed manually, please read
  the `train_and_evaluate` documentation for the new requirementds (in
  particular, the chief node and evaluator node).

## Others Classes and Functions

* `tf.contrib.learn.datasets` is deprecated. We are adding ready to use datasets
  to tensorflow/models. Many smaller datasets are available from other sources,
  such as scikits.learn. Some Python processing may have to be written, but this
  is straightforward to implement using the standard modules.
* `tf.contrib.learn.preprocessing`: Deprecated. The python-only preprocessing
  functions are not a good fit for TensorFlow. Please use `tf.data`, and
  consider tensorflow/transform for more complex use cases.
* `tf.contrib.learn.models`: Not supported, use canned estimators instead.
* `tf.contrib.learn.monitors`: Implement `SessionRunHook` instead. Hook
  implementations are in `tf.train`.
* `tf.contrib.learn.learn_io`: Use the methods in `tf.estimator.inputs`, such as
  `tf.estimator.inputs.numpy_input_fn`. Some utility functions have no
  equivalent, we encourage the use of `tf.data`.

