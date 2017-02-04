#### `tf.contrib.learn.ModelFnOps.__new__(cls, mode, predictions=None, loss=None, train_op=None, eval_metric_ops=None, output_alternatives=None, training_chief_hooks=None, training_hooks=None, scaffold=None)` {#ModelFnOps.__new__}

Creates a validated `ModelFnOps` instance.

For a multi-headed model, the predictions dict here will contain the outputs
of all of the heads.  However: at serving time, requests will be made
specifically for one or more heads, and the RPCs used for these requests may
differ by problem type (i.e., regression, classification, other).  The
purpose of the output_alternatives dict is to aid in exporting a SavedModel
from which such head-specific queries can be served.  These
output_alternatives will be combined with input_alternatives (see
`saved_model_export_utils`) to produce a set of `SignatureDef`s specifying
the valid requests that can be served from this model.

For a single-headed model, it is still adviseable to provide
output_alternatives with a single entry, because this is how the problem
type is communicated for export and serving.  If output_alternatives is not
given, the resulting SavedModel will support only one head of unspecified
type.

##### Args:


*  <b>`mode`</b>: One of `ModeKeys`. Specifies if this training, evaluation or
    prediction.
*  <b>`predictions`</b>: Predictions `Tensor` or dict of `Tensor`.
*  <b>`loss`</b>: Training loss `Tensor`.
*  <b>`train_op`</b>: Op for the training step.
*  <b>`eval_metric_ops`</b>: Dict of metric results keyed by name. The values of the
    dict are the results of calling a metric function, such as `Tensor`.
*  <b>`output_alternatives`</b>: a dict of
    `{submodel_name: (problem_type, {tensor_name: Tensor})}`, where
    `submodel_name` is a submodel identifier that should be consistent
    across the pipeline (here likely taken from the name of each `Head`,
    for models that use them), `problem_type` is a `ProblemType`,
    `tensor_name` is a symbolic name for an output Tensor possibly but not
    necessarily taken from `PredictionKey`, and `Tensor` is the
    corresponding output Tensor itself.
*  <b>`training_chief_hooks`</b>: A list of `SessionRunHook` objects that will be
    run on the chief worker during training.
*  <b>`training_hooks`</b>: A list of `SessionRunHook` objects that will be run on
    all workers during training.
*  <b>`scaffold`</b>: A `tf.train.Scaffold` object that can be used to set
    initialization, saver, and more to be used in training.

##### Returns:

  A validated `ModelFnOps` object.

##### Raises:


*  <b>`ValueError`</b>: If validation fails.

