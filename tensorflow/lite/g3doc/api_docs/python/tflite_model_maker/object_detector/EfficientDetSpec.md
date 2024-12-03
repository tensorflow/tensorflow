page_type: reference
description: A specification of the EfficientDet model.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.object_detector.EfficientDetSpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_model"/>
<meta itemprop="property" content="evaluate"/>
<meta itemprop="property" content="evaluate_tflite"/>
<meta itemprop="property" content="export_saved_model"/>
<meta itemprop="property" content="export_tflite"/>
<meta itemprop="property" content="get_default_quantization_config"/>
<meta itemprop="property" content="train"/>
<meta itemprop="property" content="compat_tf_versions"/>
</div>

# tflite_model_maker.object_detector.EfficientDetSpec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/object_detector_spec.py#L109-L513">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A specification of the EfficientDet model.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_model_maker.object_detector.EfficientDetSpec(
    model_name: str,
    uri: str,
    hparams: str = &#x27;&#x27;,
    model_dir: Optional[str] = None,
    epochs: int = 50,
    batch_size: int = 64,
    steps_per_execution: int = 1,
    moving_average_decay: int = 0,
    var_freeze_expr: str = &#x27;(efficientnet|fpn_cells|resample_p6)&#x27;,
    tflite_max_detections: int = 25,
    strategy: Optional[str] = None,
    tpu: Optional[str] = None,
    gcp_project: Optional[str] = None,
    tpu_zone: Optional[str] = None,
    use_xla: bool = False,
    profile: bool = False,
    debug: bool = False,
    tf_random_seed: int = 111111,
    verbose: int = 0
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`model_name`<a id="model_name"></a>
</td>
<td>
Model name.
</td>
</tr><tr>
<td>
`uri`<a id="uri"></a>
</td>
<td>
TF-Hub path/url to EfficientDet module.
</td>
</tr><tr>
<td>
`hparams`<a id="hparams"></a>
</td>
<td>
Hyperparameters used to overwrite default configuration. Can be

1) Dict, contains parameter names and values; 2) String, Comma separated
k=v pairs of hyperparameters; 3) String, yaml filename which's a module
containing attributes to use as hyperparameters.
</td>
</tr><tr>
<td>
`model_dir`<a id="model_dir"></a>
</td>
<td>
The location to save the model checkpoint files.
</td>
</tr><tr>
<td>
`epochs`<a id="epochs"></a>
</td>
<td>
Default training epochs.
</td>
</tr><tr>
<td>
`batch_size`<a id="batch_size"></a>
</td>
<td>
Training & Evaluation batch size.
</td>
</tr><tr>
<td>
`steps_per_execution`<a id="steps_per_execution"></a>
</td>
<td>
Number of steps per training execution.
</td>
</tr><tr>
<td>
`moving_average_decay`<a id="moving_average_decay"></a>
</td>
<td>
Float. The decay to use for maintaining moving
averages of the trained parameters.
</td>
</tr><tr>
<td>
`var_freeze_expr`<a id="var_freeze_expr"></a>
</td>
<td>
Expression to freeze variables.
</td>
</tr><tr>
<td>
`tflite_max_detections`<a id="tflite_max_detections"></a>
</td>
<td>
The max number of output detections in the TFLite
model.
</td>
</tr><tr>
<td>
`strategy`<a id="strategy"></a>
</td>
<td>
 A string specifying which distribution strategy to use.
Accepted values are 'tpu', 'gpus', None. tpu' means to use TPUStrategy.
'gpus' mean to use MirroredStrategy for multi-gpus. If None, use TF
default with OneDeviceStrategy.
</td>
</tr><tr>
<td>
`tpu`<a id="tpu"></a>
</td>
<td>
The Cloud TPU to use for training. This should be either the name
used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470
  url.
</td>
</tr><tr>
<td>
`gcp_project`<a id="gcp_project"></a>
</td>
<td>
Project name for the Cloud TPU-enabled project. If not
specified, we will attempt to automatically detect the GCE project from
metadata.
</td>
</tr><tr>
<td>
`tpu_zone`<a id="tpu_zone"></a>
</td>
<td>
GCE zone where the Cloud TPU is located in. If not specified, we
will attempt to automatically detect the GCE project from metadata.
</td>
</tr><tr>
<td>
`use_xla`<a id="use_xla"></a>
</td>
<td>
Use XLA even if strategy is not tpu. If strategy is tpu, always
use XLA, and this flag has no effect.
</td>
</tr><tr>
<td>
`profile`<a id="profile"></a>
</td>
<td>
Enable profile mode.
</td>
</tr><tr>
<td>
`debug`<a id="debug"></a>
</td>
<td>
Enable debug mode.
</td>
</tr><tr>
<td>
`tf_random_seed`<a id="tf_random_seed"></a>
</td>
<td>
Fixed random seed for deterministic execution across runs
for debugging.
</td>
</tr><tr>
<td>
`verbose`<a id="verbose"></a>
</td>
<td>
verbosity mode for <a href="https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint"><code>tf.keras.callbacks.ModelCheckpoint</code></a>, 0 or 1.
</td>
</tr>
</table>



## Methods

<h3 id="create_model"><code>create_model</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/object_detector_spec.py#L235-L238">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_model() -> tf.keras.Model
</code></pre>

Creates the EfficientDet model.


<h3 id="evaluate"><code>evaluate</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/object_detector_spec.py#L303-L346">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>evaluate(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    steps: int,
    json_file: Optional[str] = None
) -> Dict[str, float]
</code></pre>

Evaluate the EfficientDet keras model.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`model`
</td>
<td>
The keras model to be evaluated.
</td>
</tr><tr>
<td>
`dataset`
</td>
<td>
tf.data.Dataset used for evaluation.
</td>
</tr><tr>
<td>
`steps`
</td>
<td>
Number of steps to evaluate the model.
</td>
</tr><tr>
<td>
`json_file`
</td>
<td>
JSON with COCO data format containing golden bounding boxes.
Used for validation. If None, use the ground truth from the dataloader.
Refer to
<a href="https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5">https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5</a>
  for the description of COCO data format.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A dict contains AP metrics.
</td>
</tr>

</table>



<h3 id="evaluate_tflite"><code>evaluate_tflite</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/object_detector_spec.py#L348-L401">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>evaluate_tflite(
    tflite_filepath: str,
    dataset: tf.data.Dataset,
    steps: int,
    json_file: Optional[str] = None
) -> Dict[str, float]
</code></pre>

Evaluate the EfficientDet TFLite model.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`tflite_filepath`
</td>
<td>
File path to the TFLite model.
</td>
</tr><tr>
<td>
`dataset`
</td>
<td>
tf.data.Dataset used for evaluation.
</td>
</tr><tr>
<td>
`steps`
</td>
<td>
Number of steps to evaluate the model.
</td>
</tr><tr>
<td>
`json_file`
</td>
<td>
JSON with COCO data format containing golden bounding boxes.
Used for validation. If None, use the ground truth from the dataloader.
Refer to
<a href="https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5">https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5</a>
  for the description of COCO data format.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A dict contains AP metrics.
</td>
</tr>

</table>



<h3 id="export_saved_model"><code>export_saved_model</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/object_detector_spec.py#L403-L446">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>export_saved_model(
    model: tf.keras.Model,
    saved_model_dir: str,
    batch_size: Optional[int] = None,
    pre_mode: Optional[str] = &#x27;infer&#x27;,
    post_mode: Optional[str] = &#x27;global&#x27;
) -> None
</code></pre>

Saves the model to Tensorflow SavedModel.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`model`
</td>
<td>
The EfficientDetNet model used for training which doesn't have pre
and post processing.
</td>
</tr><tr>
<td>
`saved_model_dir`
</td>
<td>
Folder path for saved model.
</td>
</tr><tr>
<td>
`batch_size`
</td>
<td>
Batch size to be saved in saved_model.
</td>
</tr><tr>
<td>
`pre_mode`
</td>
<td>
Pre-processing Mode in ExportModel, must be {None, 'infer'}.
</td>
</tr><tr>
<td>
`post_mode`
</td>
<td>
Post-processing Mode in ExportModel, must be {None, 'global',
'per_class', 'tflite'}.
</td>
</tr>
</table>



<h3 id="export_tflite"><code>export_tflite</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/object_detector_spec.py#L466-L513">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>export_tflite(
    model: tf.keras.Model,
    tflite_filepath: str,
    quantization_config: Optional[<a href="../../tflite_model_maker/config/QuantizationConfig"><code>tflite_model_maker.config.QuantizationConfig</code></a>] = None
) -> None
</code></pre>

Converts the retrained model to tflite format and saves it.

The exported TFLite model has the following inputs & outputs:
One input:
  image: a float32 tensor of shape[1, height, width, 3] containing the
    normalized input image. `self.config.image_size` is [height, width].

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Four Outputs</th></tr>

<tr>
<td>
`detection_boxes`
</td>
<td>
a float32 tensor of shape [1, num_boxes, 4] with box
locations.
</td>
</tr><tr>
<td>
`detection_classes`
</td>
<td>
a float32 tensor of shape [1, num_boxes] with class
indices.
</td>
</tr><tr>
<td>
`detection_scores`
</td>
<td>
a float32 tensor of shape [1, num_boxes] with class
scores.
</td>
</tr><tr>
<td>
`num_boxes`
</td>
<td>
a float32 tensor of size 1 containing the number of detected
boxes.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`model`
</td>
<td>
The EfficientDetNet model used for training which doesn't have pre
and post processing.
</td>
</tr><tr>
<td>
`tflite_filepath`
</td>
<td>
File path to save tflite model.
</td>
</tr><tr>
<td>
`quantization_config`
</td>
<td>
Configuration for post-training quantization.
</td>
</tr>
</table>



<h3 id="get_default_quantization_config"><code>get_default_quantization_config</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/object_detector_spec.py#L448-L464">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_default_quantization_config(
    representative_data: <a href="../../tflite_model_maker/object_detector/DataLoader"><code>tflite_model_maker.object_detector.DataLoader</code></a>
) -> <a href="../../tflite_model_maker/config/QuantizationConfig"><code>tflite_model_maker.config.QuantizationConfig</code></a>
</code></pre>

Gets the default quantization configuration.


<h3 id="train"><code>train</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/object_detector_spec.py#L240-L272">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>train(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    steps_per_epoch: int,
    val_dataset: Optional[tf.data.Dataset],
    validation_steps: int,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    val_json_file: Optional[str] = None
) -> tf.keras.Model
</code></pre>

Run EfficientDet training.






<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
compat_tf_versions<a id="compat_tf_versions"></a>
</td>
<td>
`[2]`
</td>
</tr>
</table>
