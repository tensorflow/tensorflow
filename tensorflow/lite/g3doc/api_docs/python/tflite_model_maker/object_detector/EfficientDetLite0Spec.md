page_type: reference
description: Creates EfficientDet-Lite0 model spec. See also: <a href="../../tflite_model_maker/object_detector/EfficientDetSpec"><code>tflite_model_maker.object_detector.EfficientDetSpec</code></a>.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.object_detector.EfficientDetLite0Spec" />
<meta itemprop="path" content="Stable" />
</div>

# tflite_model_maker.object_detector.EfficientDetLite0Spec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Creates EfficientDet-Lite0 model spec. See also: <a href="../../tflite_model_maker/object_detector/EfficientDetSpec"><code>tflite_model_maker.object_detector.EfficientDetSpec</code></a>.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_model_maker.object_detector.EfficientDetLite0Spec(
    *,
    model_name=&#x27;efficientdet-lite0&#x27;,
    uri=&#x27;https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1&#x27;,
    hparams=&#x27;&#x27;,
    model_dir=None,
    epochs=50,
    batch_size=64,
    steps_per_execution=1,
    moving_average_decay=0,
    var_freeze_expr=&#x27;(efficientnet|fpn_cells|resample_p6)&#x27;,
    tflite_max_detections=25,
    strategy=None,
    tpu=None,
    gcp_project=None,
    tpu_zone=None,
    use_xla=False,
    profile=False,
    debug=False,
    tf_random_seed=111111,
    verbose=0
)
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
