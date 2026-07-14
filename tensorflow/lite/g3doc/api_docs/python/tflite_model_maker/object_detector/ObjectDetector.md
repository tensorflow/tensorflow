page_type: reference
description: ObjectDetector class for inference and exporting to tflite.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.object_detector.ObjectDetector" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create"/>
<meta itemprop="property" content="create_model"/>
<meta itemprop="property" content="create_serving_model"/>
<meta itemprop="property" content="evaluate"/>
<meta itemprop="property" content="evaluate_tflite"/>
<meta itemprop="property" content="export"/>
<meta itemprop="property" content="summary"/>
<meta itemprop="property" content="train"/>
<meta itemprop="property" content="ALLOWED_EXPORT_FORMAT"/>
<meta itemprop="property" content="DEFAULT_EXPORT_FORMAT"/>
</div>

# tflite_model_maker.object_detector.ObjectDetector

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/object_detector.py#L37-L264">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



ObjectDetector class for inference and exporting to tflite.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_model_maker.object_detector.ObjectDetector(
    model_spec: <a href="../../tflite_model_maker/object_detector/EfficientDetSpec"><code>tflite_model_maker.object_detector.EfficientDetSpec</code></a>,
    label_map: Dict[int, str],
    representative_data: Optional[<a href="../../tflite_model_maker/object_detector/DataLoader"><code>tflite_model_maker.object_detector.DataLoader</code></a>] = None
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`model_spec`<a id="model_spec"></a>
</td>
<td>
Specification for the model.
</td>
</tr><tr>
<td>
`label_map`<a id="label_map"></a>
</td>
<td>
 Dict, map label integer ids to string label names such as {1:
'person', 2: 'notperson'}. 0 is the reserved key for `background` and
  doesn't need to be included in `label_map`. Label names can't be
  duplicated.
</td>
</tr><tr>
<td>
`representative_data`<a id="representative_data"></a>
</td>
<td>
 Representative dataset for full integer
quantization. Used when converting the keras model to the TFLite model
with full interger quantization.
</td>
</tr>
</table>



## Methods

<h3 id="create"><code>create</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/object_detector.py#L219-L264">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create(
    train_data: <a href="../../tflite_model_maker/object_detector/DataLoader"><code>tflite_model_maker.object_detector.DataLoader</code></a>,
    model_spec: <a href="../../tflite_model_maker/object_detector/EfficientDetSpec"><code>tflite_model_maker.object_detector.EfficientDetSpec</code></a>,
    validation_data: Optional[<a href="../../tflite_model_maker/object_detector/DataLoader"><code>tflite_model_maker.object_detector.DataLoader</code></a>] = None,
    epochs: Optional[<a href="../../tflite_model_maker/object_detector/DataLoader"><code>tflite_model_maker.object_detector.DataLoader</code></a>] = None,
    batch_size: Optional[int] = None,
    train_whole_model: bool = False,
    do_train: bool = True
) -> T
</code></pre>

Loads data and train the model for object detection.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`train_data`
</td>
<td>
Training data.
</td>
</tr><tr>
<td>
`model_spec`
</td>
<td>
Specification for the model.
</td>
</tr><tr>
<td>
`validation_data`
</td>
<td>
Validation data. If None, skips validation process.
</td>
</tr><tr>
<td>
`epochs`
</td>
<td>
Number of epochs for training.
</td>
</tr><tr>
<td>
`batch_size`
</td>
<td>
Batch size for training.
</td>
</tr><tr>
<td>
`train_whole_model`
</td>
<td>
Boolean, False by default. If true, train the whole
model. Otherwise, only train the layers that are not match
`model_spec.config.var_freeze_expr`.
</td>
</tr><tr>
<td>
`do_train`
</td>
<td>
Whether to run training.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An instance based on ObjectDetector.
</td>
</tr>

</table>



<h3 id="create_model"><code>create_model</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/object_detector.py#L73-L75">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_model() -> tf.keras.Model
</code></pre>




<h3 id="create_serving_model"><code>create_serving_model</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/custom_model.py#L170-L176">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_serving_model()
</code></pre>

Returns the underlining Keras model for serving.


<h3 id="evaluate"><code>evaluate</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/object_detector.py#L127-L148">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>evaluate(
    data: <a href="../../tflite_model_maker/object_detector/DataLoader"><code>tflite_model_maker.object_detector.DataLoader</code></a>,
    batch_size: Optional[int] = None
) -> Dict[str, float]
</code></pre>

Evaluates the model.


<h3 id="evaluate_tflite"><code>evaluate_tflite</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/object_detector.py#L150-L156">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>evaluate_tflite(
    tflite_filepath: str,
    data: <a href="../../tflite_model_maker/object_detector/DataLoader"><code>tflite_model_maker.object_detector.DataLoader</code></a>
) -> Dict[str, float]
</code></pre>

Evaluate the TFLite model.


<h3 id="export"><code>export</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/custom_model.py#L95-L168">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>export(
    export_dir,
    tflite_filename=&#x27;model.tflite&#x27;,
    label_filename=&#x27;labels.txt&#x27;,
    vocab_filename=&#x27;vocab.txt&#x27;,
    saved_model_filename=&#x27;saved_model&#x27;,
    tfjs_folder_name=&#x27;tfjs&#x27;,
    export_format=None,
    **kwargs
)
</code></pre>

Converts the retrained model based on `export_format`.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`export_dir`
</td>
<td>
The directory to save exported files.
</td>
</tr><tr>
<td>
`tflite_filename`
</td>
<td>
File name to save tflite model. The full export path is
{export_dir}/{tflite_filename}.
</td>
</tr><tr>
<td>
`label_filename`
</td>
<td>
File name to save labels. The full export path is
{export_dir}/{label_filename}.
</td>
</tr><tr>
<td>
`vocab_filename`
</td>
<td>
File name to save vocabulary. The full export path is
{export_dir}/{vocab_filename}.
</td>
</tr><tr>
<td>
`saved_model_filename`
</td>
<td>
Path to SavedModel or H5 file to save the model. The
full export path is
{export_dir}/{saved_model_filename}/{saved_model.pb|assets|variables}.
</td>
</tr><tr>
<td>
`tfjs_folder_name`
</td>
<td>
Folder name to save tfjs model. The full export path is
{export_dir}/{tfjs_folder_name}.
</td>
</tr><tr>
<td>
`export_format`
</td>
<td>
List of export format that could be saved_model, tflite,
label, vocab.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Other parameters like `quantized_config` for TFLITE model.
</td>
</tr>
</table>



<h3 id="summary"><code>summary</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/custom_model.py#L65-L66">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>summary()
</code></pre>




<h3 id="train"><code>train</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/object_detector.py#L92-L125">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>train(
    train_data: <a href="../../tflite_model_maker/object_detector/DataLoader"><code>tflite_model_maker.object_detector.DataLoader</code></a>,
    validation_data: Optional[<a href="../../tflite_model_maker/object_detector/DataLoader"><code>tflite_model_maker.object_detector.DataLoader</code></a>] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None
) -> tf.keras.Model
</code></pre>

Feeds the training data for training.






<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
ALLOWED_EXPORT_FORMAT<a id="ALLOWED_EXPORT_FORMAT"></a>
</td>
<td>
`(<ExportFormat.TFLITE: 'TFLITE'>,
 <ExportFormat.SAVED_MODEL: 'SAVED_MODEL'>,
 <ExportFormat.LABEL: 'LABEL'>)`
</td>
</tr><tr>
<td>
DEFAULT_EXPORT_FORMAT<a id="DEFAULT_EXPORT_FORMAT"></a>
</td>
<td>
`<ExportFormat.TFLITE: 'TFLITE'>`
</td>
</tr>
</table>
