page_type: reference
description: Recommendation task class.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.recommendation.Recommendation" />
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
<meta itemprop="property" content="OOV_ID"/>
</div>

# tflite_model_maker.recommendation.Recommendation

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/recommendation.py#L34-L265">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Recommendation task class.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_model_maker.recommendation.Recommendation(
    model_spec,
    model_dir,
    shuffle=True,
    learning_rate=0.1,
    gradient_clip_norm=1.0
)
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
recommendation model spec.
</td>
</tr><tr>
<td>
`model_dir`<a id="model_dir"></a>
</td>
<td>
str, path to export model checkpoints and summaries.
</td>
</tr><tr>
<td>
`shuffle`<a id="shuffle"></a>
</td>
<td>
boolean, whether the training data should be shuffled.
</td>
</tr><tr>
<td>
`learning_rate`<a id="learning_rate"></a>
</td>
<td>
float, learning rate.
</td>
</tr><tr>
<td>
`gradient_clip_norm`<a id="gradient_clip_norm"></a>
</td>
<td>
float, clip threshold (<= 0 meaning no clip).
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`input_spec`<a id="input_spec"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`model_hparams`<a id="model_hparams"></a>
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="create"><code>create</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/recommendation.py#L213-L265">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create(
    train_data,
    model_spec: <a href="../../tflite_model_maker/recommendation/ModelSpec"><code>tflite_model_maker.recommendation.ModelSpec</code></a>,
    model_dir: str = None,
    validation_data=None,
    batch_size: int = 16,
    steps_per_epoch: int = 10000,
    epochs: int = 1,
    learning_rate: float = 0.1,
    gradient_clip_norm: float = 1.0,
    shuffle: bool = True,
    do_train: bool = True
)
</code></pre>

Loads data and train the model for recommendation.


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
ModelSpec, Specification for the model.
</td>
</tr><tr>
<td>
`model_dir`
</td>
<td>
str, path to export model checkpoints and summaries.
</td>
</tr><tr>
<td>
`validation_data`
</td>
<td>
Validation data.
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
`steps_per_epoch`
</td>
<td>
int, Number of step per epoch.
</td>
</tr><tr>
<td>
`epochs`
</td>
<td>
int, Number of epochs for training.
</td>
</tr><tr>
<td>
`learning_rate`
</td>
<td>
float, learning rate.
</td>
</tr><tr>
<td>
`gradient_clip_norm`
</td>
<td>
float, clip threshold (<= 0 meaning no clip).
</td>
</tr><tr>
<td>
`shuffle`
</td>
<td>
boolean, whether the training data should be shuffled.
</td>
</tr><tr>
<td>
`do_train`
</td>
<td>
boolean, whether to run training.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An instance based on Recommendation.
</td>
</tr>

</table>



<h3 id="create_model"><code>create_model</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/recommendation.py#L76-L88">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_model(
    do_train=True
)
</code></pre>

Creates a model.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`do_train`
</td>
<td>
boolean. Whether to train the model.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Keras model.
</td>
</tr>

</table>



<h3 id="create_serving_model"><code>create_serving_model</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/custom_model.py#L170-L176">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_serving_model()
</code></pre>

Returns the underlining Keras model for serving.


<h3 id="evaluate"><code>evaluate</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/recommendation.py#L127-L141">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>evaluate(
    data, batch_size=10
)
</code></pre>

Evaluate the model.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`data`
</td>
<td>
Evaluation data.
</td>
</tr><tr>
<td>
`batch_size`
</td>
<td>
int, batch size for evaluation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
History from model.evaluate().
</td>
</tr>

</table>



<h3 id="evaluate_tflite"><code>evaluate_tflite</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/recommendation.py#L173-L211">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>evaluate_tflite(
    tflite_filepath, data
)
</code></pre>

Evaluates the tflite model.

The data is padded to required length, and multiple metrics are evaluated.

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
`data`
</td>
<td>
Data to be evaluated.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Dict of (metric, value), evaluation result of TFLite model.
</td>
</tr>

</table>



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

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/recommendation.py#L90-L125">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>train(
    train_data,
    validation_data=None,
    batch_size=16,
    steps_per_epoch=100,
    epochs=1
)
</code></pre>

Feeds the training data for training.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`train_data`
</td>
<td>
Training dataset.
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
`batch_size`
</td>
<td>
int, the batch size.
</td>
</tr><tr>
<td>
`steps_per_epoch`
</td>
<td>
int, the step of each epoch.
</td>
</tr><tr>
<td>
`epochs`
</td>
<td>
int, number of epochs.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
History from model.fit().
</td>
</tr>

</table>







<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
ALLOWED_EXPORT_FORMAT<a id="ALLOWED_EXPORT_FORMAT"></a>
</td>
<td>
`(<ExportFormat.LABEL: 'LABEL'>,
 <ExportFormat.TFLITE: 'TFLITE'>,
 <ExportFormat.SAVED_MODEL: 'SAVED_MODEL'>)`
</td>
</tr><tr>
<td>
DEFAULT_EXPORT_FORMAT<a id="DEFAULT_EXPORT_FORMAT"></a>
</td>
<td>
`(<ExportFormat.TFLITE: 'TFLITE'>,)`
</td>
</tr><tr>
<td>
OOV_ID<a id="OOV_ID"></a>
</td>
<td>
`0`
</td>
</tr>
</table>
