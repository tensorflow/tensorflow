page_type: reference
description: Audio classifier for training/inference and exporing.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.audio_classifier.AudioClassifier" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="confusion_matrix"/>
<meta itemprop="property" content="create"/>
<meta itemprop="property" content="create_model"/>
<meta itemprop="property" content="create_serving_model"/>
<meta itemprop="property" content="evaluate"/>
<meta itemprop="property" content="evaluate_tflite"/>
<meta itemprop="property" content="export"/>
<meta itemprop="property" content="predict_top_k"/>
<meta itemprop="property" content="summary"/>
<meta itemprop="property" content="train"/>
<meta itemprop="property" content="ALLOWED_EXPORT_FORMAT"/>
<meta itemprop="property" content="DEFAULT_EXPORT_FORMAT"/>
</div>

# tflite_model_maker.audio_classifier.AudioClassifier

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/audio_classifier.py#L27-L140">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Audio classifier for training/inference and exporing.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_model_maker.audio_classifier.AudioClassifier(
    model_spec, index_to_label, shuffle, train_whole_model
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
Specification for the model.
</td>
</tr><tr>
<td>
`index_to_label`<a id="index_to_label"></a>
</td>
<td>
A list that map from index to label class name.
</td>
</tr><tr>
<td>
`shuffle`<a id="shuffle"></a>
</td>
<td>
Whether the data should be shuffled.
</td>
</tr><tr>
<td>
`train_whole_model`<a id="train_whole_model"></a>
</td>
<td>
If true, the Hub module is trained together with the
classification layer on top. Otherwise, only train the top
classification layer.
</td>
</tr>
</table>



## Methods

<h3 id="confusion_matrix"><code>confusion_matrix</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/audio_classifier.py#L86-L101">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>confusion_matrix(
    data, batch_size=32
)
</code></pre>




<h3 id="create"><code>create</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/audio_classifier.py#L103-L140">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create(
    train_data,
    model_spec,
    validation_data=None,
    batch_size=32,
    epochs=5,
    model_dir=None,
    do_train=True,
    train_whole_model=False
)
</code></pre>

Loads data and retrains the model.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`train_data`
</td>
<td>
A instance of audio_dataloader.DataLoader class.
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
Validation DataLoader. If None, skips validation process.
</td>
</tr><tr>
<td>
`batch_size`
</td>
<td>
Number of samples per training step. If `use_hub_library` is
False, it represents the base learning rate when train batch size is 256
and it's linear to the batch size.
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
`model_dir`
</td>
<td>
The location of the model checkpoint files.
</td>
</tr><tr>
<td>
`do_train`
</td>
<td>
Whether to run training.
</td>
</tr><tr>
<td>
`train_whole_model`
</td>
<td>
Boolean. By default, only the classification head is
trained. When True, the base model is also trained.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An instance based on AudioClassifier.
</td>
</tr>

</table>



<h3 id="create_model"><code>create_model</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/audio_classifier.py#L59-L62">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_model(
    num_classes, train_whole_model
)
</code></pre>




<h3 id="create_serving_model"><code>create_serving_model</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/custom_model.py#L170-L176">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_serving_model()
</code></pre>

Returns the underlining Keras model for serving.


<h3 id="evaluate"><code>evaluate</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/classification_model.py#L53-L65">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>evaluate(
    data, batch_size=32
)
</code></pre>

Evaluates the model.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`data`
</td>
<td>
Data to be evaluated.
</td>
</tr><tr>
<td>
`batch_size`
</td>
<td>
Number of samples per evaluation step.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The loss value and accuracy.
</td>
</tr>

</table>



<h3 id="evaluate_tflite"><code>evaluate_tflite</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/classification_model.py#L105-L143">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>evaluate_tflite(
    tflite_filepath, data, postprocess_fn=None
)
</code></pre>

Evaluates the tflite model.


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
</tr><tr>
<td>
`postprocess_fn`
</td>
<td>
Postprocessing function that will be applied to the output
of `lite_runner.run` before calculating the probabilities.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The evaluation result of TFLite model - accuracy.
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



<h3 id="predict_top_k"><code>predict_top_k</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/classification_model.py#L67-L95">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>predict_top_k(
    data, k=1, batch_size=32
)
</code></pre>

Predicts the top-k predictions.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`data`
</td>
<td>
Data to be evaluated. Either an instance of DataLoader or just raw
data entries such TF tensor or numpy array.
</td>
</tr><tr>
<td>
`k`
</td>
<td>
Number of top results to be predicted.
</td>
</tr><tr>
<td>
`batch_size`
</td>
<td>
Number of samples per evaluation step.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
top k results. Each one is (label, probability).
</td>
</tr>

</table>



<h3 id="summary"><code>summary</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/custom_model.py#L65-L66">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>summary()
</code></pre>




<h3 id="train"><code>train</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/audio_classifier.py#L36-L57">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>train(
    train_data, validation_data, epochs, batch_size
)
</code></pre>








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
`<ExportFormat.TFLITE: 'TFLITE'>`
</td>
</tr>
</table>
