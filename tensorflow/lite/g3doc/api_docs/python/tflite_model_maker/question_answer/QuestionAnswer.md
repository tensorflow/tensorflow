page_type: reference
description: QuestionAnswer class for inference and exporting to tflite.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.question_answer.QuestionAnswer" />
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

# tflite_model_maker.question_answer.QuestionAnswer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/question_answer.py#L51-L232">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



QuestionAnswer class for inference and exporting to tflite.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_model_maker.question_answer.QuestionAnswer(
    model_spec, shuffle
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
`shuffle`<a id="shuffle"></a>
</td>
<td>
Whether the training data should be shuffled.
</td>
</tr>
</table>



## Methods

<h3 id="create"><code>create</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/question_answer.py#L193-L232">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create(
    train_data,
    model_spec,
    batch_size=None,
    epochs=2,
    steps_per_epoch=None,
    shuffle=False,
    do_train=True
)
</code></pre>

Loads data and train the model for question answer.


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
`batch_size`
</td>
<td>
Batch size for training.
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
`steps_per_epoch`
</td>
<td>
Integer or None. Total number of steps (batches of
samples) before declaring one epoch finished and starting the next
epoch. If `steps_per_epoch` is None, the epoch will run until the input
dataset is exhausted.
</td>
</tr><tr>
<td>
`shuffle`
</td>
<td>
Whether the data should be shuffled.
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
An instance based on QuestionAnswer.
</td>
</tr>

</table>



<h3 id="create_model"><code>create_model</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/question_answer.py#L84-L85">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_model()
</code></pre>




<h3 id="create_serving_model"><code>create_serving_model</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/custom_model.py#L170-L176">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_serving_model()
</code></pre>

Returns the underlining Keras model for serving.


<h3 id="evaluate"><code>evaluate</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/question_answer.py#L87-L118">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>evaluate(
    data,
    max_answer_length=30,
    null_score_diff_threshold=0.0,
    verbose_logging=False,
    output_dir=None
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
Data to be evaluated.
</td>
</tr><tr>
<td>
`max_answer_length`
</td>
<td>
The maximum length of an answer that can be generated.
This is needed because the start and end predictions are not conditioned
on one another.
</td>
</tr><tr>
<td>
`null_score_diff_threshold`
</td>
<td>
If null_score - best_non_null is greater than
the threshold, predict null. This is only used for SQuAD v2.
</td>
</tr><tr>
<td>
`verbose_logging`
</td>
<td>
If true, all of the warnings related to data processing
will be printed. A number of warnings are expected for a normal SQuAD
evaluation.
</td>
</tr><tr>
<td>
`output_dir`
</td>
<td>
The output directory to save output to json files:
predictions.json, nbest_predictions.json, null_odds.json. If None, skip
saving to json files.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A dict contains two metrics: Exact match rate and F1 score.
</td>
</tr>

</table>



<h3 id="evaluate_tflite"><code>evaluate_tflite</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/question_answer.py#L120-L151">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>evaluate_tflite(
    tflite_filepath,
    data,
    max_answer_length=30,
    null_score_diff_threshold=0.0,
    verbose_logging=False,
    output_dir=None
)
</code></pre>

Evaluate the model.


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
`max_answer_length`
</td>
<td>
The maximum length of an answer that can be generated.
This is needed because the start and end predictions are not conditioned
on one another.
</td>
</tr><tr>
<td>
`null_score_diff_threshold`
</td>
<td>
If null_score - best_non_null is greater than
the threshold, predict null. This is only used for SQuAD v2.
</td>
</tr><tr>
<td>
`verbose_logging`
</td>
<td>
If true, all of the warnings related to data processing
will be printed. A number of warnings are expected for a normal SQuAD
evaluation.
</td>
</tr><tr>
<td>
`output_dir`
</td>
<td>
The output directory to save output to json files:
predictions.json, nbest_predictions.json, null_odds.json. If None, skip
saving to json files.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A dict contains two metrics: Exact match rate and F1 score.
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

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/question_answer.py#L59-L82">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>train(
    train_data, epochs=None, batch_size=None, steps_per_epoch=None
)
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
 <ExportFormat.VOCAB: 'VOCAB'>,
 <ExportFormat.SAVED_MODEL: 'SAVED_MODEL'>)`
</td>
</tr><tr>
<td>
DEFAULT_EXPORT_FORMAT<a id="DEFAULT_EXPORT_FORMAT"></a>
</td>
<td>
`(<ExportFormat.TFLITE: 'TFLITE'>, <ExportFormat.VOCAB: 'VOCAB'>)`
</td>
</tr>
</table>
