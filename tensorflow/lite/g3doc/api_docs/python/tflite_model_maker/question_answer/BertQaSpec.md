page_type: reference
description: A specification of BERT model for question answering.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.question_answer.BertQaSpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="build"/>
<meta itemprop="property" content="convert_examples_to_features"/>
<meta itemprop="property" content="create_model"/>
<meta itemprop="property" content="evaluate"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="get_default_quantization_config"/>
<meta itemprop="property" content="get_name_to_features"/>
<meta itemprop="property" content="predict"/>
<meta itemprop="property" content="predict_tflite"/>
<meta itemprop="property" content="reorder_input_details"/>
<meta itemprop="property" content="reorder_output_details"/>
<meta itemprop="property" content="save_vocab"/>
<meta itemprop="property" content="select_data_from_record"/>
<meta itemprop="property" content="train"/>
<meta itemprop="property" content="compat_tf_versions"/>
<meta itemprop="property" content="convert_from_saved_model_tf2"/>
<meta itemprop="property" content="need_gen_vocab"/>
</div>

# tflite_model_maker.question_answer.BertQaSpec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L754-L1083">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A specification of BERT model for question answering.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_model_maker.question_answer.BertQaSpec(
    uri=&#x27;https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1&#x27;,
    model_dir=None,
    seq_len=384,
    query_len=64,
    doc_stride=128,
    dropout_rate=0.1,
    initializer_range=0.02,
    learning_rate=8e-05,
    distribution_strategy=&#x27;mirrored&#x27;,
    num_gpus=-1,
    tpu=&#x27;&#x27;,
    trainable=True,
    predict_batch_size=8,
    do_lower_case=True,
    is_tf2=True,
    tflite_input_name=None,
    tflite_output_name=None,
    init_from_squad_model=False,
    default_batch_size=16,
    name=&#x27;Bert&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`uri`<a id="uri"></a>
</td>
<td>
TF-Hub path/url to Bert module.
</td>
</tr><tr>
<td>
`model_dir`<a id="model_dir"></a>
</td>
<td>
The location of the model checkpoint files.
</td>
</tr><tr>
<td>
`seq_len`<a id="seq_len"></a>
</td>
<td>
Length of the sequence to feed into the model.
</td>
</tr><tr>
<td>
`query_len`<a id="query_len"></a>
</td>
<td>
Length of the query to feed into the model.
</td>
</tr><tr>
<td>
`doc_stride`<a id="doc_stride"></a>
</td>
<td>
The stride when we do a sliding window approach to take chunks
of the documents.
</td>
</tr><tr>
<td>
`dropout_rate`<a id="dropout_rate"></a>
</td>
<td>
The rate for dropout.
</td>
</tr><tr>
<td>
`initializer_range`<a id="initializer_range"></a>
</td>
<td>
The stdev of the truncated_normal_initializer for
initializing all weight matrices.
</td>
</tr><tr>
<td>
`learning_rate`<a id="learning_rate"></a>
</td>
<td>
The initial learning rate for Adam.
</td>
</tr><tr>
<td>
`distribution_strategy`<a id="distribution_strategy"></a>
</td>
<td>
 A string specifying which distribution strategy to
use. Accepted values are 'off', 'one_device', 'mirrored',
'parameter_server', 'multi_worker_mirrored', and 'tpu' -- case
insensitive. 'off' means not to use Distribution Strategy; 'tpu' means
to use TPUStrategy using `tpu_address`.
</td>
</tr><tr>
<td>
`num_gpus`<a id="num_gpus"></a>
</td>
<td>
How many GPUs to use at each worker with the
DistributionStrategies API. The default is -1, which means utilize all
available GPUs.
</td>
</tr><tr>
<td>
`tpu`<a id="tpu"></a>
</td>
<td>
TPU address to connect to.
</td>
</tr><tr>
<td>
`trainable`<a id="trainable"></a>
</td>
<td>
boolean, whether pretrain layer is trainable.
</td>
</tr><tr>
<td>
`predict_batch_size`<a id="predict_batch_size"></a>
</td>
<td>
Batch size for prediction.
</td>
</tr><tr>
<td>
`do_lower_case`<a id="do_lower_case"></a>
</td>
<td>
boolean, whether to lower case the input text. Should be
True for uncased models and False for cased models.
</td>
</tr><tr>
<td>
`is_tf2`<a id="is_tf2"></a>
</td>
<td>
boolean, whether the hub module is in TensorFlow 2.x format.
</td>
</tr><tr>
<td>
`tflite_input_name`<a id="tflite_input_name"></a>
</td>
<td>
Dict, input names for the TFLite model.
</td>
</tr><tr>
<td>
`tflite_output_name`<a id="tflite_output_name"></a>
</td>
<td>
Dict, output names for the TFLite model.
</td>
</tr><tr>
<td>
`init_from_squad_model`<a id="init_from_squad_model"></a>
</td>
<td>
boolean, whether to initialize from the model that
is already retrained on Squad 1.1.
</td>
</tr><tr>
<td>
`default_batch_size`<a id="default_batch_size"></a>
</td>
<td>
Default batch size for training.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
Name of the object.
</td>
</tr>
</table>



## Methods

<h3 id="build"><code>build</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L438-L445">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build()
</code></pre>

Builds the class. Used for lazy initialization.


<h3 id="convert_examples_to_features"><code>convert_examples_to_features</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L871-L885">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>convert_examples_to_features(
    examples, is_training, output_fn, batch_size
)
</code></pre>

Converts examples to features and write them into TFRecord file.


<h3 id="create_model"><code>create_model</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L887-L899">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_model()
</code></pre>

Creates the model for qa task.


<h3 id="evaluate"><code>evaluate</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L1016-L1083">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>evaluate(
    model,
    tflite_filepath,
    dataset,
    num_steps,
    eval_examples,
    eval_features,
    predict_file,
    version_2_with_negative,
    max_answer_length,
    null_score_diff_threshold,
    verbose_logging,
    output_dir
)
</code></pre>

Evaluate QA model.


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
`num_steps`
</td>
<td>
Number of steps to evaluate the model.
</td>
</tr><tr>
<td>
`eval_examples`
</td>
<td>
List of `squad_lib.SquadExample` for evaluation data.
</td>
</tr><tr>
<td>
`eval_features`
</td>
<td>
List of `squad_lib.InputFeatures` for evaluation data.
</td>
</tr><tr>
<td>
`predict_file`
</td>
<td>
The input predict file.
</td>
</tr><tr>
<td>
`version_2_with_negative`
</td>
<td>
Whether the input predict file is SQuAD 2.0
format.
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



<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L861-L869">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config()
</code></pre>

Gets the configuration.


<h3 id="get_default_quantization_config"><code>get_default_quantization_config</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L420-L424">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_default_quantization_config()
</code></pre>

Gets the default quantization configuration.


<h3 id="get_name_to_features"><code>get_name_to_features</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L831-L845">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_name_to_features(
    is_training
)
</code></pre>

Gets the dictionary describing the features.


<h3 id="predict"><code>predict</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L982-L984">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>predict(
    model, dataset, num_steps
)
</code></pre>

Predicts the dataset for `model`.


<h3 id="predict_tflite"><code>predict_tflite</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L996-L1014">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>predict_tflite(
    tflite_filepath, dataset
)
</code></pre>

Predicts the dataset for TFLite model in `tflite_filepath`.


<h3 id="reorder_input_details"><code>reorder_input_details</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L426-L436">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reorder_input_details(
    tflite_input_details
)
</code></pre>

Reorders the tflite input details to map the order of keras model.


<h3 id="reorder_output_details"><code>reorder_output_details</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L986-L994">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reorder_output_details(
    tflite_output_details
)
</code></pre>

Reorders the tflite output details to map the order of keras model.


<h3 id="save_vocab"><code>save_vocab</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L447-L452">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>save_vocab(
    vocab_filename
)
</code></pre>

Prints the file path to the vocabulary.


<h3 id="select_data_from_record"><code>select_data_from_record</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L847-L859">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>select_data_from_record(
    record
)
</code></pre>

Dispatches records to features and labels.


<h3 id="train"><code>train</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L901-L946">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>train(
    train_ds, epochs, steps_per_epoch, **kwargs
)
</code></pre>

Run bert QA training.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`train_ds`
</td>
<td>
tf.data.Dataset, training data to be fed in
tf.keras.Model.fit().
</td>
</tr><tr>
<td>
`epochs`
</td>
<td>
Integer, training epochs.
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
`**kwargs`
</td>
<td>
Other parameters used in the tf.keras.Model.fit().
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
tf.keras.Model, the keras model that's already trained.
</td>
</tr>

</table>







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
</tr><tr>
<td>
convert_from_saved_model_tf2<a id="convert_from_saved_model_tf2"></a>
</td>
<td>
`True`
</td>
</tr><tr>
<td>
need_gen_vocab<a id="need_gen_vocab"></a>
</td>
<td>
`False`
</td>
</tr>
</table>
