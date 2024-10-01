page_type: reference
description: A specification of BERT model for text classification.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.text_classifier.BertClassifierSpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="build"/>
<meta itemprop="property" content="convert_examples_to_features"/>
<meta itemprop="property" content="create_model"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="get_default_quantization_config"/>
<meta itemprop="property" content="get_name_to_features"/>
<meta itemprop="property" content="reorder_input_details"/>
<meta itemprop="property" content="run_classifier"/>
<meta itemprop="property" content="save_vocab"/>
<meta itemprop="property" content="select_data_from_record"/>
<meta itemprop="property" content="compat_tf_versions"/>
<meta itemprop="property" content="convert_from_saved_model_tf2"/>
<meta itemprop="property" content="need_gen_vocab"/>
</div>

# tflite_model_maker.text_classifier.BertClassifierSpec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L455-L626">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A specification of BERT model for text classification.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_model_maker.text_classifier.BertClassifierSpec(
    uri=&#x27;https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1&#x27;,
    model_dir=None,
    seq_len=128,
    dropout_rate=0.1,
    initializer_range=0.02,
    learning_rate=3e-05,
    distribution_strategy=&#x27;mirrored&#x27;,
    num_gpus=-1,
    tpu=&#x27;&#x27;,
    trainable=True,
    do_lower_case=True,
    is_tf2=True,
    name=&#x27;Bert&#x27;,
    tflite_input_name=None,
    default_batch_size=32,
    index_to_label=None
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
`name`<a id="name"></a>
</td>
<td>
The name of the object.
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
`default_batch_size`<a id="default_batch_size"></a>
</td>
<td>
Default batch size for training.
</td>
</tr><tr>
<td>
`index_to_label`<a id="index_to_label"></a>
</td>
<td>
List of labels in the training data. e.g. ['neg', 'pos'].
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

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L544-L549">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>convert_examples_to_features(
    examples, tfrecord_file, label_names
)
</code></pre>

Converts examples to features and write them into TFRecord file.


<h3 id="create_model"><code>create_model</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L551-L577">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_model(
    num_classes, optimizer=&#x27;adam&#x27;, with_loss_and_metrics=True
)
</code></pre>

Creates the keras model.


<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L623-L626">View source</a>

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

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L523-L532">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_name_to_features()
</code></pre>

Gets the dictionary describing the features.


<h3 id="reorder_input_details"><code>reorder_input_details</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L426-L436">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reorder_input_details(
    tflite_input_details
)
</code></pre>

Reorders the tflite input details to map the order of keras model.


<h3 id="run_classifier"><code>run_classifier</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L579-L621">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>run_classifier(
    train_ds, validation_ds, epochs, steps_per_epoch, num_classes, **kwargs
)
</code></pre>

Creates classifier and runs the classifier training.


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
`validation_ds`
</td>
<td>
tf.data.Dataset, validation data to be fed in
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
`num_classes`
</td>
<td>
Interger, number of classes.
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



<h3 id="save_vocab"><code>save_vocab</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L447-L452">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>save_vocab(
    vocab_filename
)
</code></pre>

Prints the file path to the vocabulary.


<h3 id="select_data_from_record"><code>select_data_from_record</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L534-L542">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>select_data_from_record(
    record
)
</code></pre>

Dispatches records to features and labels.






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
