page_type: reference
description: A specification of averaging word vector model.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.text_classifier.AverageWordVecSpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="convert_examples_to_features"/>
<meta itemprop="property" content="create_model"/>
<meta itemprop="property" content="gen_vocab"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="get_default_quantization_config"/>
<meta itemprop="property" content="get_name_to_features"/>
<meta itemprop="property" content="load_vocab"/>
<meta itemprop="property" content="preprocess"/>
<meta itemprop="property" content="run_classifier"/>
<meta itemprop="property" content="save_vocab"/>
<meta itemprop="property" content="select_data_from_record"/>
<meta itemprop="property" content="PAD"/>
<meta itemprop="property" content="START"/>
<meta itemprop="property" content="UNKNOWN"/>
<meta itemprop="property" content="compat_tf_versions"/>
<meta itemprop="property" content="convert_from_saved_model_tf2"/>
<meta itemprop="property" content="need_gen_vocab"/>
</div>

# tflite_model_maker.text_classifier.AverageWordVecSpec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L55-L255">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A specification of averaging word vector model.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_model_maker.text_classifier.AverageWordVecSpec(
    num_words=10000,
    seq_len=256,
    wordvec_dim=16,
    lowercase=True,
    dropout_rate=0.2,
    name=&#x27;AverageWordVec&#x27;,
    default_training_epochs=2,
    default_batch_size=32,
    model_dir=None,
    index_to_label=None
)
</code></pre>




<h3>Used in the notebooks</h3>
<table class="vertical-rules">
  <thead>
    <tr>
      <th>Used in the tutorials</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
  <ul>
    <li><a href="https://www.tensorflow.org/lite/models/modify/model_maker/text_classification">Text classification with TensorFlow Lite Model Maker</a></li>
  </ul>
</td>
    </tr>
  </tbody>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`num_words`<a id="num_words"></a>
</td>
<td>
Number of words to generate the vocabulary from data.
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
`wordvec_dim`<a id="wordvec_dim"></a>
</td>
<td>
Dimension of the word embedding.
</td>
</tr><tr>
<td>
`lowercase`<a id="lowercase"></a>
</td>
<td>
Whether to convert all uppercase character to lowercase during
preprocessing.
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
`name`<a id="name"></a>
</td>
<td>
Name of the object.
</td>
</tr><tr>
<td>
`default_training_epochs`<a id="default_training_epochs"></a>
</td>
<td>
Default training epochs for training.
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
`model_dir`<a id="model_dir"></a>
</td>
<td>
The location of the model checkpoint files.
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

<h3 id="convert_examples_to_features"><code>convert_examples_to_features</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L120-L135">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>convert_examples_to_features(
    examples, tfrecord_file, label_names
)
</code></pre>

Converts examples to features and write them into TFRecord file.


<h3 id="create_model"><code>create_model</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L137-L159">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_model(
    num_classes, optimizer=&#x27;rmsprop&#x27;, with_loss_and_metrics=True
)
</code></pre>

Creates the keras model.


<h3 id="gen_vocab"><code>gen_vocab</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L181-L195">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gen_vocab(
    examples
)
</code></pre>

Generates vocabulary list in `examples` with maximum `num_words` words.


<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L244-L251">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config()
</code></pre>

Gets the configuration.


<h3 id="get_default_quantization_config"><code>get_default_quantization_config</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L253-L255">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_default_quantization_config()
</code></pre>

Gets the default quantization configuration.


<h3 id="get_name_to_features"><code>get_name_to_features</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L106-L112">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_name_to_features()
</code></pre>

Gets the dictionary describing the features.


<h3 id="load_vocab"><code>load_vocab</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L234-L242">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>load_vocab(
    vocab_filename
)
</code></pre>

Loads vocabulary from `vocab_filename`.


<h3 id="preprocess"><code>preprocess</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L197-L216">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>preprocess(
    raw_text
)
</code></pre>

Preprocess the text for text classification.


<h3 id="run_classifier"><code>run_classifier</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L161-L179">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>run_classifier(
    train_ds, validation_ds, epochs, steps_per_epoch, num_classes, **kwargs
)
</code></pre>

Creates classifier and runs the classifier training.


<h3 id="save_vocab"><code>save_vocab</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L226-L232">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>save_vocab(
    vocab_filename
)
</code></pre>

Saves the vocabulary in `vocab_filename`.


<h3 id="select_data_from_record"><code>select_data_from_record</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/text_spec.py#L114-L118">View source</a>

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
PAD<a id="PAD"></a>
</td>
<td>
`'<PAD>'`
</td>
</tr><tr>
<td>
START<a id="START"></a>
</td>
<td>
`'<START>'`
</td>
</tr><tr>
<td>
UNKNOWN<a id="UNKNOWN"></a>
</td>
<td>
`'<UNKNOWN>'`
</td>
</tr><tr>
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
`False`
</td>
</tr><tr>
<td>
need_gen_vocab<a id="need_gen_vocab"></a>
</td>
<td>
`True`
</td>
</tr>
</table>
