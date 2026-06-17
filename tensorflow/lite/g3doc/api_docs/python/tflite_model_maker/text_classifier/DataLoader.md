page_type: reference
description: DataLoader for text classifier.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.text_classifier.DataLoader" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="from_csv"/>
<meta itemprop="property" content="from_folder"/>
<meta itemprop="property" content="gen_dataset"/>
<meta itemprop="property" content="split"/>
</div>

# tflite_model_maker.text_classifier.DataLoader

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/text_dataloader.py#L83-L289">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



DataLoader for text classifier.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_model_maker.text_classifier.DataLoader(
    dataset, size, index_to_label
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
`dataset`<a id="dataset"></a>
</td>
<td>
A tf.data.Dataset object that contains a potentially large set of
elements, where each element is a pair of (input_data, target). The
`input_data` means the raw input data, like an image, a text etc., while
the `target` means some ground truth of the raw input data, such as the
classification label of the image etc.
</td>
</tr><tr>
<td>
`size`<a id="size"></a>
</td>
<td>
The size of the dataset. tf.data.Dataset donesn't support a function
to get the length directly since it's lazy-loaded and may be infinite.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`num_classes`<a id="num_classes"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`size`<a id="size"></a>
</td>
<td>
Returns the size of the dataset.

Note that this function may return None becuase the exact size of the
dataset isn't a necessary parameter to create an instance of this class,
and tf.data.Dataset donesn't support a function to get the length directly
since it's lazy-loaded and may be infinite.
In most cases, however, when an instance of this class is created by helper
functions like 'from_folder', the size of the dataset will be preprocessed,
and this function can return an int representing the size of the dataset.
</td>
</tr>
</table>



## Methods

<h3 id="from_csv"><code>from_csv</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/text_dataloader.py#L165-L236">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_csv(
    filename,
    text_column,
    label_column,
    fieldnames=None,
    model_spec=&#x27;average_word_vec&#x27;,
    is_training=True,
    delimiter=&#x27;,&#x27;,
    quotechar=&#x27;\&#x27;&quot;,
    shuffle=False,
    cache_dir=None
)
</code></pre>

Loads text with labels from the csv file and preproecess text according to `model_spec`.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`filename`
</td>
<td>
Name of the file.
</td>
</tr><tr>
<td>
`text_column`
</td>
<td>
String, Column name for input text.
</td>
</tr><tr>
<td>
`label_column`
</td>
<td>
String, Column name for labels.
</td>
</tr><tr>
<td>
`fieldnames`
</td>
<td>
A sequence, used in csv.DictReader. If fieldnames is omitted,
the values in the first row of file f will be used as the fieldnames.
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
`is_training`
</td>
<td>
Whether the loaded data is for training or not.
</td>
</tr><tr>
<td>
`delimiter`
</td>
<td>
Character used to separate fields.
</td>
</tr><tr>
<td>
`quotechar`
</td>
<td>
Character used to quote fields containing special characters.
</td>
</tr><tr>
<td>
`shuffle`
</td>
<td>
boolean, if shuffle, random shuffle data.
</td>
</tr><tr>
<td>
`cache_dir`
</td>
<td>
The cache directory to save preprocessed data. If None,
generates a temporary directory to cache preprocessed data.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
TextDataset containing text, labels and other related info.
</td>
</tr>

</table>



<h3 id="from_folder"><code>from_folder</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/text_dataloader.py#L87-L163">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_folder(
    filename,
    model_spec=&#x27;average_word_vec&#x27;,
    is_training=True,
    class_labels=None,
    shuffle=True,
    cache_dir=None
)
</code></pre>

Loads text with labels and preproecess text according to `model_spec`.

Assume the text data of the same label are in the same subdirectory. each
file is one text.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`filename`
</td>
<td>
Name of the file.
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
`is_training`
</td>
<td>
Whether the loaded data is for training or not.
</td>
</tr><tr>
<td>
`class_labels`
</td>
<td>
Class labels that should be considered. Name of the
subdirectory not in `class_labels` will be ignored. If None, all the
subdirectories will be considered.
</td>
</tr><tr>
<td>
`shuffle`
</td>
<td>
boolean, if shuffle, random shuffle data.
</td>
</tr><tr>
<td>
`cache_dir`
</td>
<td>
The cache directory to save preprocessed data. If None,
generates a temporary directory to cache preprocessed data.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
TextDataset containing text, labels and other related info.
</td>
</tr>

</table>



<h3 id="gen_dataset"><code>gen_dataset</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/dataloader.py#L76-L124">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gen_dataset(
    batch_size=1,
    is_training=False,
    shuffle=False,
    input_pipeline_context=None,
    preprocess=None,
    drop_remainder=False
)
</code></pre>

Generate a shared and batched tf.data.Dataset for training/evaluation.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`batch_size`
</td>
<td>
A integer, the returned dataset will be batched by this size.
</td>
</tr><tr>
<td>
`is_training`
</td>
<td>
A boolean, when True, the returned dataset will be optionally
shuffled and repeated as an endless dataset.
</td>
</tr><tr>
<td>
`shuffle`
</td>
<td>
A boolean, when True, the returned dataset will be shuffled to
create randomness during model training.
</td>
</tr><tr>
<td>
`input_pipeline_context`
</td>
<td>
A InputContext instance, used to shared dataset
among multiple workers when distribution strategy is used.
</td>
</tr><tr>
<td>
`preprocess`
</td>
<td>
A function taking three arguments in order, feature, label and
boolean is_training.
</td>
</tr><tr>
<td>
`drop_remainder`
</td>
<td>
boolean, whether the finaly batch drops remainder.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A TF dataset ready to be consumed by Keras model.
</td>
</tr>

</table>



<h3 id="split"><code>split</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/dataloader.py#L185-L197">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>split(
    fraction
)
</code></pre>

Splits dataset into two sub-datasets with the given fraction.

Primarily used for splitting the data set into training and testing sets.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`fraction`
</td>
<td>
float, demonstrates the fraction of the first returned
subdataset in the original data.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The splitted two sub datasets.
</td>
</tr>

</table>



<h3 id="__len__"><code>__len__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/dataloader.py#L126-L130">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__()
</code></pre>
