page_type: reference
description: DataLoader for question answering.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.question_answer.DataLoader" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="from_squad"/>
<meta itemprop="property" content="gen_dataset"/>
<meta itemprop="property" content="split"/>
</div>

# tflite_model_maker.question_answer.DataLoader

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/text_dataloader.py#L292-L393">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



DataLoader for question answering.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_model_maker.question_answer.DataLoader(
    dataset, size, version_2_with_negative, examples, features, squad_file
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
    <li><a href="https://www.tensorflow.org/lite/models/modify/model_maker/question_answer">BERT Question Answer with TensorFlow Lite Model Maker</a></li>
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

<h3 id="from_squad"><code>from_squad</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/text_dataloader.py#L304-L350">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_squad(
    filename,
    model_spec,
    is_training=True,
    version_2_with_negative=False,
    cache_dir=None
)
</code></pre>

Loads data in SQuAD format and preproecess text according to `model_spec`.


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
`version_2_with_negative`
</td>
<td>
Whether it's SQuAD 2.0 format.
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
QuestionAnswerDataLoader object.
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

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/dataloader.py#L132-L144">View source</a>

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
