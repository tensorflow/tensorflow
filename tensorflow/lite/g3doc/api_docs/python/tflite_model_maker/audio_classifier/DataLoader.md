page_type: reference
description: DataLoader for audio tasks.

<devsite-mathjax config="TeX-AMS-MML_SVG"></devsite-mathjax>
<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.audio_classifier.DataLoader" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="from_esc50"/>
<meta itemprop="property" content="from_folder"/>
<meta itemprop="property" content="gen_dataset"/>
<meta itemprop="property" content="split"/>
</div>

# tflite_model_maker.audio_classifier.DataLoader

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/audio_dataloader.py#L132-L386">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



DataLoader for audio tasks.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_model_maker.audio_classifier.DataLoader(
    dataset, size, index_to_label, spec, cache=False
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
    <li><a href="https://www.tensorflow.org/lite/models/modify/model_maker/speech_recognition">Retrain a speech recognition model with TensorFlow Lite Model Maker</a></li>
<li><a href="https://www.tensorflow.org/lite/models/modify/model_maker/audio_classification">Transfer Learning for the Audio Domain with TensorFlow Lite Model Maker</a></li>
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

<h3 id="from_esc50"><code>from_esc50</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/audio_dataloader.py#L205-L260">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_esc50(
    spec, data_path, folds=None, categories=None, shuffle=True, cache=False
)
</code></pre>

Load ESC50 style audio samples.

ESC50 file structure is expalined in <a href="https://github.com/karolpiczak/ESC-50">https://github.com/karolpiczak/ESC-50</a>
Audio files should be put in `${data_path}/audio`
Metadata file should be put in `${data_path}/meta/esc50.csv`

Note that instead of relying on the `target` field in the CSV, a new
`index_to_label` mapping is created based on the alphabet order of the
available categories.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`spec`
</td>
<td>
An instance of audio_spec.YAMNet
</td>
</tr><tr>
<td>
`data_path`
</td>
<td>
A string, location of the ESC50 dataset. It should contain at
</td>
</tr><tr>
<td>
`folds`
</td>
<td>
A integer list of selected folds. If empty, all folds will be
selected.
</td>
</tr><tr>
<td>
`categories`
</td>
<td>
A string list of selected categories. If empty, all categories
will be selected.
</td>
</tr><tr>
<td>
`shuffle`
</td>
<td>
boolean, if True, random shuffle data.
</td>
</tr><tr>
<td>
`cache`
</td>
<td>
str or boolean. When set to True, intermediate results will be
cached in ram. When set to a file path in string, intermediate results
will be cached in this file. Please note that, once file based cache is
created, changes to the input data will have no effects until the cache
file is removed or the filename is changed. More details can be found at
<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#cache">https://www.tensorflow.org/api_docs/python/tf/data/Dataset#cache</a>
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An instance of AudioDataLoader containing audio samples and labels.
</td>
</tr>

</table>



<h3 id="from_folder"><code>from_folder</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/audio_dataloader.py#L150-L203">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_folder(
    spec, data_path, categories=None, shuffle=True, cache=False
)
</code></pre>

Load audio files from a data_path.

- The root `data_path` folder contains a number of folders. The name for
each folder is the name of the audio class.

- Within each folder, there are a number of .wav files. Each .wav file
corresponds to an example. Each .wav file is mono (single-channel) and has
the typical 16 bit pulse-code modulation (PCM) encoding.

- .wav files will be resampled to `spec.target_sample_rate` then fed into
`spec.preprocess_ds` for split and other operations. Normally long wav files
will be framed into multiple clips. And wav files shorter than a certain
threshold will be ignored.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`spec`
</td>
<td>
instance of `audio_spec.BaseSpec`.
</td>
</tr><tr>
<td>
`data_path`
</td>
<td>
string, location to the audio files.
</td>
</tr><tr>
<td>
`categories`
</td>
<td>
A string list of selected categories. If empty, all categories
will be selected.
</td>
</tr><tr>
<td>
`shuffle`
</td>
<td>
boolean, if True, random shuffle data.
</td>
</tr><tr>
<td>
`cache`
</td>
<td>
str or boolean. When set to True, intermediate results will be
cached in ram. When set to a file path in string, intermediate results
will be cached in this file. Please note that, once file based cache is
created, changes to the input data will have no effects until the cache
file is removed or the filename is changed. More details can be found at
<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#cache">https://www.tensorflow.org/api_docs/python/tf/data/Dataset#cache</a>
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`AudioDataLoader` containing audio spectrogram (or any data type generated
by `spec.preprocess_ds`) and labels.
</td>
</tr>

</table>



<h3 id="gen_dataset"><code>gen_dataset</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/audio_dataloader.py#L265-L386">View source</a>

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
shuffled. Data augmentation, if exists, will also be applied to the
returned dataset.
</td>
</tr><tr>
<td>
`shuffle`
</td>
<td>
A boolean, when True, the returned dataset will be shuffled to
create randomness during model training. Only applies when `is_training`
is set to True.
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
Not in use.
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

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/audio_dataloader.py#L262-L263">View source</a>

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

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/audio_dataloader.py#L141-L148">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__()
</code></pre>

Returns the number of audio files in the DataLoader.

Note that one audio file could be framed (mostly via a sliding window of
fixed size) into None or multiple audio clips during training and
evaluation.
