page_type: reference
description: Model good at detecting environmental sounds, using YAMNet embedding.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.audio_classifier.YamNetSpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_model"/>
<meta itemprop="property" content="create_serving_model"/>
<meta itemprop="property" content="export_tflite"/>
<meta itemprop="property" content="get_default_quantization_config"/>
<meta itemprop="property" content="preprocess_ds"/>
<meta itemprop="property" content="run_classifier"/>
<meta itemprop="property" content="EMBEDDING_SIZE"/>
<meta itemprop="property" content="EXPECTED_WAVEFORM_LENGTH"/>
</div>

# tflite_model_maker.audio_classifier.YamNetSpec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/audio_spec.py#L411-L641">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Model good at detecting environmental sounds, using YAMNet embedding.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_model_maker.audio_classifier.YamNetSpec(
    model_dir: None = None,
    strategy: None = None,
    yamnet_model_handle=&#x27;https://tfhub.dev/google/yamnet/1&#x27;,
    frame_length=EXPECTED_WAVEFORM_LENGTH,
    frame_step=(EXPECTED_WAVEFORM_LENGTH // 2),
    keep_yamnet_and_custom_heads=True
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
`model_dir`<a id="model_dir"></a>
</td>
<td>
The location to save the model checkpoint files.
</td>
</tr><tr>
<td>
`strategy`<a id="strategy"></a>
</td>
<td>
An instance of TF distribute strategy. If none, it will use the
default strategy (either SingleDeviceStrategy or the current scoped
strategy.
</td>
</tr><tr>
<td>
`yamnet_model_handle`<a id="yamnet_model_handle"></a>
</td>
<td>
Path of the TFHub model for retrining.
</td>
</tr><tr>
<td>
`frame_length`<a id="frame_length"></a>
</td>
<td>
The number of samples in each audio frame. If the audio file
is shorter than `frame_length`, then the audio file will be ignored.
</td>
</tr><tr>
<td>
`frame_step`<a id="frame_step"></a>
</td>
<td>
The number of samples between two audio frames. This value
should be smaller than `frame_length`, otherwise some samples will be
ignored.
</td>
</tr><tr>
<td>
`keep_yamnet_and_custom_heads`<a id="keep_yamnet_and_custom_heads"></a>
</td>
<td>
Boolean, decides if the final TFLite model
contains both YAMNet and custom trained classification heads. When set
to False, only the trained custom head will be preserved.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`target_sample_rate`<a id="target_sample_rate"></a>
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="create_model"><code>create_model</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/audio_spec.py#L476-L485">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_model(
    num_classes, train_whole_model=False
)
</code></pre>




<h3 id="create_serving_model"><code>create_serving_model</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/audio_spec.py#L583-L602">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_serving_model(
    training_model
)
</code></pre>

Create a model for serving.


<h3 id="export_tflite"><code>export_tflite</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/audio_spec.py#L604-L641">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>export_tflite(
    model,
    tflite_filepath,
    with_metadata=True,
    export_metadata_json_file=True,
    index_to_label=None,
    quantization_config=None
)
</code></pre>

Converts the retrained model to tflite format and saves it.

This method overrides the default `CustomModel._export_tflite` method, and
include the spectrom extraction in the model.

The exported model has input shape (1, number of wav samples)

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`model`
</td>
<td>
An instance of the keras classification model to be exported.
</td>
</tr><tr>
<td>
`tflite_filepath`
</td>
<td>
File path to save tflite model.
</td>
</tr><tr>
<td>
`with_metadata`
</td>
<td>
Whether the output tflite model contains metadata.
</td>
</tr><tr>
<td>
`export_metadata_json_file`
</td>
<td>
Whether to export metadata in json file. If
True, export the metadata in the same directory as tflite model. Used
only if `with_metadata` is True.
</td>
</tr><tr>
<td>
`index_to_label`
</td>
<td>
A list that map from index to label class name.
</td>
</tr><tr>
<td>
`quantization_config`
</td>
<td>
Configuration for post-training quantization.
</td>
</tr>
</table>



<h3 id="get_default_quantization_config"><code>get_default_quantization_config</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/audio_spec.py#L169-L171">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_default_quantization_config()
</code></pre>

Gets the default quantization configuration.


<h3 id="preprocess_ds"><code>preprocess_ds</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/audio_spec.py#L528-L539">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>preprocess_ds(
    ds, is_training=False, cache_fn=None
)
</code></pre>

Returns a preprocessed dataset.


<h3 id="run_classifier"><code>run_classifier</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/audio_spec.py#L487-L493">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>run_classifier(
    model, epochs, train_ds, validation_ds, **kwargs
)
</code></pre>








<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
EMBEDDING_SIZE<a id="EMBEDDING_SIZE"></a>
</td>
<td>
`1024`
</td>
</tr><tr>
<td>
EXPECTED_WAVEFORM_LENGTH<a id="EXPECTED_WAVEFORM_LENGTH"></a>
</td>
<td>
`15600`
</td>
</tr>
</table>
