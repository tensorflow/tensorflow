page_type: reference
description: Class that performs dense feature vector extraction on audio.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.task.audio.AudioEmbedder" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="cosine_similarity"/>
<meta itemprop="property" content="create_audio_record"/>
<meta itemprop="property" content="create_from_file"/>
<meta itemprop="property" content="create_from_options"/>
<meta itemprop="property" content="create_input_tensor_audio"/>
<meta itemprop="property" content="embed"/>
<meta itemprop="property" content="get_embedding_dimension"/>
</div>

# tflite_support.task.audio.AudioEmbedder

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/audio/audio_embedder.py#L47-L167">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Class that performs dense feature vector extraction on audio.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.task.audio.AudioEmbedder(
    options: <a href="../../../tflite_support/task/audio/AudioEmbedderOptions"><code>tflite_support.task.audio.AudioEmbedderOptions</code></a>,
    cpp_embedder: _CppAudioEmbedder
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`number_of_output_layers`<a id="number_of_output_layers"></a>
</td>
<td>
Gets the number of output layers of the model.
</td>
</tr><tr>
<td>
`required_audio_format`<a id="required_audio_format"></a>
</td>
<td>
Gets the required audio format for the model.
</td>
</tr><tr>
<td>
`required_input_buffer_size`<a id="required_input_buffer_size"></a>
</td>
<td>
Gets the required input buffer size for the model.
</td>
</tr>
</table>



## Methods

<h3 id="cosine_similarity"><code>cosine_similarity</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/audio/audio_embedder.py#L133-L136">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cosine_similarity(
    u: <a href="../../../tflite_support/task/processor/FeatureVector"><code>tflite_support.task.processor.FeatureVector</code></a>,
    v: <a href="../../../tflite_support/task/processor/FeatureVector"><code>tflite_support.task.processor.FeatureVector</code></a>
) -> float
</code></pre>

Computes cosine similarity [1] between two feature vectors.


<h3 id="create_audio_record"><code>create_audio_record</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/audio/audio_embedder.py#L105-L113">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_audio_record() -> <a href="../../../tflite_support/task/audio/AudioRecord"><code>tflite_support.task.audio.AudioRecord</code></a>
</code></pre>

Creates an AudioRecord instance to record audio.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An AudioRecord instance.
</td>
</tr>

</table>



<h3 id="create_from_file"><code>create_from_file</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/audio/audio_embedder.py#L56-L73">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create_from_file(
    file_path: str
) -> 'AudioEmbedder'
</code></pre>

Creates the `AudioEmbedder` object from a TensorFlow Lite model.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`file_path`
</td>
<td>
Path to the model.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`AudioEmbedder` object that's created from `options`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If failed to create `AudioEmbedder` object from the provided
file such as invalid file.
</td>
</tr><tr>
<td>
`RuntimeError`
</td>
<td>
If other types of error occurred.
</td>
</tr>
</table>



<h3 id="create_from_options"><code>create_from_options</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/audio/audio_embedder.py#L75-L93">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create_from_options(
    options: <a href="../../../tflite_support/task/audio/AudioEmbedderOptions"><code>tflite_support.task.audio.AudioEmbedderOptions</code></a>
) -> 'AudioEmbedder'
</code></pre>

Creates the `AudioEmbedder` object from audio embedder options.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`options`
</td>
<td>
Options for the audio embedder task.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`AudioEmbedder` object that's created from `options`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If failed to create `AudioEmbedder` object from
`AudioEmbedderOptions` such as missing the model.
</td>
</tr><tr>
<td>
`RuntimeError`
</td>
<td>
If other types of error occurred.
</td>
</tr>
</table>



<h3 id="create_input_tensor_audio"><code>create_input_tensor_audio</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/audio/audio_embedder.py#L95-L103">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_input_tensor_audio() -> <a href="../../../tflite_support/task/audio/TensorAudio"><code>tflite_support.task.audio.TensorAudio</code></a>
</code></pre>

Creates a TensorAudio instance to store the audio input.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A TensorAudio instance.
</td>
</tr>

</table>



<h3 id="embed"><code>embed</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/audio/audio_embedder.py#L115-L131">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>embed(
    audio: <a href="../../../tflite_support/task/audio/TensorAudio"><code>tflite_support.task.audio.TensorAudio</code></a>
) -> <a href="../../../tflite_support/task/processor/EmbeddingResult"><code>tflite_support.task.processor.EmbeddingResult</code></a>
</code></pre>

Performs actual feature vector extraction on the provided audio.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`audio`
</td>
<td>
Tensor audio, used to extract the feature vectors.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
embedding result.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If any of the input arguments is invalid.
</td>
</tr><tr>
<td>
`RuntimeError`
</td>
<td>
If failed to calculate the embedding vector.
</td>
</tr>
</table>



<h3 id="get_embedding_dimension"><code>get_embedding_dimension</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/audio/audio_embedder.py#L138-L148">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_embedding_dimension(
    output_index: int
) -> int
</code></pre>

Gets the dimensionality of the embedding output.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`output_index`
</td>
<td>
The output index of output layer.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Dimensionality of the embedding output by the output_index'th output
layer. Returns -1 if `output_index` is out of bounds.
</td>
</tr>

</table>
