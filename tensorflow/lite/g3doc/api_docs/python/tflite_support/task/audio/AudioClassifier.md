page_type: reference
description: Class that performs classification on audio.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.task.audio.AudioClassifier" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="classify"/>
<meta itemprop="property" content="create_audio_record"/>
<meta itemprop="property" content="create_from_file"/>
<meta itemprop="property" content="create_from_options"/>
<meta itemprop="property" content="create_input_tensor_audio"/>
</div>

# tflite_support.task.audio.AudioClassifier

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/audio/audio_classifier.py#L48-L150">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Class that performs classification on audio.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.task.audio.AudioClassifier(
    options: <a href="../../../tflite_support/task/audio/AudioClassifierOptions"><code>tflite_support.task.audio.AudioClassifierOptions</code></a>,
    classifier: _CppAudioClassifier
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
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

<h3 id="classify"><code>classify</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/audio/audio_classifier.py#L117-L136">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>classify(
    audio: <a href="../../../tflite_support/task/audio/TensorAudio"><code>tflite_support.task.audio.TensorAudio</code></a>
) -> <a href="../../../tflite_support/task/processor/ClassificationResult"><code>tflite_support.task.processor.ClassificationResult</code></a>
</code></pre>

Performs classification on the provided TensorAudio.


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
classification result.
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
If failed to run audio classification.
</td>
</tr>
</table>



<h3 id="create_audio_record"><code>create_audio_record</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/audio/audio_classifier.py#L107-L115">View source</a>

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

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/audio/audio_classifier.py#L58-L75">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create_from_file(
    file_path: str
) -> 'AudioClassifier'
</code></pre>

Creates the `AudioClassifier` object from a TensorFlow Lite model.


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
`AudioClassifier` object that's created from `options`.
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
If failed to create `AudioClassifier` object from the provided
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

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/audio/audio_classifier.py#L77-L95">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create_from_options(
    options: <a href="../../../tflite_support/task/audio/AudioClassifierOptions"><code>tflite_support.task.audio.AudioClassifierOptions</code></a>
) -> 'AudioClassifier'
</code></pre>

Creates the `AudioClassifier` object from audio classifier options.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`options`
</td>
<td>
Options for the audio classifier task.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`AudioClassifier` object that's created from `options`.
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
If failed to create `AudioClassifier` object from
`AudioClassifierOptions` such as missing the model.
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

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/audio/audio_classifier.py#L97-L105">View source</a>

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
