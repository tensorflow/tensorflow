page_type: reference
description: A wrapper class to store the input audio.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.task.audio.TensorAudio" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="clear"/>
<meta itemprop="property" content="create_from_wav_file"/>
<meta itemprop="property" content="load_from_array"/>
<meta itemprop="property" content="load_from_audio_record"/>
</div>

# tflite_support.task.audio.TensorAudio

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/audio/core/tensor_audio.py#L25-L159">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A wrapper class to store the input audio.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.task.audio.TensorAudio(
    audio_format: <a href="../../../tflite_support/task/audio/AudioFormat"><code>tflite_support.task.audio.AudioFormat</code></a>,
    buffer_size: int
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`audio_format`<a id="audio_format"></a>
</td>
<td>
format of the audio.
</td>
</tr><tr>
<td>
`buffer_size`<a id="buffer_size"></a>
</td>
<td>
buffer size of the audio.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`buffer`<a id="buffer"></a>
</td>
<td>
Gets the internal buffer.
</td>
</tr><tr>
<td>
`buffer_size`<a id="buffer_size"></a>
</td>
<td>
Gets the sample count of the audio.
</td>
</tr><tr>
<td>
`format`<a id="format"></a>
</td>
<td>
Gets the audio format of the audio.
</td>
</tr>
</table>



## Methods

<h3 id="clear"><code>clear</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/audio/core/tensor_audio.py#L40-L42">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>clear()
</code></pre>

Clear the internal buffer and fill it with zeros.


<h3 id="create_from_wav_file"><code>create_from_wav_file</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/audio/core/tensor_audio.py#L44-L75">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create_from_wav_file(
    file_name: str, sample_count: int, offset: int = 0
) -> 'TensorAudio'
</code></pre>

Creates `TensorAudio` object from the WAV file.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`file_name`
</td>
<td>
WAV file name.
</td>
</tr><tr>
<td>
`sample_count`
</td>
<td>
The number of samples to read from the WAV file. This value
should match with the input size of the TensorFlow Lite audio model that
will consume the created TensorAudio object. If the WAV file contains
more samples than sample_count, only the samples at the beginning of the
WAV file will be loaded.
</td>
</tr><tr>
<td>
`offset`
</td>
<td>
An optional offset for allowing the user to skip a certain number
samples at the beginning.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`TensorAudio` object.
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
If an input parameter, such as the audio file, is invalid.
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



<h3 id="load_from_array"><code>load_from_array</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/audio/core/tensor_audio.py#L104-L144">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>load_from_array(
    src: np.ndarray, offset: int = 0, size: int = -1
) -> None
</code></pre>

Loads the audio data from a NumPy array.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`src`
</td>
<td>
A NumPy source array contains the input audio.
</td>
</tr><tr>
<td>
`offset`
</td>
<td>
An optional offset for loading a slice of the `src` array to the
buffer.
</td>
</tr><tr>
<td>
`size`
</td>
<td>
An optional size parameter denoting the number of samples to load
from the `src` array.
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
If the input array has an incorrect shape or if
`offset` + `size` exceeds the length of the `src` array.
</td>
</tr>
</table>



<h3 id="load_from_audio_record"><code>load_from_audio_record</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/audio/core/tensor_audio.py#L77-L102">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>load_from_audio_record(
    record: <a href="../../../tflite_support/task/audio/AudioRecord"><code>tflite_support.task.audio.AudioRecord</code></a>
) -> None
</code></pre>

Loads audio data from an AudioRecord instance.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`record`
</td>
<td>
An AudioRecord instance.
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
Raised if the audio record's config is invalid.
</td>
</tr><tr>
<td>
`RuntimeError`
</td>
<td>
Raised if other types of error occurred.
</td>
</tr>
</table>
