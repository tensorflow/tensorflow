page_type: reference
description: A class to record audio in a streaming basis.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.task.audio.AudioRecord" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="read"/>
<meta itemprop="property" content="start_recording"/>
<meta itemprop="property" content="stop"/>
</div>

# tflite_support.task.audio.AudioRecord

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/audio/core/audio_record.py#L30-L126">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A class to record audio in a streaming basis.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.task.audio.AudioRecord(
    channels: int, sampling_rate: int, buffer_size: int
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`channels`<a id="channels"></a>
</td>
<td>
Number of input channels.
</td>
</tr><tr>
<td>
`sampling_rate`<a id="sampling_rate"></a>
</td>
<td>
Sampling rate in Hertz.
</td>
</tr><tr>
<td>
`buffer_size`<a id="buffer_size"></a>
</td>
<td>
Size of the ring buffer in number of samples.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
if any of the arguments is non-positive.
</td>
</tr><tr>
<td>
`ImportError`<a id="ImportError"></a>
</td>
<td>
if failed to import `sounddevice`.
</td>
</tr><tr>
<td>
`OSError`<a id="OSError"></a>
</td>
<td>
if failed to load `PortAudio`.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`buffer_size`<a id="buffer_size"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`channels`<a id="channels"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`sampling_rate`<a id="sampling_rate"></a>
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="read"><code>read</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/audio/core/audio_record.py#L108-L126">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>read(
    size: int
) -> np.ndarray
</code></pre>

Reads the latest audio data captured in the buffer.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`size`
</td>
<td>
Number of samples to read from the buffer.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A NumPy array containing the audio data.
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
Raised if `size` is larger than the buffer size.
</td>
</tr>
</table>



<h3 id="start_recording"><code>start_recording</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/audio/core/audio_record.py#L96-L102">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>start_recording() -> None
</code></pre>

Starts the audio recording.


<h3 id="stop"><code>stop</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/audio/core/audio_record.py#L104-L106">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>stop() -> None
</code></pre>

Stops the audio recording.
