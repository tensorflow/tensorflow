page_type: reference
description: A container for the input audio tensor metadata information.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.metadata_writers.metadata_info.InputAudioTensorMd" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_metadata"/>
</div>

# tflite_support.metadata_writers.metadata_info.InputAudioTensorMd

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_info.py#L542-L594">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A container for the input audio tensor metadata information.

Inherits From: [`TensorMd`](../../../tflite_support/metadata_writers/metadata_info/TensorMd)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p><a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/InputAudioTensorMd"><code>tflite_support.metadata_writers.audio_classifier.metadata_info.InputAudioTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/InputAudioTensorMd"><code>tflite_support.metadata_writers.audio_classifier.metadata_writer.metadata_info.InputAudioTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/InputAudioTensorMd"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_info.InputAudioTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/InputAudioTensorMd"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_writer.metadata_info.InputAudioTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/InputAudioTensorMd"><code>tflite_support.metadata_writers.image_classifier.metadata_info.InputAudioTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/InputAudioTensorMd"><code>tflite_support.metadata_writers.image_classifier.metadata_writer.metadata_info.InputAudioTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/InputAudioTensorMd"><code>tflite_support.metadata_writers.image_segmenter.metadata_info.InputAudioTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/InputAudioTensorMd"><code>tflite_support.metadata_writers.image_segmenter.metadata_writer.metadata_info.InputAudioTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/InputAudioTensorMd"><code>tflite_support.metadata_writers.nl_classifier.metadata_info.InputAudioTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/InputAudioTensorMd"><code>tflite_support.metadata_writers.nl_classifier.metadata_writer.metadata_info.InputAudioTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/InputAudioTensorMd"><code>tflite_support.metadata_writers.object_detector.metadata_info.InputAudioTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/InputAudioTensorMd"><code>tflite_support.metadata_writers.object_detector.metadata_writer.metadata_info.InputAudioTensorMd</code></a></p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.metadata_writers.metadata_info.InputAudioTensorMd(
    name: Optional[str] = None,
    description: Optional[str] = None,
    sample_rate: int = 0,
    channels: int = 0
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`name`<a id="name"></a>
</td>
<td>
name of the tensor.
</td>
</tr><tr>
<td>
`description`<a id="description"></a>
</td>
<td>
description of what the tensor is.
</td>
</tr><tr>
<td>
`sample_rate`<a id="sample_rate"></a>
</td>
<td>
the sample rate in Hz when the audio was captured.
</td>
</tr><tr>
<td>
`channels`<a id="channels"></a>
</td>
<td>
the channel count of the audio.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`sample_rate`<a id="sample_rate"></a>
</td>
<td>
the sample rate in Hz when the audio was captured.
</td>
</tr><tr>
<td>
`channels`<a id="channels"></a>
</td>
<td>
the channel count of the audio.
</td>
</tr>
</table>



## Methods

<h3 id="create_metadata"><code>create_metadata</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_info.py#L571-L594">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_metadata() -> <a href="../../../tflite_support/metadata_schema_py_generated/TensorMetadataT"><code>tflite_support.metadata_schema_py_generated.TensorMetadataT</code></a>
</code></pre>

Creates the input audio metadata based on the information.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Flatbuffers Python object of the input audio metadata.
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
if any value of sample_rate, channels is negative.
</td>
</tr>
</table>
