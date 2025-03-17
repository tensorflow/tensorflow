page_type: reference
description: A container for input image tensor metadata information.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.metadata_writers.metadata_info.InputImageTensorMd" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_metadata"/>
</div>

# tflite_support.metadata_writers.metadata_info.InputImageTensorMd

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_info.py#L398-L489">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A container for input image tensor metadata information.

Inherits From: [`TensorMd`](../../../tflite_support/metadata_writers/metadata_info/TensorMd)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p><a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/InputImageTensorMd"><code>tflite_support.metadata_writers.audio_classifier.metadata_info.InputImageTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/InputImageTensorMd"><code>tflite_support.metadata_writers.audio_classifier.metadata_writer.metadata_info.InputImageTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/InputImageTensorMd"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_info.InputImageTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/InputImageTensorMd"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_writer.metadata_info.InputImageTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/InputImageTensorMd"><code>tflite_support.metadata_writers.image_classifier.metadata_info.InputImageTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/InputImageTensorMd"><code>tflite_support.metadata_writers.image_classifier.metadata_writer.metadata_info.InputImageTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/InputImageTensorMd"><code>tflite_support.metadata_writers.image_segmenter.metadata_info.InputImageTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/InputImageTensorMd"><code>tflite_support.metadata_writers.image_segmenter.metadata_writer.metadata_info.InputImageTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/InputImageTensorMd"><code>tflite_support.metadata_writers.nl_classifier.metadata_info.InputImageTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/InputImageTensorMd"><code>tflite_support.metadata_writers.nl_classifier.metadata_writer.metadata_info.InputImageTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/InputImageTensorMd"><code>tflite_support.metadata_writers.object_detector.metadata_info.InputImageTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/InputImageTensorMd"><code>tflite_support.metadata_writers.object_detector.metadata_writer.metadata_info.InputImageTensorMd</code></a></p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.metadata_writers.metadata_info.InputImageTensorMd(
    name: Optional[str] = None,
    description: Optional[str] = None,
    norm_mean: Optional[List[float]] = None,
    norm_std: Optional[List[float]] = None,
    color_space_type: Optional[<a href="../../../tflite_support/metadata_schema_py_generated/ColorSpaceType"><code>tflite_support.metadata_schema_py_generated.ColorSpaceType</code></a>] = _metadata_fb.ColorSpaceType.UNKNOWN,
    tensor_type: Optional[_schema_fb.TensorType] = None
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
`norm_mean`<a id="norm_mean"></a>
</td>
<td>
the mean value used in tensor normalization [1].
</td>
</tr><tr>
<td>
`norm_std`<a id="norm_std"></a>
</td>
<td>
the std value used in the tensor normalization [1]. norm_mean
and norm_std must have the same dimension.
</td>
</tr><tr>
<td>
`color_space_type`<a id="color_space_type"></a>
</td>
<td>
the color space type of the input image [2].
</td>
</tr><tr>
<td>
`tensor_type`<a id="tensor_type"></a>
</td>
<td>
data type of the tensor.
[1]:
  https://www.tensorflow.org/lite/convert/metadata#normalization_and_quantization_parameters
[2]:
https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L172
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
if norm_mean and norm_std have different dimensions.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`norm_mean`<a id="norm_mean"></a>
</td>
<td>
the mean value used in tensor normalization [1].
</td>
</tr><tr>
<td>
`norm_std`<a id="norm_std"></a>
</td>
<td>
the std value used in the tensor normalization [1]. norm_mean and
norm_std must have the same dimension.
</td>
</tr><tr>
<td>
`color_space_type`<a id="color_space_type"></a>
</td>
<td>
the color space type of the input image [2].
[1]:
  https://www.tensorflow.org/lite/convert/metadata#normalization_and_quantization_parameters
[2]:
  https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L172
</td>
</tr>
</table>



## Methods

<h3 id="create_metadata"><code>create_metadata</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_info.py#L472-L489">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_metadata() -> <a href="../../../tflite_support/metadata_schema_py_generated/TensorMetadataT"><code>tflite_support.metadata_schema_py_generated.TensorMetadataT</code></a>
</code></pre>

Creates the input image metadata based on the information.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Flatbuffers Python object of the input image metadata.
</td>
</tr>

</table>
