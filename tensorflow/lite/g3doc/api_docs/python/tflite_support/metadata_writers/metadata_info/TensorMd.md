page_type: reference
description: A container for common tensor metadata information.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.metadata_writers.metadata_info.TensorMd" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_metadata"/>
</div>

# tflite_support.metadata_writers.metadata_info.TensorMd

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_info.py#L322-L395">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A container for common tensor metadata information.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p><a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/TensorMd"><code>tflite_support.metadata_writers.audio_classifier.metadata_info.TensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/TensorMd"><code>tflite_support.metadata_writers.audio_classifier.metadata_writer.metadata_info.TensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/TensorMd"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_info.TensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/TensorMd"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_writer.metadata_info.TensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/TensorMd"><code>tflite_support.metadata_writers.image_classifier.metadata_info.TensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/TensorMd"><code>tflite_support.metadata_writers.image_classifier.metadata_writer.metadata_info.TensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/TensorMd"><code>tflite_support.metadata_writers.image_segmenter.metadata_info.TensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/TensorMd"><code>tflite_support.metadata_writers.image_segmenter.metadata_writer.metadata_info.TensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/TensorMd"><code>tflite_support.metadata_writers.nl_classifier.metadata_info.TensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/TensorMd"><code>tflite_support.metadata_writers.nl_classifier.metadata_writer.metadata_info.TensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/TensorMd"><code>tflite_support.metadata_writers.object_detector.metadata_info.TensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/TensorMd"><code>tflite_support.metadata_writers.object_detector.metadata_writer.metadata_info.TensorMd</code></a></p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.metadata_writers.metadata_info.TensorMd(
    name: Optional[str] = None,
    description: Optional[str] = None,
    min_values: Optional[List[float]] = None,
    max_values: Optional[List[float]] = None,
    content_type: <a href="../../../tflite_support/metadata_schema_py_generated/ContentProperties"><code>tflite_support.metadata_schema_py_generated.ContentProperties</code></a> = _metadata_fb.ContentProperties.FeatureProperties,
    associated_files: Optional[List[Type[AssociatedFileMd]]] = None,
    tensor_name: Optional[str] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

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
`min_values`<a id="min_values"></a>
</td>
<td>
per-channel minimum value of the tensor.
</td>
</tr><tr>
<td>
`max_values`<a id="max_values"></a>
</td>
<td>
per-channel maximum value of the tensor.
</td>
</tr><tr>
<td>
`content_type`<a id="content_type"></a>
</td>
<td>
content_type of the tensor.
</td>
</tr><tr>
<td>
`associated_files`<a id="associated_files"></a>
</td>
<td>
information of the associated files in the tensor.
</td>
</tr><tr>
<td>
`tensor_name`<a id="tensor_name"></a>
</td>
<td>
name of the corresponding tensor [1] in the TFLite model. It is
  used to locate the corresponding tensor and decide the order of the tensor
  metadata [2] when populating model metadata.
[1]:
  https://github.com/tensorflow/tensorflow/blob/cb67fef35567298b40ac166b0581cd8ad68e5a3a/tensorflow/lite/schema/schema.fbs#L1129-L1136
[2]:
  https://github.com/tensorflow/tflite-support/blob/b2a509716a2d71dfff706468680a729cc1604cff/tensorflow_lite_support/metadata/metadata_schema.fbs#L595-L612
</td>
</tr>
</table>



## Methods

<h3 id="create_metadata"><code>create_metadata</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_info.py#L358-L395">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_metadata() -> <a href="../../../tflite_support/metadata_schema_py_generated/TensorMetadataT"><code>tflite_support.metadata_schema_py_generated.TensorMetadataT</code></a>
</code></pre>

Creates the input tensor metadata based on the information.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Flatbuffers Python object of the input metadata.
</td>
</tr>

</table>
