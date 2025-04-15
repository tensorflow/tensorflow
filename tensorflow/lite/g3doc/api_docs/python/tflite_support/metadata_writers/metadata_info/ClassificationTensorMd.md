page_type: reference
description: A container for the classification tensor metadata information.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.metadata_writers.metadata_info.ClassificationTensorMd" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_metadata"/>
</div>

# tflite_support.metadata_writers.metadata_info.ClassificationTensorMd

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_info.py#L597-L674">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A container for the classification tensor metadata information.

Inherits From: [`TensorMd`](../../../tflite_support/metadata_writers/metadata_info/TensorMd)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p><a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/ClassificationTensorMd"><code>tflite_support.metadata_writers.audio_classifier.metadata_info.ClassificationTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/ClassificationTensorMd"><code>tflite_support.metadata_writers.audio_classifier.metadata_writer.metadata_info.ClassificationTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/ClassificationTensorMd"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_info.ClassificationTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/ClassificationTensorMd"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_writer.metadata_info.ClassificationTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/ClassificationTensorMd"><code>tflite_support.metadata_writers.image_classifier.metadata_info.ClassificationTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/ClassificationTensorMd"><code>tflite_support.metadata_writers.image_classifier.metadata_writer.metadata_info.ClassificationTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/ClassificationTensorMd"><code>tflite_support.metadata_writers.image_segmenter.metadata_info.ClassificationTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/ClassificationTensorMd"><code>tflite_support.metadata_writers.image_segmenter.metadata_writer.metadata_info.ClassificationTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/ClassificationTensorMd"><code>tflite_support.metadata_writers.nl_classifier.metadata_info.ClassificationTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/ClassificationTensorMd"><code>tflite_support.metadata_writers.nl_classifier.metadata_writer.metadata_info.ClassificationTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/ClassificationTensorMd"><code>tflite_support.metadata_writers.object_detector.metadata_info.ClassificationTensorMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/ClassificationTensorMd"><code>tflite_support.metadata_writers.object_detector.metadata_writer.metadata_info.ClassificationTensorMd</code></a></p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.metadata_writers.metadata_info.ClassificationTensorMd(
    name: Optional[str] = None,
    description: Optional[str] = None,
    label_files: Optional[List[LabelFileMd]] = None,
    tensor_type: Optional[_schema_fb.TensorType] = None,
    score_calibration_md: Optional[<a href="../../../tflite_support/metadata_writers/metadata_info/ScoreCalibrationMd"><code>tflite_support.metadata_writers.metadata_info.ScoreCalibrationMd</code></a>] = None,
    tensor_name: Optional[str] = None
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
`label_files`<a id="label_files"></a>
</td>
<td>
information of the label files [1] in the classification
tensor.
</td>
</tr><tr>
<td>
`tensor_type`<a id="tensor_type"></a>
</td>
<td>
data type of the tensor.
</td>
</tr><tr>
<td>
`score_calibration_md`<a id="score_calibration_md"></a>
</td>
<td>
information of the score calibration files operation
[2] in the classification tensor.
</td>
</tr><tr>
<td>
`tensor_name`<a id="tensor_name"></a>
</td>
<td>
name of the corresponding tensor [3] in the TFLite model. It
  is used to locate the corresponding classification tensor and decide the
  order of the tensor metadata [4] when populating model metadata.
[1]:
  https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L95
[2]:
  https://github.com/tensorflow/tflite-support/blob/5e0cdf5460788c481f5cd18aab8728ec36cf9733/tensorflow_lite_support/metadata/metadata_schema.fbs#L434
[3]:
  https://github.com/tensorflow/tensorflow/blob/cb67fef35567298b40ac166b0581cd8ad68e5a3a/tensorflow/lite/schema/schema.fbs#L1129-L1136
[4]:
  https://github.com/tensorflow/tflite-support/blob/b2a509716a2d71dfff706468680a729cc1604cff/tensorflow_lite_support/metadata/metadata_schema.fbs#L595-L612
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`label_files`<a id="label_files"></a>
</td>
<td>
information of the label files [1] in the classification
tensor.
</td>
</tr><tr>
<td>
`score_calibration_md`<a id="score_calibration_md"></a>
</td>
<td>
information of the score calibration operation [2] in
  the classification tensor.
[1]:
  https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L95
[2]:
  https://github.com/tensorflow/tflite-support/blob/5e0cdf5460788c481f5cd18aab8728ec36cf9733/tensorflow_lite_support/metadata/metadata_schema.fbs#L434
</td>
</tr>
</table>



## Methods

<h3 id="create_metadata"><code>create_metadata</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_info.py#L667-L674">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_metadata() -> <a href="../../../tflite_support/metadata_schema_py_generated/TensorMetadataT"><code>tflite_support.metadata_schema_py_generated.TensorMetadataT</code></a>
</code></pre>

Creates the classification tensor metadata based on the information.
