page_type: reference
description: Writes metadata into an object detector.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.metadata_writers.object_detector.MetadataWriter" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_for_inference"/>
<meta itemprop="property" content="create_from_metadata"/>
<meta itemprop="property" content="create_from_metadata_info"/>
<meta itemprop="property" content="get_metadata_json"/>
<meta itemprop="property" content="get_populated_metadata_json"/>
<meta itemprop="property" content="populate"/>
</div>

# tflite_support.metadata_writers.object_detector.MetadataWriter

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/object_detector.py#L103-L295">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Writes metadata into an object detector.

Inherits From: [`MetadataWriter`](../../../tflite_support/metadata_writers/audio_classifier/metadata_writer/MetadataWriter)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.metadata_writers.object_detector.MetadataWriter(
    model_buffer: bytearray,
    metadata_buffer: Optional[bytearray] = None,
    associated_files: Optional[List[str]] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`model_buffer`<a id="model_buffer"></a>
</td>
<td>
valid buffer of the model file.
</td>
</tr><tr>
<td>
`metadata_buffer`<a id="metadata_buffer"></a>
</td>
<td>
valid buffer of the metadata.
</td>
</tr><tr>
<td>
`associated_files`<a id="associated_files"></a>
</td>
<td>
path to the associated files to be populated.
</td>
</tr>
</table>



## Methods

<h3 id="create_for_inference"><code>create_for_inference</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/object_detector.py#L234-L295">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create_for_inference(
    model_buffer: bytearray,
    input_norm_mean: List[float],
    input_norm_std: List[float],
    label_file_paths: List[str],
    score_calibration_md: Optional[<a href="../../../tflite_support/metadata_writers/metadata_info/ScoreCalibrationMd"><code>tflite_support.metadata_writers.metadata_info.ScoreCalibrationMd</code></a>] = None
)
</code></pre>

Creates mandatory metadata for TFLite Support inference.

The parameters required in this method are mandatory when using TFLite
Support features, such as Task library and Codegen tool (Android Studio ML
Binding). Other metadata fields will be set to default. If other fields need
to be filled, use the method `create_from_metadata_info` to edit them.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`model_buffer`
</td>
<td>
valid buffer of the model file.
</td>
</tr><tr>
<td>
`input_norm_mean`
</td>
<td>
the mean value used in the input tensor normalization
[1].
</td>
</tr><tr>
<td>
`input_norm_std`
</td>
<td>
the std value used in the input tensor normalizarion [1].
</td>
</tr><tr>
<td>
`label_file_paths`
</td>
<td>
paths to the label files [2] in the category tensor.
Pass in an empty list, If the model does not have any label file.
</td>
</tr><tr>
<td>
`score_calibration_md`
</td>
<td>
information of the score calibration operation [3]
  in the classification tensor. Optional if the model does not use score
  calibration.
[1]:
  https://www.tensorflow.org/lite/convert/metadata#normalization_and_quantization_parameters
[2]:
  https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L108
[3]:
  https://github.com/tensorflow/tflite-support/blob/5e0cdf5460788c481f5cd18aab8728ec36cf9733/tensorflow_lite_support/metadata/metadata_schema.fbs#L434
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A MetadataWriter object.
</td>
</tr>

</table>



<h3 id="create_from_metadata"><code>create_from_metadata</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_writer.py#L92-L162">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create_from_metadata(
    model_buffer: bytearray,
    model_metadata: Optional[<a href="../../../tflite_support/metadata_schema_py_generated/ModelMetadataT"><code>tflite_support.metadata_schema_py_generated.ModelMetadataT</code></a>] = None,
    input_metadata: Optional[List[_metadata_fb.TensorMetadataT]] = None,
    output_metadata: Optional[List[_metadata_fb.TensorMetadataT]] = None,
    associated_files: Optional[List[str]] = None,
    input_process_units: Optional[List[_metadata_fb.ProcessUnitT]] = None,
    output_process_units: Optional[List[_metadata_fb.ProcessUnitT]] = None
)
</code></pre>

Creates MetadataWriter based on the metadata Flatbuffers Python Objects.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`model_buffer`
</td>
<td>
valid buffer of the model file.
</td>
</tr><tr>
<td>
`model_metadata`
</td>
<td>
general model metadata [1]. The subgraph_metadata will be
refreshed with input_metadata and output_metadata.
</td>
</tr><tr>
<td>
`input_metadata`
</td>
<td>
a list of metadata of the input tensors [2].
</td>
</tr><tr>
<td>
`output_metadata`
</td>
<td>
a list of metadata of the output tensors [3].
</td>
</tr><tr>
<td>
`associated_files`
</td>
<td>
path to the associated files to be populated.
</td>
</tr><tr>
<td>
`input_process_units`
</td>
<td>
a lits of metadata of the input process units [4].
</td>
</tr><tr>
<td>
`output_process_units`
</td>
<td>
a lits of metadata of the output process units [5].
[1]:
  https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L640-L681
[2]:
  https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L590
[3]:
  https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L599
[4]:
  https://github.com/tensorflow/tflite-support/blob/b5cc57c74f7990d8bc055795dfe8d50267064a57/tensorflow_lite_support/metadata/metadata_schema.fbs#L646
[5]:
  https://github.com/tensorflow/tflite-support/blob/b5cc57c74f7990d8bc055795dfe8d50267064a57/tensorflow_lite_support/metadata/metadata_schema.fbs#L650
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A MetadataWriter Object.
</td>
</tr>

</table>



<h3 id="create_from_metadata_info"><code>create_from_metadata_info</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/object_detector.py#L106-L232">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create_from_metadata_info(
    model_buffer: bytearray,
    general_md: Optional[<a href="../../../tflite_support/metadata_writers/metadata_info/GeneralMd"><code>tflite_support.metadata_writers.metadata_info.GeneralMd</code></a>] = None,
    input_md: Optional[<a href="../../../tflite_support/metadata_writers/metadata_info/InputImageTensorMd"><code>tflite_support.metadata_writers.metadata_info.InputImageTensorMd</code></a>] = None,
    output_location_md: Optional[<a href="../../../tflite_support/metadata_writers/metadata_info/TensorMd"><code>tflite_support.metadata_writers.metadata_info.TensorMd</code></a>] = None,
    output_category_md: Optional[<a href="../../../tflite_support/metadata_writers/metadata_info/CategoryTensorMd"><code>tflite_support.metadata_writers.metadata_info.CategoryTensorMd</code></a>] = None,
    output_score_md: Union[None, <a href="../../../tflite_support/metadata_writers/metadata_info/TensorMd"><code>tflite_support.metadata_writers.metadata_info.TensorMd</code></a>, <a href="../../../tflite_support/metadata_writers/metadata_info/ClassificationTensorMd"><code>tflite_support.metadata_writers.metadata_info.ClassificationTensorMd</code></a>] = None,
    output_number_md: Optional[<a href="../../../tflite_support/metadata_writers/metadata_info/TensorMd"><code>tflite_support.metadata_writers.metadata_info.TensorMd</code></a>] = None
)
</code></pre>

Creates MetadataWriter based on general/input/outputs information.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`model_buffer`
</td>
<td>
valid buffer of the model file.
</td>
</tr><tr>
<td>
`general_md`
</td>
<td>
general information about the model.
</td>
</tr><tr>
<td>
`input_md`
</td>
<td>
input image tensor informaton.
</td>
</tr><tr>
<td>
`output_location_md`
</td>
<td>
output location tensor informaton. The location tensor
is a multidimensional array of [N][4] floating point values between 0
and 1, the inner arrays representing bounding boxes in the form [top,
left, bottom, right].
</td>
</tr><tr>
<td>
`output_category_md`
</td>
<td>
output category tensor information. The category
tensor is an array of N integers (output as floating point values) each
indicating the index of a class label from the labels file.
</td>
</tr><tr>
<td>
`output_score_md`
</td>
<td>
output score tensor information. The score tensor is an
array of N floating point values between 0 and 1 representing
probability that a class was detected. Use ClassificationTensorMd to
calibrate score.
</td>
</tr><tr>
<td>
`output_number_md`
</td>
<td>
output number of detections tensor information. This
tensor is an integer value of N.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A MetadataWriter object.
</td>
</tr>

</table>



<h3 id="get_metadata_json"><code>get_metadata_json</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_writer.py#L183-L194">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_metadata_json() -> str
</code></pre>

Gets the generated JSON metadata string before populated into model.

This method returns the metadata buffer before populated into the model.
More fields could be filled by MetadataPopulator, such as
min_parser_version. Use get_populated_metadata_json() if you want to get the
final metadata string.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The generated JSON metadata string before populated into model.
</td>
</tr>

</table>



<h3 id="get_populated_metadata_json"><code>get_populated_metadata_json</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_writer.py#L196-L207">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_populated_metadata_json() -> str
</code></pre>

Gets the generated JSON metadata string after populated into model.

More fields could be filled by MetadataPopulator, such as
min_parser_version. Use get_metadata_json() if you want to get the
original metadata string.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The generated JSON metadata string after populated into model.
</td>
</tr>

</table>



<h3 id="populate"><code>populate</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_writer.py#L164-L181">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>populate() -> bytearray
</code></pre>

Populates the metadata and label file to the model file.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A new model buffer with the metadata and associated files.
</td>
</tr>

</table>
